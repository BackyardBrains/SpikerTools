"""Explicit spike previews and new annotated NWB copies; never edit a source."""

import hashlib
import json
import math
import os
import re
import stat
import uuid
from array import array
from pathlib import Path

import numpy as np

from .experiment import AnalysisError, _number, _text

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_ADDED_BYTES = 8 * 1024 * 1024
MAX_DETECT_SAMPLES = 20_000_000
MAX_SPIKES = 100_000
MAX_CANDIDATES = 1_000_000
SIGMA_DIVISOR = 0.6744897501960817
DETECT_SCHEMA = 'spikertools.spike-detection.v1'
OPERATIONS_ATTR = 'spikertools_annotation_operations_v1'
ROOT_SOURCE_ATTR = 'spikertools_root_source_sha256_v1'


def _error(code, message, **details):
    raise AnalysisError(code, message, **details)


def _settings(settings):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {'polarity', 'sigma_multiplier', 'refractory_ms'}:
        _error('invalid_detector_settings', 'Unknown spike detector settings.')
    polarity = settings.get('polarity', 'positive')
    sigma = _number(settings.get('sigma_multiplier', 4.), 'sigma_multiplier')
    refractory = _number(settings.get('refractory_ms', 1.), 'refractory_ms')
    if polarity not in ('positive', 'negative', 'both') or not 1 <= sigma <= 20 or not .1 <= refractory <= 100:
        _error('invalid_detector_settings', 'Use positive/negative/both polarity, 1–20 sigma, and 0.1–100 ms separation.')
    return dict(polarity=polarity, sigma_multiplier=sigma, refractory_ms=refractory)


def _data_hash(values):
    digest = hashlib.sha256()
    for start in range(0, len(values), 65_536):
        digest.update(np.asarray(values[start:start + 65_536], dtype='<f8').tobytes())
    return digest.hexdigest()


def detect_spikes(data, *, sample_rate, channel_id, unit='volts', settings=None):
    """Preview median/MAD threshold excursions with explicit refractory suppression.

    Strongest absolute excursion wins within the requested separation; ties keep
    the earliest sample. No filtering, clustering or isolated-neuron claim.
    """
    config = _settings(settings)
    rate = _number(sample_rate, 'sample_rate')
    channel_id = _text(channel_id, 'channel_id')
    if not 0 < rate <= 1_000_000:
        _error('bounds', 'Sample rate must be positive and at most 1 MHz.')
    values = np.asarray(data)
    if values.ndim != 1 or values.dtype.kind not in 'iuf' or not 2 <= len(values) <= MAX_DETECT_SAMPLES:
        _error('bounds', 'Detection requires 2–20000000 selected-channel samples.')
    factors = {'volts': 1., 'V': 1., 'mV': .001, 'uV': .000001}
    if unit not in factors:
        _error('incompatible_units', 'Detection requires calibrated voltage samples.')
    values = np.asarray(values, dtype=np.float64)
    if factors[unit] != 1:
        values = values * factors[unit]
    if not np.isfinite(values).all():
        _error('nonfinite_signal', 'The selected channel contains nonfinite samples; review the recording before detection.')
    scratch = values.copy()
    baseline = float(np.median(scratch, overwrite_input=True))
    with np.errstate(over='ignore', invalid='ignore'):
        scratch -= baseline
        np.abs(scratch, out=scratch)
        sigma = float(np.median(scratch, overwrite_input=True)) / SIGMA_DIVISOR
    del scratch
    if not math.isfinite(baseline) or not math.isfinite(sigma):
        _error('nonfinite_signal', 'Signal scaling exceeds finite detector precision.')
    if sigma <= 0:
        _error('zero_noise', 'Estimated noise is zero; a 4-sigma threshold cannot be estimated from this channel.')
    low, high = baseline - config['sigma_multiplier'] * sigma, baseline + config['sigma_multiplier'] * sigma
    if not math.isfinite(low) or not math.isfinite(high):
        _error('nonfinite_signal', 'The physical threshold exceeds finite detector precision.')
    candidate_indices, candidate_heights = array('q'), array('d')
    active, best_index, best_height = 0, -1, 0.

    def finish():
        if best_index >= 0:
            candidate_indices.append(best_index)
            candidate_heights.append(best_height)
            if len(candidate_indices) > MAX_CANDIDATES:
                _error('spike_limit', 'Too many threshold excursions; increase the threshold.')

    for first in range(0, len(values), 65_536):
        chunk = values[first:first + 65_536]
        labels = np.zeros(len(chunk), dtype=np.int8)
        if config['polarity'] in ('positive', 'both'):
            labels[chunk > high] = 1
        if config['polarity'] in ('negative', 'both'):
            labels[chunk < low] = -1
        edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
        for start, stop in zip(edges[:-1], edges[1:]):
            label = int(labels[start])
            if label != active:
                finish()
                active, best_index, best_height = label, -1, 0.
            if label:
                segment = chunk[start:stop]
                at = int(np.argmax(segment) if label > 0 else np.argmin(segment)) + int(start)
                height = abs(float(chunk[at]) - baseline)
                if best_index < 0 or height > best_height:
                    best_index, best_height = first + at, height
    finish()
    candidates = np.frombuffer(candidate_indices, dtype=np.int64)
    heights = np.frombuffer(candidate_heights, dtype=np.float64)
    separation = max(1, math.ceil(config['refractory_ms'] * rate / 1000))
    suppressed = np.zeros(len(candidates), dtype=bool)
    retained = []
    for index in np.argsort(-heights, kind='stable'):
        if suppressed[index]:
            continue
        peak = int(candidates[index])
        retained.append(peak)
        if len(retained) > MAX_SPIKES:
            _error('spike_limit', 'More than 100000 spikes were detected; increase threshold or minimum separation.')
        left = np.searchsorted(candidates, peak - separation + 1, side='left')
        right = np.searchsorted(candidates, peak + separation, side='left')
        suppressed[left:right] = True
    retained.sort()
    peaks = np.asarray(retained, dtype=np.int64)
    middle = retained[0] if retained else len(values) // 2
    begin = max(0, min(len(values) - min(2000, len(values)), middle - 1000))
    end = min(len(values), begin + 2000)
    shown = peaks[(peaks >= begin) & (peaks < end)]
    return dict(schema_version=DETECT_SCHEMA, settings=config, channel_id=channel_id,
                sample_rate=rate, unit='volts', sample_count=len(values), baseline=baseline,
                noise_sigma=sigma, threshold_positive=high, threshold_negative=low,
                spike_count=len(retained), sample_indices=retained, spike_times=(peaks / rate).tolist(),
                data_sha256=_data_hash(values),
                preview=dict(time=(np.arange(begin, end) / rate).tolist(), signal=values[begin:end].tolist(),
                             peak_times=(shown / rate).tolist(), peak_values=values[shown].tolist(),
                             start_seconds=begin / rate, end_seconds=end / rate),
                provenance=dict(method='median_mad_threshold_v1', version='1', sigma_divisor=SIGMA_DIVISOR,
                                channel_id=channel_id, unit='volts', sample_rate=rate, settings=config,
                                refractory_samples=separation, nonfinite=False, saturation='not_assessed'))


def _decode(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _sha_file(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _preflight(file, changed_paths):
    """Reject hidden IO, aliasing and unsupported references before making a copy."""
    import h5py
    addresses, nodes, expanded, text_bytes = set(), 0, 0, 0
    known_refs = {('/units/spike_times_index', 'target'): '/units/spike_times',
                  ('/units/electrodes_index', 'target'): '/units/electrodes'}

    def references(values, owner, attribute=None):
        for value in np.asarray(values, dtype=object).reshape(-1):
            if isinstance(value, h5py.RegionReference):
                _error('unsupported_layout', 'Region references cannot be safely preserved by annotation.')
            if isinstance(value, h5py.Reference) and value:
                try:
                    target = file[value].name
                except (KeyError, ValueError):
                    _error('unsupported_layout', 'The source contains a dangling object reference.')
                if target in changed_paths and known_refs.get((owner, attribute)) != target:
                    _error('unsupported_layout', 'An unknown reference points to data that annotation must extend.', path=owner)

    def visit(group):
        nonlocal nodes, expanded, text_bytes
        for key in group:
            link = group.get(key, getlink=True)
            if isinstance(link, h5py.SoftLink):
                # Both the native writer and PyNWB use this standard metadata
                # link. Resolve only its hard-linked canonical target; never
                # follow arbitrary paths, link chains, cycles or external IO.
                if (key != 'device' or not re.fullmatch(r'/general/extracellular_ephys/[^/]+', group.name)
                        or _decode(group.attrs.get('neurodata_type', '')) != 'ElectrodeGroup'
                        or not re.fullmatch(r'/general/devices/[^/]+', link.path)):
                    _error('unsupported_layout', 'Unsupported internal HDF5 soft link.')
                target = file
                for component in link.path.split('/')[1:]:
                    if component in ('.', '..') or not isinstance(target.get(component, getlink=True), h5py.HardLink):
                        _error('unsupported_layout', 'Device links require a direct internal hard-linked target.')
                    target = target[component]
                if (not isinstance(target, h5py.Group) or
                        _decode(target.attrs.get('neurodata_type', '')) != 'Device'):
                    _error('unsupported_layout', 'Device link does not reference a Device.')
                continue
            if not isinstance(link, h5py.HardLink):
                _error('unsupported_layout', 'External HDF5 links are not supported.')
            obj = group[key]
            address = h5py.h5o.get_info(obj.id).addr
            if address in addresses:
                _error('unsupported_layout', 'Aliased or cyclic HDF5 hard links are not supported.')
            addresses.add(address)
            nodes += 1
            if nodes > 20_000:
                _error('bounds', 'The source has too many HDF5 objects.')
            check(obj)
            if isinstance(obj, h5py.Group):
                visit(obj)

    def check(obj):
        nonlocal expanded, text_bytes
        for name, value in obj.attrs.items():
            references(value, obj.name, name)
            if isinstance(value, (str, bytes)):
                text_bytes += len(value.encode('utf-8') if isinstance(value, str) else value)
            else:
                array_value = np.asarray(value)
                if array_value.nbytes > MAX_ADDED_BYTES:
                    _error('bounds', 'An HDF5 attribute is too large.')
                if array_value.dtype.kind in 'OSU':
                    text_bytes += sum(len(_decode(v).encode('utf-8')) for v in array_value.reshape(-1)
                                      if not isinstance(v, h5py.Reference))
        if isinstance(obj, h5py.Dataset):
            if obj.is_virtual or obj.external:
                _error('unsupported_layout', 'Virtual or external-storage datasets are not supported.')
            expanded += obj.size * max(1, obj.dtype.itemsize)
            if expanded > 512 * 1024 * 1024:
                _error('bounds', 'Expanded HDF5 data exceeds 512 MiB.')
            if h5py.check_dtype(ref=obj.dtype) is not None:
                if obj.size > 1_000_000:
                    _error('bounds', 'Too many HDF5 references.')
                references(obj[()], obj.name)
            elif obj.dtype.hasobject:
                if h5py.check_string_dtype(obj.dtype) is None:
                    _error('unsupported_layout', 'Unsupported variable-length dataset.')
                if obj.ndim > 1 or obj.size > 100_000:
                    _error('bounds', 'Too many variable-length metadata strings.')
                chunks = [obj[()]] if obj.ndim == 0 else (obj[i:i + 128] for i in range(0, len(obj), 128))
                for chunk in chunks:
                    for value in np.asarray(chunk, dtype=object).reshape(-1):
                        length = len(_decode(value).encode('utf-8'))
                        if length > 1024 * 1024:
                            _error('bounds', 'A metadata string exceeds 1 MiB.')
                        text_bytes += length
            elif obj.dtype.fields and any(h5py.check_dtype(ref=dtype[0]) is not None for dtype in obj.dtype.fields.values()):
                _error('unsupported_layout', 'Compound reference datasets are not supported.')
        if text_bytes > 16 * 1024 * 1024:
            _error('bounds', 'Encoded metadata exceeds 16 MiB.')

    addresses.add(h5py.h5o.get_info(file.id).addr)
    check(file)
    visit(file)


def _metadata(group, kind, namespace='hdmf-common', description=None):
    group.attrs['neurodata_type'] = kind
    group.attrs['namespace'] = namespace
    group.attrs['object_id'] = str(uuid.uuid4())
    if description is not None:
        group.attrs['description'] = description


def _columns(group, names):
    import h5py
    group.attrs['colnames'] = np.asarray(names, dtype=h5py.string_dtype('utf-8'))


def _replace(file, path, values, *, kind='VectorData', description='', namespace='hdmf-common'):
    import h5py
    attrs, options = {}, {}
    if path in file:
        old = file[path]
        attrs = dict(old.attrs)
        if old.compression not in (None, 'gzip', 'lzf'):
            _error('unsupported_layout', 'Unsupported compression on an annotation column.')
        if old.compression is not None:
            options.update(compression=old.compression)
            if old.compression_opts is not None:
                options['compression_opts'] = old.compression_opts
        del file[path]
    values = np.asarray(values)
    dtype = h5py.string_dtype('utf-8') if values.dtype.kind in 'OSU' else values.dtype
    if values.dtype.kind == 'S':
        values = np.asarray([_decode(v) for v in values], dtype=object)
    dataset = file.create_dataset(path, data=values, dtype=dtype, **options)
    if attrs:
        for name, value in attrs.items():
            dataset.attrs[name] = value
    else:
        _metadata(dataset, kind, namespace, description)
    return dataset


def _event_requests(file, requests):
    from .nwb import _event_definitions
    if requests is None:
        return [], set()
    if not isinstance(requests, list) or len(requests) > 64:
        _error('bounds', 'At most 64 event definitions can be named at once.')
    existing = {(d['table'], d['code']): d for d in _event_definitions(file, 64)}
    occupied = {d['shortcut']: (d['table'], d['code']) for d in existing.values() if d['shortcut']}
    seen, changes, paths = set(), [], set()
    for request in requests:
        if not isinstance(request, dict) or set(request) - {'table', 'code', 'name', 'shortcut'}:
            _error('invalid_event_name', 'Invalid event-name request.')
        table, code, name = request.get('table'), request.get('code'), request.get('name')
        if not isinstance(table, str) or not re.fullmatch(r'/events/[A-Za-z0-9_-]{1,64}', table):
            _error('unsupported_layout', 'Select an explicit safe top-level events table.')
        if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 2_147_483_647:
            _error('invalid_event_name', 'Event codes must be nonnegative int32 values.')
        if (not isinstance(name, str) or not name.strip() or len(name) > 32 or len(name.encode('utf-8')) > 128
                or any(ord(c) < 32 for c in name)):
            _error('invalid_event_name', 'Event names require 1–32 characters without control characters.')
        identity = (table, code)
        if identity in seen:
            _error('invalid_event_name', 'Duplicate table/code edit.')
        seen.add(identity)
        shortcut = request.get('shortcut')
        if shortcut is not None and (not isinstance(shortcut, str) or len(shortcut) > 1 or
                                     (shortcut and ord(shortcut) < 32)):
            _error('shortcut_conflict', 'A shortcut must be one character or empty.')
        base = table + '/meanings_tables/event_type_meanings'
        defined = base in file and code in file[base]['value'][:]
        if defined:
            old = existing[identity]['shortcut']
            if shortcut is not None and shortcut != old:
                _error('shortcut_conflict', 'Existing event shortcuts are preserved; this request conflicts.')
            shortcut = old
        else:
            shortcut = shortcut or ''
        if shortcut and shortcut in occupied and occupied[shortcut] != identity:
            _error('shortcut_conflict', 'Another event already uses this shortcut.')
        if shortcut:
            occupied[shortcut] = identity
        if table in file and 'event_type' not in file[table]:
            _error('unsupported_layout', 'Cannot add numeric definitions to an annotation-only events table.')
        if base in file:
            meanings = file[base]
            values = meanings['value'][:]
            if len(set(values.tolist())) != len(values):
                _error('unsupported_layout', 'Duplicate meanings codes.')
            if code not in values:
                if set(meanings) - {'id', 'value', 'meaning', 'shortcut'}:
                    _error('unsupported_layout', 'Unknown meanings columns cannot be extended safely.')
                paths.update(base + '/' + k for k in ('id', 'value', 'meaning', 'shortcut') if k in meanings)
            else:
                paths.add(base + '/meaning')
        changes.append(dict(table=table, code=code, name=name, shortcut=shortcut))
    if len(set(existing) | seen) > 64:
        _error('bounds', 'Annotated files may contain at most 64 event definitions.')
    return changes, paths


def _apply_event_names(file, changes):
    for change in changes:
        table_path, code = change['table'], change['code']
        if table_path not in file:
            table = file.require_group(table_path)
            _metadata(table, 'EventsTable', 'core', 'Event definitions and occurrences')
            _columns(table, ['timestamp', 'event_type'])
            _replace(file, table_path + '/id', np.array([], dtype=np.int64), kind='ElementIdentifiers')
            stamp = _replace(file, table_path + '/timestamp', np.array([], dtype=float),
                             kind='TimestampVectorData', namespace='core', description='Event times in seconds')
            stamp.attrs['unit'] = 'seconds'
            _replace(file, table_path + '/event_type', np.array([], dtype=np.int32), description='Numeric event code')
        table = file[table_path]
        base = table_path + '/meanings_tables/event_type_meanings'
        if base not in file:
            meanings = file.require_group(base)
            _metadata(meanings, 'MeaningsTable', description='Meanings for event_type')
            meanings.attrs['target'] = table['event_type'].ref
            ids, codes, names, shortcuts = [], [], [], []
        else:
            meanings = file[base]
            ids = meanings['id'][:].tolist()
            codes = meanings['value'][:].tolist()
            names = [_decode(v) for v in meanings['meaning'][:]]
            shortcuts = [_decode(v) for v in meanings['shortcut'][:]] if 'shortcut' in meanings else [''] * len(codes)
        if code in codes:
            names[codes.index(code)] = change['name']
            _replace(file, base + '/meaning', np.asarray(names, dtype=object), description='The meaning of the value.')
        else:
            ids.append(max(ids, default=-1) + 1)
            codes.append(code)
            names.append(change['name'])
            shortcuts.append(change['shortcut'])
            _replace(file, base + '/id', np.asarray(ids, dtype=np.int64), kind='ElementIdentifiers')
            _replace(file, base + '/value', np.asarray(codes, dtype=np.int64), description='The referenced event code.')
            _replace(file, base + '/meaning', np.asarray(names, dtype=object), description='The meaning of the value.')
            _replace(file, base + '/shortcut', np.asarray(shortcuts, dtype=object), description='Optional keyboard shortcut.')
            _columns(meanings, ['value', 'meaning', 'shortcut'])


def _unit_plan(file):
    allowed = {'id', 'spike_times', 'spike_times_index', 'electrodes', 'electrodes_index',
               'band', 'threshold_low', 'threshold_high', 'threshold_count', 'unit_name', 'spikertools_detector_v1'}
    if 'units' not in file:
        return set()
    table = file['units']
    if set(table) - allowed or not {'id', 'spike_times', 'spike_times_index'} <= set(table):
        _error('unsupported_layout', 'Existing units contain columns that cannot be safely extended.')
    count = len(table['id'])
    if count >= 128:
        _error('bounds', 'At most 128 saved units are supported.')
    if any(table[k].ndim != 1 for k in table):
        _error('unsupported_layout', 'Only one-dimensional supported unit columns can be extended.')
    for key in set(table) - {'spike_times', 'electrodes'}:
        if len(table[key]) != count:
            _error('unsupported_layout', 'Existing unit columns have inconsistent lengths.')
    if ('electrodes' in table) != ('electrodes_index' in table):
        _error('unsupported_layout', 'Unit electrodes require a ragged index.')
    for name in ('spike_times', 'electrodes'):
        if name in table:
            ends = table[name + '_index'][:]
            if (ends.dtype.kind not in 'iu' or np.any(np.diff(ends.astype(np.int64)) < 0)
                    or (int(ends[-1]) if count else 0) != len(table[name])):
                _error('unsupported_layout', 'Malformed unit ragged indexes.')
    ids = table['id'][:]
    if (len(set(ids.tolist())) != len(ids) or ids.dtype.kind not in 'iu'
            or (len(ids) and (np.any(ids < 0) or int(max(ids)) >= np.iinfo(np.int64).max))):
        _error('unsupported_layout', 'Units require unique integer IDs.')
    for name in ('spike_times', 'electrodes'):
        if name in table:
            index = table[name + '_index']
            if 'target' not in index.attrs or file[index.attrs['target']].id != table[name].id:
                _error('unsupported_layout', 'Unit indexes must reference their corresponding data column.')
    return {'/units/' + k for k in table}


def _validate_detection(file, detection, channel_index):
    from .nwb import _timing, _scalar
    if not isinstance(detection, dict) or detection.get('schema_version') != DETECT_SCHEMA:
        _error('invalid_detection', 'Expected a verified detector result.')
    _settings(detection.get('settings'))
    acquisitions = file.get('acquisition', {})
    candidates = [value for value in acquisitions.values()
                  if _decode(value.attrs.get('neurodata_type', '')) == 'ElectricalSeries']
    if len(candidates) != 1:
        _error('unsupported_layout', 'Annotation currently requires exactly one acquisition ElectricalSeries.')
    series = candidates[0]
    data = series['data']
    count = data.shape[1] if data.ndim == 2 else 1
    if (isinstance(channel_index, bool) or not isinstance(channel_index, int) or not 0 <= channel_index < count
            or data.ndim not in (1, 2) or data.dtype.kind not in 'iuf'):
        _error('invalid_detection', 'Choose a valid source channel.')
    start, rate = _timing(series, len(data))
    if (detection.get('channel_id') != str(channel_index) or detection.get('sample_count') != len(data)
            or detection.get('sample_rate') != rate or detection.get('unit') != 'volts'):
        _error('source_changed', 'Detector channel, sample count or clock does not match the source.')
    conversion = _scalar(data.attrs.get('conversion', 1.), 'conversion')
    offset = _scalar(data.attrs.get('offset', 0.), 'offset')
    factor = _scalar(series['channel_conversion'][channel_index], 'channel_conversion') if 'channel_conversion' in series else 1.
    digest, combined_digest = hashlib.sha256(), hashlib.sha256()
    for first in range(0, len(data), 65_536):
        values = np.asarray(data[first:first + 65_536] if data.ndim == 1 else data[first:first + 65_536, channel_index], dtype=float)
        sequential = values * conversion * factor + offset
        combined = values * (conversion * factor) + offset
        digest.update(np.asarray(sequential, dtype='<f8').tobytes())
        combined_digest.update(np.asarray(combined, dtype='<f8').tobytes())
    if detection.get('data_sha256') not in (digest.hexdigest(), combined_digest.hexdigest()):
        _error('source_changed', 'Detector samples do not match the calibrated source channel.')
    indices = np.asarray(detection.get('sample_indices'))
    times = np.asarray(detection.get('spike_times'))
    if (indices.ndim != 1 or times.shape != indices.shape or len(indices) > MAX_SPIKES
            or (len(indices) and indices.dtype.kind not in 'iu')
            or len(indices) != detection.get('spike_count') or not np.isfinite(times).all()
            or (len(indices) and (np.any(indices < 0) or np.any(indices >= len(data)) or np.any(np.diff(indices) <= 0)))
            or not np.array_equal(times, indices / rate)):
        _error('invalid_detection', 'Invalid or mismatched saved spike times.')
    electrodes = series['electrodes']
    if electrodes.shape != (count,) or 'table' not in electrodes.attrs:
        _error('unsupported_layout', 'ElectricalSeries electrode mapping is missing.')
    row = int(electrodes[channel_index])
    table = file[electrodes.attrs['table']]
    if not 0 <= row < len(table['id']):
        _error('unsupported_layout', 'Invalid electrode row reference.')
    if 'units/electrodes' in file:
        existing = file['units/electrodes']
        if 'table' not in existing.attrs or file[existing.attrs['table']].id != table.id:
            _error('unsupported_layout', 'Existing units and this channel must reference the same electrode table.')
    arithmetic = 'sequential_conversion_factors' if detection['data_sha256'] == digest.hexdigest() else 'combined_conversion_factors'
    return start, rate, row, table.name, arithmetic


def _append_unit(file, detection, info, operation_id, source_sha256, root_source_sha256):
    start, rate, row, electrode_path, arithmetic = info
    table = file.require_group('units')
    if 'id' not in table:
        _metadata(table, 'Units', 'core', 'Saved spike trains, including explicit threshold detections')
    old = {key: table[key][:] for key in table}
    ids = old.get('id', np.array([], dtype=np.int64))
    count = len(ids)
    identity = int(max(ids, default=-1)) + 1
    if identity > np.iinfo(np.int64).max:
        _error('bounds', 'Cannot assign another unit ID.')
    spikes = old.get('spike_times', np.array([], dtype=float))
    new_spikes = np.asarray(detection['spike_times']) + start
    spike_ends = old.get('spike_times_index', np.array([], dtype=np.uint64))
    electrodes = old.get('electrodes', np.array([], dtype=np.int64))
    electrode_ends = old.get('electrodes_index', np.zeros(count, dtype=np.uint64))
    provenance = dict(detection['provenance'], operation_id=operation_id, source_sha256=source_sha256,
                      root_source_sha256=root_source_sha256,
                      calibration_evaluation=arithmetic,
                      baseline=detection['baseline'], noise_sigma=detection['noise_sigma'],
                      threshold_positive=detection['threshold_positive'], threshold_negative=detection['threshold_negative'],
                      data_sha256=detection['data_sha256'], starting_time=start, electrode_row=row,
                      native_band_configuration='unavailable')
    _replace(file, '/units/id', np.r_[ids, identity].astype(np.int64), kind='ElementIdentifiers')
    _replace(file, '/units/spike_times', np.r_[spikes, new_spikes], description='Spike times in absolute NWB seconds')
    _replace(file, '/units/spike_times_index', np.r_[spike_ends, len(spikes) + len(new_spikes)].astype(np.uint64), kind='VectorIndex')
    _replace(file, '/units/electrodes', np.r_[electrodes, row].astype(np.int64), kind='DynamicTableRegion')
    _replace(file, '/units/electrodes_index', np.r_[electrode_ends, len(electrodes) + 1].astype(np.uint64), kind='VectorIndex')
    table['spike_times_index'].attrs['target'] = table['spike_times'].ref
    table['electrodes_index'].attrs['target'] = table['electrodes'].ref
    table['electrodes'].attrs['table'] = file[electrode_path].ref
    defaults = {'band': 0, 'threshold_count': 0, 'threshold_low': np.nan, 'threshold_high': np.nan}
    for key, default in defaults.items():
        if key in old:
            if key in ('threshold_low', 'threshold_high') and old[key].dtype.kind != 'f':
                _error('unsupported_layout', 'Native threshold columns cannot represent missing configuration.')
            _replace(file, '/units/' + key, np.r_[old[key], default].astype(old[key].dtype))
    names = [_decode(v) for v in old.get('unit_name', [''] * count)] + ['Detected spikes']
    _replace(file, '/units/unit_name', np.asarray(names, dtype=object), description='Train name')
    annotations = [_decode(v) for v in old.get('spikertools_detector_v1', [''] * count)]
    annotations.append(json.dumps(provenance, allow_nan=False, sort_keys=True, separators=(',', ':')))
    _replace(file, '/units/spikertools_detector_v1', np.asarray(annotations, dtype=object), description='Threshold detector provenance; no isolated-unit claim')
    columns = [_decode(v) for v in table.attrs.get('colnames', [])]
    for key in ('spike_times', 'electrodes', 'unit_name', 'spikertools_detector_v1'):
        if key not in columns:
            columns.append(key)
    _columns(table, columns)
    return str(identity)


def write_annotated_copy(source_path, output_path, *, detection=None, channel_index=None,
                         event_names=None, expected_source_sha256=None, operation_id):
    """Create a private new file; preserve every unmodified source object.

    Unknown references into changed datasets, unsupported unit columns and HDF5
    links fail explicitly. The application owns permission checks and publication.
    """
    import h5py
    from .nwb import read_nwb
    source_path, output_path = Path(source_path), Path(output_path)
    if not isinstance(operation_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', operation_id):
        _error('invalid_input', 'operation_id must be a bounded identifier.')
    try:
        source_stat = source_path.lstat()
    except OSError:
        _error('invalid_source', 'The source file is unavailable.')
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > MAX_FILE_BYTES:
        _error('bounds', 'Source must be a regular NWB file of at most 100 MiB.')
    if output_path.exists() or output_path.is_symlink():
        _error('output_exists', 'The output path already exists; never overwrite an annotated copy.')
    source_hash = _sha_file(source_path)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        _error('source_changed', 'The source file changed since preview.')
    with h5py.File(source_path, 'r') as source:
        # First reject linked/aliased layouts before traversing event requests.
        _preflight(source, set())
        try:
            operations = json.loads(_decode(source.attrs.get(OPERATIONS_ATTR, '[]')))
        except ValueError:
            _error('unsupported_layout', 'Invalid existing annotation operation metadata.')
        if not isinstance(operations, list) or any(not isinstance(v, str) for v in operations) or len(operations) >= 64:
            _error('bounds', 'Annotation operation history is unsupported or full.')
        if operation_id in operations:
            _error('duplicate_operation', 'This operation is already recorded in the source.')
        root_source_hash = _decode(source.attrs.get(ROOT_SOURCE_ATTR, source_hash))
        if not re.fullmatch(r'[0-9a-f]{64}', root_source_hash):
            _error('unsupported_layout', 'Invalid original-source lineage metadata.')
        changes, paths = _event_requests(source, event_names)
        if detection is None and not changes:
            _error('invalid_input', 'Choose spike detection or at least one event name.')
        info = None
        if detection is not None:
            if detection.get('spike_count', 0):
                paths |= _unit_plan(source)
            info = _validate_detection(source, detection, channel_index)
        _preflight(source, paths)
    created = False
    try:
        # O_EXCL plus no-follow checks protect output races. Only our own partial
        # output is removed if validation fails; the source is never writable.
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, 'wb') as target, open(source_path, 'rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                target.write(chunk)
        if _sha_file(output_path) != source_hash or _sha_file(source_path) != source_hash:
            _error('source_changed', 'The source changed while its copy was prepared.')
        with h5py.File(output_path, 'r+') as target:
            new_unit = (_append_unit(target, detection, info, operation_id, source_hash, root_source_hash)
                        if detection is not None and detection['spike_count'] else None)
            _apply_event_names(target, changes)
            target.attrs[OPERATIONS_ATTR] = json.dumps(operations + [operation_id], separators=(',', ':'))
            target.attrs[ROOT_SOURCE_ATTR] = root_source_hash
            summary = dict(operation_id=operation_id, source_sha256=source_hash,
                           root_source_sha256=root_source_hash, event_names=changes)
            if detection is not None:
                summary['detection'] = {key: detection[key] for key in (
                    'settings', 'channel_id', 'sample_rate', 'unit', 'sample_count', 'baseline',
                    'noise_sigma', 'threshold_positive', 'threshold_negative', 'spike_count', 'data_sha256', 'provenance')}
            target.attrs['spikertools_last_annotation_v1'] = json.dumps(summary, allow_nan=False, sort_keys=True)
        size = output_path.stat().st_size
        if size > MAX_FILE_BYTES or size - source_stat.st_size > MAX_ADDED_BYTES:
            _error('bounds', 'Annotated copy exceeds the 100 MiB total or 8 MiB addition limit.')
        # Reopen through the supported reader, with bounded annotations and no
        # signal materialization. Native contract readback is tested separately.
        read_nwb(output_path, load_data=False, max_event_rows=100_000, max_spike_times=10_000_000, max_units=128)
        if _sha_file(source_path) != source_hash:
            _error('source_changed', 'The source changed while its annotations were prepared.')
        return dict(schema_version='spikertools.annotated-copy.v1', operation_id=operation_id,
                    source_sha256=source_hash, sha256=_sha_file(output_path), bytes=size,
                    root_source_sha256=root_source_hash,
                    source_bytes=source_stat.st_size, added_bytes=size - source_stat.st_size,
                    new_unit_id=new_unit, event_names=changes)
    except FileExistsError:
        _error('output_exists', 'The output path already exists.')
    except Exception:
        if created:
            output_path.unlink(missing_ok=True)
        raise
