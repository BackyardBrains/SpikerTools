"""Read local NWB recordings into the existing SpikerTools models.

PyNWB supplies the installed core schema, including EventsTable. File-provided
namespaces are deliberately not loaded: early SpikeRecorder writers cache core
2.9 while already writing the standard 2.10 EventsTable layout.
"""

import json

import numpy as np

from .models import Channel, Channels, Event, Events, Neuron

CLOCK_CHUNK_SAMPLES = 65_536


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _scalar(value, name):
    values = np.asarray(value)
    if values.size != 1 or values.dtype.kind not in 'iuf':
        raise ValueError('{} must be a finite scalar.'.format(name))
    result = float(values.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError('{} must be a finite scalar.'.format(name))
    return result


def _vector(group, name, length=None, default=None):
    if name not in group:
        if default is not None:
            return np.full(length, default)
        raise ValueError('Missing NWB column: {}/{}'.format(group.name, name))
    values = np.asarray(group[name][:])
    if values.ndim != 1 or (length is not None and len(values) != length):
        raise ValueError('Invalid NWB column shape: {}/{}'.format(group.name, name))
    return values


def _unique_name(name, used, suffix):
    candidate = name
    index = 2
    while candidate in used:
        candidate = '{} [{}{}]'.format(name, suffix, '' if index == 2 else ':' + str(index))
        index += 1
    used.add(candidate)
    return candidate


def _display_metadata(value):
    """Read only bounded optional BYB display hints; never apply display filters."""
    text = _text(value)
    if len(text) > 512 or len(text.encode('utf-8')) > 512:
        return {}
    try:
        display = json.loads(text)
    except (ValueError, RecursionError):
        return {}
    if not isinstance(display, dict) or display.get('v') != 1:
        return {}
    result = {}
    prep = display.get('prep')
    if isinstance(prep, dict) and prep.get('kind') in ('neuron', 'plant', 'heart', 'eye', 'brain', 'muscle', 'custom'):
        result['signal_role'] = prep['kind']
    filters = display.get('filters')
    if isinstance(filters, dict):
        result['recorded_filters'] = {
            key: value for key, value in filters.items()
            if key in ('highPassHz', 'lowPassHz', 'notchHz') and
            (value is None or (isinstance(value, (float, int)) and not isinstance(value, bool)
                               and np.isfinite(value) and 0 < value <= 999999))}
    return result


def _timing(group, samples):
    if 'timestamps' in group:
        if 'starting_time' in group:
            raise ValueError('NWB signal has conflicting timestamps and starting_time.')
        dataset = group['timestamps']
        if dataset.shape != (samples,) or dataset.dtype.kind not in 'iuf' or samples < 2:
            raise ValueError('NWB timestamps must contain at least two increasing finite times.')
        start = _scalar(dataset[0], 'first timestamp')
        end = _scalar(dataset[-1], 'last timestamp')
        interval = (end - start) / (samples - 1)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError('NWB timestamps must contain at least two increasing finite times.')
        rate = 1 / interval
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('NWB sample rate must be finite and positive.')
        # Validate the full clock in bounded chunks, including adjacent chunk
        # boundaries. Metadata-only reads must not allocate the full time axis.
        previous = None
        for first in range(0, samples, CLOCK_CHUNK_SAMPLES):
            last = min(samples, first + CLOCK_CHUNK_SAMPLES)
            times = np.asarray(dataset[first:last], dtype=float)
            if (not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
                    or (previous is not None and times[0] <= previous)):
                raise ValueError('NWB timestamps must contain at least two increasing finite times.')
            expected = start + np.arange(first, last) * interval
            # No resampling: existing filters, FFTs and epochs need a regular clock.
            if not np.allclose(times, expected, rtol=0, atol=max(1e-9, interval * 0.001)):
                raise ValueError('Irregular NWB timestamps are unsupported; resample explicitly first.')
            previous = times[-1]
        return float(start), float(rate)
    if 'starting_time' not in group:
        raise ValueError('NWB signal has neither timestamps nor starting_time/rate.')
    dataset = group['starting_time']
    # AqNWB can write a one-element dataset instead of a scalar.
    start = _scalar(dataset[()], 'starting_time')
    rate = _scalar(dataset.attrs.get('rate'), 'rate')
    if rate <= 0:
        raise ValueError('NWB sample rate must be positive.')
    return start, rate


def _meanings(table, column):
    path = 'meanings_tables/{}_meanings'.format(column)
    if path not in table:
        return {}
    meanings = table[path]
    values = _vector(meanings, 'value')
    names = _vector(meanings, 'meaning', len(values))
    if len(set(values.tolist())) != len(values):
        raise ValueError('Duplicate values in NWB meanings table: ' + meanings.name)
    return dict(zip(values.tolist(), map(_text, names)))


def _events(file, start):
    """Keep distinct table/type identities, even when display names collide."""
    events, names = [], set()
    for table in file.get('events', {}).values():
        times = _vector(table, 'timestamp').astype(float)
        if not np.all(np.isfinite(times)):
            raise ValueError('NWB event timestamps must be finite.')
        count = len(times)
        deleted = _vector(table, 'deleted', count, 0)
        codes = _vector(table, 'event_type', count) if 'event_type' in table else None
        annotations = _vector(table, 'annotation', count) if 'annotation' in table else None
        meanings = _meanings(table, 'event_type')
        sources = _meanings(table, 'source')
        columns = {key: _vector(table, key, count) for key in
                   ('id', 'source', 'channel', 'duration') if key in table}
        grouped = {}
        for i in np.argsort(times, kind='stable'):
            if deleted[i] != 0:
                continue
            code = codes[i].item() if codes is not None else None
            annotation = _text(annotations[i]) if annotations is not None else None
            key = (code, annotation)
            if key not in grouped:
                label = annotation or meanings.get(code) or (str(code) if code is not None else table.name.rsplit('/', 1)[-1])
                name = _unique_name(label, names, table.name.rsplit('/', 1)[-1] + ':' + str(code))
                event = Event(name)
                event.metadata = {'table': table.name, 'event_type': code,
                                  'annotation': annotation, 'label': label}
                event.occurrences = []
                grouped[key] = event
                events.append(event)
            event = grouped[key]
            event.timestamps.append(float(times[i] - start))
            occurrence = {'timestamp': float(times[i]), 'row_index': int(i)}
            for column, values in columns.items():
                value = values[i].item()
                occurrence[column] = value
            if 'source' in occurrence:
                occurrence['source_name'] = sources.get(occurrence['source'])
            event.occurrences.append(occurrence)
    return Events(events)


def _neurons(nwb, start, channel_rows):
    units = nwb.units
    if units is None:
        return []
    neurons, used = [], set()
    for row, unit_id in enumerate(units.id[:]):
        times = np.asarray(units['spike_times'][row], dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError('NWB unit spike_times must be finite.')
        label = _text(units['unit_name'][row]) if 'unit_name' in units.colnames else ''
        name = _unique_name(label or 'Unit {}'.format(unit_id), used, 'unit ' + str(unit_id))
        neuron = Neuron(name, timestamps=sorted((times - start).tolist()))
        neuron.unit_id = int(unit_id)
        neuron.electrode_rows = []
        if 'electrodes' in units.colnames:
            # VectorIndex -> DynamicTableRegion. Read row indices, not electrode IDs.
            index = units['electrodes']
            first = 0 if row == 0 else int(index.data[row - 1])
            last = int(index.data[row])
            neuron.electrode_rows = [int(v) for v in index.target.data[first:last]]
            if any(v < 0 or v >= len(index.target.table) for v in neuron.electrode_rows):
                raise ValueError('NWB unit references an invalid electrode row.')
        neuron.channel_numbers = [i for i, electrode in enumerate(channel_rows)
                                  if electrode in neuron.electrode_rows]
        neuron.metadata = {}
        for field in ('band', 'threshold_count', 'threshold_low', 'threshold_high'):
            if field in units.colnames:
                neuron.metadata[field] = np.asarray(units[field][row]).item()
        # SpikeRecorder stores these custom thresholds in converter counts.
        # Preserve that meaning; they are not calibrated signal amplitudes.
        neuron.thresh_low = neuron.metadata.get('threshold_low')
        neuron.thresh_high = neuron.metadata.get('threshold_high')
        neuron.threshold_unit = 'ADC counts' if neuron.thresh_low is not None or neuron.thresh_high is not None else None
        neurons.append(neuron)
    return neurons


def read_nwb(path, series_name=None, *, load_data=True):
    """Read NWB models, optionally without any signal samples or channel clocks.

    ``load_data=False`` is a metadata adapter for bounded file workers. It returns
    the same dict with ``data=None`` and empty Channel.data/time arrays, alongside
    ``sample_count`` and ``duration_seconds``. Events and saved spike trains remain
    available. The caller can read one calibrated channel with its own bounded
    reader and pass it to ``analyze_session``. This function performs no remote IO.
    """
    if not isinstance(load_data, bool):
        raise ValueError('load_data must be a bool.')
    try:
        import h5py
        from pynwb import NWBHDF5IO
        from pynwb.ecephys import ElectricalSeries, SpikeEventSeries
    except ImportError as error:
        raise ImportError('NWB support requires Python 3.10+ and spikertools[nwb]. '
                          'Install with: pip install "spikertools[nwb]"') from error

    with h5py.File(path, 'r') as file, NWBHDF5IO(file=file, mode='r', load_namespaces=False) as io:
        nwb = io.read()
        candidates = {name: series for name, series in nwb.acquisition.items()
                      if isinstance(series, ElectricalSeries) and not isinstance(series, SpikeEventSeries)}
        if series_name is not None:
            if series_name not in candidates:
                raise ValueError('No acquisition ElectricalSeries named {!r}. Available: {}'.format(series_name, sorted(candidates)))
            series = candidates[series_name]
        elif len(candidates) == 1:
            series = next(iter(candidates.values()))
        elif 'ElectricalSeries1' in candidates:
            series = candidates['ElectricalSeries1']
        else:
            raise ValueError('Select one acquisition ElectricalSeries with series_name. Available: {}'.format(sorted(candidates)))
        dataset = series.data
        if dataset.ndim not in (1, 2) or dataset.dtype.kind not in 'iuf' or not dataset.size:
            raise ValueError('NWB signal must be a nonempty numeric time-by-channel array.')
        samples = dataset.shape[0]
        count = dataset.shape[1] if dataset.ndim == 2 else 1
        start, rate = _timing(file['acquisition'][series.name], samples)
        conversion = _scalar(series.conversion, 'conversion')
        offset = _scalar(series.offset, 'offset')
        factors = np.ones(count) if series.channel_conversion is None else np.asarray(series.channel_conversion[:], dtype=float)
        if factors.shape != (count,) or not np.all(np.isfinite(factors * conversion)):
            raise ValueError('NWB channel_conversion must have one finite value per channel.')
        data = None
        if load_data:
            data = np.asarray(dataset[:], dtype=np.float64)
            data *= conversion
            data *= factors if data.ndim == 2 else factors[0]
            data += offset

        electrodes = series.electrodes.table
        rows = np.asarray(series.electrodes.data[:])
        if rows.shape != (count,) or rows.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= len(electrodes)):
            raise ValueError('NWB electrodes must reference one valid table row per signal channel.')
        channels, names = Channels([]), set()
        for number, row in enumerate(rows):
            label = _text(electrodes['label'][row]) if 'label' in electrodes.colnames else ''
            name = _unique_name(label or 'Channel {}'.format(number), names, 'channel ' + str(number))
            values = np.empty(0) if data is None else (data if data.ndim == 1 else data[:, number])
            channel = Channel(values, rate, number, name)
            channel.unit = 'volts'
            channel.metadata = {'electrode_row': int(row), 'electrode_id': int(electrodes.id[row]),
                                'label': label, 'conversion': conversion,
                                'channel_conversion': float(factors[number]), 'offset': offset}
            for field in ('location', 'group_name'):
                if field in electrodes.colnames:
                    channel.metadata[field] = _text(electrodes[field][row])
            if 'byb_display_v1' in electrodes.colnames:
                channel.metadata.update(_display_metadata(electrodes['byb_display_v1'][row]))
            channels.append(channel)
        metadata = {key: getattr(nwb, key) for key in
                    ('identifier', 'session_description', 'session_start_time', 'timestamps_reference_time')}
        metadata.update(nwb_version=_text(file.attrs['nwb_version']), series_name=series.name,
                        description=series.description, comments=series.comments,
                        session_id=nwb.session_id,
                        subject_id=nwb.subject.subject_id if nwb.subject is not None else None)
        return dict(format='nwb', sample_rate=rate, data=data, data_loaded=load_data, starting_time=start,
                    sample_count=samples, duration_seconds=samples / rate,
                    session_id=metadata['session_id'], subject_id=metadata['subject_id'],
                    unit='volts', channels=channels, events=_events(file, start),
                    neurons=_neurons(nwb, start, rows.tolist()), metadata=metadata,
                    datetime=nwb.session_start_time)
