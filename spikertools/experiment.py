"""Bounded, event-aligned numerical analysis, shared by notebooks and file workers.

Version 1 deliberately requires one session per explicit subject. No resampling,
spike detection, inferential statistics or source-file edits happen here.
"""

import copy
import json
import math
import os
from collections.abc import Mapping

import numpy as np


SESSION_SCHEMA = 'spikertools.session-analysis.v1'
EXPERIMENT_SCHEMA = 'spikertools.experiment-analysis.v1'
MAX_SUBJECTS = 24
MAX_EVENTS = 64
MAX_TRIALS = 100_000
MAX_SAMPLES = 50_000_000
MAX_SPIKES = 10_000_000
MAX_EPOCH_SAMPLES = 2_000_000
MAX_WORK_SAMPLES = 200_000_000
MAX_OUTPUT_POINTS = 10_000
MAX_RESULT_VALUES = 10_000_000


class AnalysisError(ValueError):
    """An input or analysis failure with a stable code for a caller's UI."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def _fail(code, message, **details):
    raise AnalysisError(code, message, **details)


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 2048:
        _fail('invalid_input', '{} must be a nonempty string of at most 2048 characters.'.format(name))
    return value


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        _fail('invalid_input', '{} must be a finite number.'.format(name))
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError):
        _fail('invalid_input', '{} must be a finite number.'.format(name))
    if not math.isfinite(result):
        _fail('invalid_input', '{} must be a finite number.'.format(name))
    return result


def _integer(value, name, lower, upper):
    number = _number(value, name)
    if number != int(number) or not lower <= number <= upper:
        _fail('bounds', '{} must be an integer between {} and {}.'.format(name, lower, upper))
    return int(number)


def event_key(table, event_type=None, annotation=None):
    """Stable NWB identity: compact JSON [table path, type code, annotation]."""
    _text(table, 'table')
    if isinstance(event_type, np.generic):
        event_type = event_type.item()
    if event_type is not None and not isinstance(event_type, (str, int, float)):
        _fail('invalid_input', 'event_type must be a string, number or null.')
    if isinstance(event_type, (float, int)):
        _number(event_type, 'event_type')
    if annotation is not None and not isinstance(annotation, str):
        _fail('invalid_input', 'annotation must be a string or null.')
    return _text(json.dumps([table, event_type, annotation], ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False), 'event key')


def _settings(settings):
    if not isinstance(settings, Mapping):
        _fail('invalid_input', 'settings must be an object.')
    allowed = {'method', 'event_keys', 'tmin', 'tmax', 'baseline', 'bin_size',
               'rms_window', 'max_output_points'}
    if set(settings) - allowed:
        _fail('invalid_input', 'Unknown analysis settings.', fields=sorted(set(settings) - allowed))
    method = settings.get('method')
    if method not in ('event_average', 'rms', 'peth'):
        _fail('invalid_input', 'Select event_average, rms or peth.')
    keys = settings.get('event_keys')
    if not isinstance(keys, (list, tuple)) or not 1 <= len(keys) <= MAX_EVENTS:
        _fail('bounds', 'Select between 1 and 64 event keys.')
    keys = [_text(key, 'event key') for key in keys]
    if len(set(keys)) != len(keys):
        _fail('ambiguous_event', 'Selected event keys must be unique.')
    tmin, tmax = _number(settings.get('tmin'), 'tmin'), _number(settings.get('tmax'), 'tmax')
    if not -120 <= tmin < tmax <= 120 or tmax - tmin > 120:
        _fail('bounds', 'Response must be a positive window at most 120 seconds within [-120,120].')
    baseline = settings.get('baseline')
    if baseline is not None:
        if method == 'peth' or not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
            _fail('invalid_input', 'baseline requires two endpoints and is unavailable for PETH.')
        baseline = [_number(v, 'baseline endpoint') for v in baseline]
        if not -120 <= baseline[0] < baseline[1] <= 120 or baseline[1] - baseline[0] > 120:
            _fail('bounds', 'Baseline must be a positive window at most 120 seconds within [-120,120].')
    result = dict(method=method, event_keys=keys, tmin=tmin, tmax=tmax, baseline=baseline,
                  max_output_points=_integer(settings.get('max_output_points', 2000),
                                             'max_output_points', 1, MAX_OUTPUT_POINTS))
    # Accept both optional controls from a shared form, but do not silently ignore invalid values.
    for key, default in [('bin_size', .05), ('rms_window', .05)]:
        value = _number(settings.get(key, default), key)
        if not 0 < value <= 120:
            _fail('bounds', '{} must be positive and at most 120 seconds.'.format(key))
        result[key] = value
    return result


def _events(events, keys, excluded):
    if not isinstance(events, (list, tuple)) or len(events) > MAX_EVENTS:
        _fail('bounds', 'events must be a list of at most 64 event types.')
    catalogue, total = {}, 0
    for event in events:
        if not isinstance(event, Mapping):
            _fail('invalid_input', 'Each event must be an object.')
        key = _text(event.get('key'), 'event key')
        if key in catalogue:
            _fail('ambiguous_event', 'Duplicate event key.', event_key=key)
        label = _text(event.get('label'), 'event label')
        trials = event.get('trials')
        if not isinstance(trials, (list, tuple)):
            _fail('invalid_input', 'Event trials must be a list.')
        total += len(trials)
        if total > MAX_TRIALS:
            _fail('bounds', 'A session may contain at most 100000 trial occurrences.')
        rows, seen = [], set()
        for trial in trials:
            if not isinstance(trial, Mapping):
                _fail('invalid_input', 'Each trial must be an object.')
            identity = _text(trial.get('id'), 'trial id')
            if identity in seen:
                _fail('ambiguous_event', 'Duplicate trial ID within event.', event_key=key, trial_id=identity)
            seen.add(identity)
            rows.append(dict(id=identity, time=_number(trial.get('time'), 'trial time')))
        catalogue[key] = dict(key=key, label=label, trials=rows)
    missing = [key for key in keys if key not in catalogue]
    if missing:
        _fail('missing_event', 'Selected events are absent; review the experiment selection.', event_keys=missing)
    if not isinstance(excluded, Mapping):
        _fail('invalid_input', 'excluded_trial_ids must be an object keyed by event identity.')
    exclusions = {}
    for key, ids in excluded.items():
        if key not in catalogue or not isinstance(ids, (list, tuple)):
            _fail('invalid_input', 'Unknown exclusion event or invalid trial list.', event_key=key)
        ids = [_text(v, 'excluded trial id') for v in ids]
        known = {trial['id'] for trial in catalogue[key]['trials']}
        if len(ids) != len(set(ids)) or set(ids) - known:
            _fail('invalid_input', 'Unknown or duplicate excluded trial IDs.', event_key=key)
        exclusions[key] = set(ids)
    return [catalogue[key] for key in keys], exclusions


def _array(values, name, maximum, finite=False):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        _fail('invalid_input', '{} must be a one-dimensional numeric array.'.format(name))
    if array.ndim != 1 or array.dtype.kind not in 'iuf' or len(array) > maximum:
        _fail('bounds', '{} must be a bounded one-dimensional numeric array.'.format(name))
    array = np.asarray(array, dtype=np.float64)
    if finite and not np.isfinite(array).all():
        _fail('invalid_input', '{} must contain only finite values.'.format(name))
    return array


def _offsets(start, end, rate):
    # Tolerance only removes floating arithmetic around exact sample boundaries.
    lo, hi = math.ceil(start * rate - 1e-9), math.ceil(end * rate - 1e-9)
    if not 1 <= hi - lo <= MAX_EPOCH_SAMPLES:
        _fail('bounds', 'The native epoch must contain between 1 and 2000000 samples.')
    return lo, hi


def _compress(values, cuts):
    return np.add.reduceat(values, cuts[:-1]) / np.diff(cuts)


def _rms(values, center, bounds, width):
    lo, hi = bounds
    left, right = width // 2, width - width // 2
    signal = values[center + lo - left:center + hi + right - 1]
    # Subtracting cumulative sums can lose precision for huge dynamic range; the
    # scale limits overflow while keeping the operation linear in epoch length.
    scale = np.max(np.abs(signal))
    if not np.isfinite(scale):
        return None
    if scale == 0:
        return np.zeros(hi - lo)
    squared = np.square(signal / scale)
    sums = np.concatenate(([0.], np.cumsum(squared)))
    return np.sqrt(np.maximum(0., (sums[width:] - sums[:-width]) / width)) * scale


def analyze_session(*, session_id, subject_id, data, sample_rate, unit, channel_id,
                    events, settings, excluded_trial_ids=None, spike_times=None,
                    spike_channel_ids=None, spike_unit_id=None, duration_seconds=None):
    """Compute one selected channel's event results without modifying its data.

    See README for the versioned JSON and half-open timing conventions. Spike
    timestamps, like event times, must already be relative to the first sample.
    """
    session_id, subject_id = _text(session_id, 'session_id'), _text(subject_id, 'subject_id')
    channel_id = _text(channel_id, 'channel_id')
    rate = _number(sample_rate, 'sample_rate')
    if not 0 < rate <= 1_000_000:
        _fail('bounds', 'sample_rate must be positive and at most 1000000 Hz.')
    config = _settings(settings)
    selected, exclusions = _events(events, config['event_keys'],
                                    {} if excluded_trial_ids is None else excluded_trial_ids)
    units = {'volts': ('volts', 1.), 'V': ('volts', 1.), 'mV': ('volts', .001),
             'uV': ('volts', .000001), 'dimensionless': ('dimensionless', 1.),
             'ADC counts': ('ADC counts', 1.)}
    if not isinstance(unit, str) or unit not in units:
        _fail('incompatible_units', 'Supply calibrated volts, mV, uV, dimensionless, or ADC counts.')
    output_unit, scale = units[unit]
    values = None if data is None else _array(data, 'data', MAX_SAMPLES)
    if values is not None and not len(values):
        _fail('invalid_input', 'The selected channel is empty.')
    if duration_seconds is None:
        if values is None:
            _fail('invalid_input', 'duration_seconds is required when data is absent.')
        duration = len(values) / rate
    else:
        duration = _number(duration_seconds, 'duration_seconds')
        if duration <= 0 or (values is not None and not math.isclose(duration, len(values) / rate,
                                                                     rel_tol=1e-12, abs_tol=1e-12)):
            _fail('invalid_input', 'duration_seconds must match the selected channel duration.')
    if not math.isfinite(duration) or duration <= 0:
        _fail('bounds', 'Recording duration must be finite and positive.')
    method, tmin, tmax = config['method'], config['tmin'], config['tmax']
    processing = dict(alignment='nearest_sample_half_up', interval='[start,end)',
                      output='adjacent_bin_means_after_full_sample_calculation',
                      source_unit=unit, spike_unit_id=None, spike_channel_ids=[])
    baseline_bounds = None
    if method == 'peth':
        if spike_times is None:
            _fail('missing_spikes', 'Select a saved spike train before running PETH.')
        spike_unit_id = _text(spike_unit_id, 'spike_unit_id')
        if not isinstance(spike_channel_ids, (list, tuple)) or channel_id not in spike_channel_ids:
            _fail('spike_channel_mismatch', 'The saved unit must belong to the selected channel.')
        links = [_text(v, 'spike channel id') for v in spike_channel_ids]
        spikes = _array(spike_times, 'spike_times', MAX_SPIKES, finite=True)
        if np.any(spikes < 0) or np.any(spikes >= duration):
            _fail('invalid_input', 'Spike times must be within [0,duration).')
        spikes = np.sort(spikes)
        bin_count = (tmax - tmin) / config['bin_size']
        if not math.isfinite(bin_count) or bin_count > config['max_output_points'] + 1e-12:
            _fail('bounds', 'PETH bin count exceeds max_output_points; increase bin_size.')
        bins = math.ceil(bin_count - 1e-12)
        if not 1 <= bins <= config['max_output_points']:
            _fail('bounds', 'PETH bin count exceeds max_output_points; increase bin_size.')
        edges = tmin + np.arange(bins + 1) * config['bin_size']
        edges[-1] = tmax
        widths = np.diff(edges)
        if np.any(widths <= 0):
            _fail('bounds', 'PETH bins must have positive widths.')
        time = (edges[:-1] + edges[1:]) / 2
        output_unit = 'spikes/s'
        processing.update(alignment='exact_event_time', output='counts_per_true_bin_width',
                          spike_unit_id=spike_unit_id, spike_channel_ids=links,
                          bin_edges=edges.tolist(), bin_widths=widths.tolist())
        work_per_trial = bins + 1
    else:
        if values is None:
            _fail('invalid_input', 'Signal data is required for event_average and rms.')
        bounds = _offsets(tmin, tmax, rate)
        if config['baseline'] is not None:
            baseline_bounds = _offsets(*config['baseline'], rate)
        width = max(1, int(math.floor(config['rms_window'] * rate + .5))) if method == 'rms' else 1
        if width > MAX_EPOCH_SAMPLES:
            _fail('bounds', 'RMS window exceeds 2000000 native samples.')
        left, right = width // 2, width - width // 2
        length = bounds[1] - bounds[0]
        cuts = np.linspace(0, length, min(length, config['max_output_points']) + 1, dtype=int)
        time = _compress(np.arange(*bounds, dtype=float) / rate, cuts)
        if method == 'rms':
            processing.update(rms_window_samples=width, rms_window_seconds=width / rate,
                              rms_left_samples=left, rms_right_samples=right - 1)
        work_per_trial = length + width
        if baseline_bounds is not None:
            work_per_trial += baseline_bounds[1] - baseline_bounds[0] + width
    if len(selected) * len(time) > MAX_RESULT_VALUES:
        _fail('bounds', 'Too many output values.')
    if work_per_trial * sum(len(e['trials']) for e in selected) > MAX_WORK_SAMPLES:
        _fail('bounds', 'Selected analysis exceeds the 200 million sample-work limit.')
    output, retained = [], 0
    for event in selected:
        mean, count, trials = np.zeros(len(time)), 0, []
        for source in event['trials']:
            stamp = source['time']
            trial = dict(source, included=False, reason=None, alignment_error_seconds=None)
            if source['id'] in exclusions.get(event['key'], set()):
                trial['reason'] = 'excluded'
            elif stamp + tmin < 0 or stamp + tmax > duration:
                trial['reason'] = 'edge_window'
            elif config['baseline'] is not None and (stamp + config['baseline'][0] < 0 or
                                                     stamp + config['baseline'][1] > duration):
                trial['reason'] = 'edge_window'
            else:
                if method == 'peth':
                    indices = np.searchsorted(spikes, stamp + edges, side='left')
                    response = np.diff(indices) / widths
                    trial['alignment_error_seconds'] = 0.
                else:
                    center = int(math.floor(stamp * rate + .5))
                    trial['alignment_error_seconds'] = center / rate - stamp
                    windows = [bounds] + ([] if baseline_bounds is None else [baseline_bounds])
                    if any(center + lo - left < 0 or center + hi + right - 1 > len(values)
                           for lo, hi in windows):
                        trial['reason'] = 'edge_window'
                    else:
                        if method == 'rms':
                            response = _rms(values, center, bounds, width)
                            baseline = None if baseline_bounds is None else _rms(values, center, baseline_bounds, width)
                        else:
                            response = values[center + bounds[0]:center + bounds[1]]
                            baseline = None if baseline_bounds is None else values[center + baseline_bounds[0]:center + baseline_bounds[1]]
                        if response is None or not np.isfinite(response).all() or (baseline_bounds is not None and
                                (baseline is None or not np.isfinite(baseline).all())):
                            trial['reason'] = 'nonfinite'
                        else:
                            # Never subtract from the source slice in place.
                            with np.errstate(over='ignore', invalid='ignore'):
                                adjusted = response - (np.mean(baseline) if baseline is not None else 0.)
                                response = _compress(adjusted * scale, cuts)
                if trial['reason'] is None:
                    if not np.isfinite(response).all():
                        trial['reason'] = 'nonfinite'
                    else:
                        # Stable incremental average; avoid overflow in a sum of trials.
                        count += 1
                        with np.errstate(over='ignore', invalid='ignore'):
                            mean = mean * ((count - 1) / count) + response / count
                        if not np.isfinite(mean).all():
                            _fail('invalid_input', 'Analysis overflowed; inspect signal scaling.')
                        trial['included'] = True
            trials.append(trial)
        retained += count
        output.append(dict(key=event['key'], label=event['label'], time=time.tolist(),
                           mean=mean.tolist() if count else None, trial_count=count,
                           trial_total=len(trials), trials=trials))
    if not retained:
        _fail('no_valid_trials', 'No complete, finite, included trials remain.',
              events=[dict(key=e['key'], trial_total=e['trial_total'],
                           excluded_trial_count=sum(t['reason'] == 'excluded' for t in e['trials']),
                           invalid_trial_count=sum(t['reason'] in ('edge_window', 'nonfinite') for t in e['trials']))
                      for e in output])
    return dict(schema_version=SESSION_SCHEMA, session_id=session_id, subject_id=subject_id,
                channel_id=channel_id, sample_rate=rate, duration_seconds=duration, unit=output_unit,
                settings=config, events=output, processing=processing)


def _validate_result(result):
    """Do not accept unbounded or non-finite stored results into aggregation."""
    if not isinstance(result, Mapping) or result.get('schema_version') != SESSION_SCHEMA:
        _fail('invalid_result', 'Expected a version 1 session analysis result.')
    for field in ('session_id', 'subject_id', 'channel_id', 'unit'):
        _text(result.get(field), field)
    config = _settings(result.get('settings'))
    rate = _number(result.get('sample_rate'), 'sample_rate')
    duration = _number(result.get('duration_seconds'), 'duration_seconds')
    if not 0 < rate <= 1_000_000 or duration <= 0:
        _fail('invalid_result', 'Invalid cached timing.')
    expected_unit = 'spikes/s' if config['method'] == 'peth' else None
    if ((expected_unit is not None and result['unit'] != expected_unit) or
            (expected_unit is None and result['unit'] not in ('volts', 'dimensionless', 'ADC counts'))):
        _fail('invalid_result', 'Cached output units do not match the method.')
    processing = result.get('processing')
    if not isinstance(processing, Mapping):
        _fail('invalid_result', 'Cached processing provenance is missing.')
    if config['method'] == 'peth':
        _text(processing.get('spike_unit_id'), 'spike_unit_id')
        links = processing.get('spike_channel_ids')
        if not isinstance(links, list) or result['channel_id'] not in links:
            _fail('invalid_result', 'Cached spike channel linkage is missing.')
    events = result.get('events')
    if not isinstance(events, list) or [e.get('key') for e in events if isinstance(e, Mapping)] != config['event_keys']:
        _fail('invalid_result', 'Cached event keys do not match settings.')
    total, values = 0, 0
    for event in events:
        _text(event.get('label'), 'label')
        time = _array(event.get('time'), 'time', config['max_output_points'], finite=True)
        if (not len(time) or np.any(np.diff(time) <= 0) or time[0] < config['tmin'] - 1e-9 / rate
                or time[-1] >= config['tmax'] + 1e-9 / rate):
            _fail('invalid_result', 'Invalid cached output grid.')
        count = _integer(event.get('trial_count'), 'trial_count', 0, MAX_TRIALS)
        trials = event.get('trials')
        if not isinstance(trials, list) or event.get('trial_total') != len(trials):
            _fail('invalid_result', 'Invalid cached trial catalogue.')
        total += len(trials)
        ids, included = set(), 0
        for trial in trials:
            if not isinstance(trial, Mapping):
                _fail('invalid_result', 'Invalid cached trial.')
            identity = _text(trial.get('id'), 'trial id')
            if identity in ids:
                _fail('invalid_result', 'Duplicate cached trial ID.')
            ids.add(identity)
            _number(trial.get('time'), 'trial time')
            flag, reason = trial.get('included'), trial.get('reason')
            if not isinstance(flag, bool) or reason not in (None, 'excluded', 'edge_window', 'nonfinite') or flag != (reason is None):
                _fail('invalid_result', 'Invalid cached trial inclusion.')
            if trial.get('alignment_error_seconds') is not None:
                _number(trial['alignment_error_seconds'], 'alignment error')
            included += flag
        if included != count:
            _fail('invalid_result', 'Cached included count disagrees with trial catalogue.')
        if count:
            mean = _array(event.get('mean'), 'mean', config['max_output_points'], finite=True)
            if len(mean) != len(time):
                _fail('invalid_result', 'Cached output lengths differ.')
        elif event.get('mean') is not None:
            _fail('invalid_result', 'An empty event must have a null mean.')
        values += len(time)
    if total > MAX_TRIALS or values > MAX_RESULT_VALUES:
        _fail('bounds', 'Cached session result exceeds limits.')
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        _fail('invalid_result', 'Session result must be finite JSON.')
    return config


def aggregate_results(results):
    """Equally weight subjects, each already represented by its own trial mean."""
    if not isinstance(results, (list, tuple)) or not 1 <= len(results) <= MAX_SUBJECTS:
        _fail('bounds', 'Aggregate between 1 and 24 subjects.')
    seen_subjects, seen_sessions, value_count = set(), set(), 0
    first = results[0]
    for result in results:
        config = _validate_result(result)
        if result['subject_id'] in seen_subjects:
            _fail('duplicate_subject', 'Version 1 requires one session per unique subject.')
        if result['session_id'] in seen_sessions:
            _fail('duplicate_session', 'A session cannot appear twice.')
        seen_subjects.add(result['subject_id'])
        seen_sessions.add(result['session_id'])
        if config != first['settings']:
            _fail('incompatible_grid', 'Session analysis settings must match exactly.')
        if result['unit'] != first['unit']:
            _fail('incompatible_units', 'Session output units must match.')
        if config['method'] != 'peth' and result['sample_rate'] != first['sample_rate']:
            _fail('incompatible_grid', 'Sample rates differ; explicit resampling is required before analysis.')
        value_count += sum(len(e['time']) for e in result['events'])
        if value_count > MAX_RESULT_VALUES:
            _fail('bounds', 'Combined results exceed the output-value limit.')
    output = []
    for index, event in enumerate(first['events']):
        traces, subjects, excluded, trial_count = [], [], [], 0
        for result in results:
            other = result['events'][index]
            if other['time'] != event['time']:
                _fail('incompatible_grid', 'Event-relative output grids must match exactly.')
            if other['trial_count']:
                traces.append(other['mean'])
                subjects.append(result['subject_id'])
                trial_count += other['trial_count']
            else:
                excluded.append(dict(subject_id=result['subject_id'], session_id=result['session_id'],
                                     reason='no_valid_trials',
                                     excluded_trial_count=sum(t['reason'] == 'excluded' for t in other['trials']),
                                     invalid_trial_count=sum(t['reason'] in ('edge_window', 'nonfinite') for t in other['trials'])))
        if not traces:
            _fail('no_valid_trials', 'An event has no contributing subjects.', event_key=event['key'])
        array = np.asarray(traces)
        # Scaling before summation protects large, otherwise finite signal values.
        mean = np.sum(array / len(traces), axis=0)
        with np.errstate(over='ignore', invalid='ignore'):
            sd = np.std(array, axis=0, ddof=1) if len(traces) > 1 else None
        if not np.isfinite(mean).all() or (sd is not None and not np.isfinite(sd).all()):
            _fail('invalid_result', 'Group calculation overflowed; inspect signal scaling.')
        output.append(dict(key=event['key'], label=event['label'], time=list(event['time']),
                           mean=mean.tolist(), subject_sd=sd.tolist() if sd is not None else None,
                           subject_count=len(traces), trial_count=trial_count,
                           subject_ids=subjects, excluded_subjects=excluded))
    return dict(schema_version=EXPERIMENT_SCHEMA, settings=copy.deepcopy(first['settings']),
                unit=first['unit'], weighting='equal_subject', subject_count=len(results),
                sessions=copy.deepcopy(list(results)), events=output)


def session_catalogue(session):
    """Expose channel/unit/event identities, preserving NWB source row identity.

    A mutated NWB Event.timestamps list cannot safely reuse its original row IDs;
    reject it instead of attaching exclusions to different occurrences.
    """
    events = []
    for event in session.events:
        meta = getattr(event, 'metadata', {})
        key = event_key(meta.get('table', 'legacy'), meta.get('event_type'),
                        meta.get('annotation') if 'table' in meta else event.name)
        occurrences = getattr(event, 'occurrences', None)
        if occurrences is not None:
            if len(occurrences) != len(event.timestamps) or any(
                    not math.isclose(float(row['timestamp']) - session.starting_time, float(stamp),
                                     rel_tol=0, abs_tol=1e-12)
                    for row, stamp in zip(occurrences, event.timestamps)):
                _fail('ambiguous_event', 'NWB events were edited without updating occurrence identities; reload the original session.')
            trials = [dict(id='row:{}'.format(row['row_index']), time=float(stamp))
                      for row, stamp in zip(occurrences, event.timestamps)]
        else:
            trials = [dict(id='row:{}'.format(i), time=float(stamp)) for i, stamp in enumerate(event.timestamps)]
        events.append(dict(key=key, label=event.name, trials=trials))
    if events:
        _events(events, [e['key'] for e in events], {})
    return dict(session_id=getattr(session, 'session_id', None),
                subject_id=getattr(session, 'subject_id', None),
                data_loaded=getattr(session, 'data_loaded', True),
                sample_count=getattr(session, 'sample_count', None),
                duration_seconds=getattr(session, 'duration_seconds', None), events=events,
                channels=[dict(id=str(c.number), name=c.name, sample_rate=float(c.sample_rate),
                               unit=getattr(c, 'unit', session.unit),
                               calibration={key: getattr(c, 'metadata', {}).get(key)
                                            for key in ('conversion', 'channel_conversion', 'offset')},
                               signal_role=getattr(c, 'metadata', {}).get('signal_role'),
                               recorded_filters=getattr(c, 'metadata', {}).get('recorded_filters'))
                          for c in session.channels],
                neurons=[dict(id=str(getattr(n, 'unit_id', i)), name=n.name,
                              channel_ids=[str(c) for c in getattr(n, 'channel_numbers', [])],
                              spike_count=len(n.timestamps)) for i, n in enumerate(session.neurons)])


class Experiment:
    """An ordered collection of independent subjects' Session objects.

    IDs may be supplied explicitly or come from NWB metadata. Every method
    delegates to analyze_session/aggregate_results, including trial exclusions.
    """

    def __init__(self, sessions, *, subject_ids=None, session_ids=None):
        self.sessions = tuple(sessions)
        if not 1 <= len(self.sessions) <= MAX_SUBJECTS:
            _fail('bounds', 'An experiment requires between 1 and 24 sessions.')
        self.subject_ids = tuple(subject_ids) if subject_ids is not None else tuple(
            getattr(s, 'subject_id', None) for s in self.sessions)
        self.session_ids = tuple(session_ids) if session_ids is not None else tuple(
            getattr(s, 'session_id', None) for s in self.sessions)
        if len(self.subject_ids) != len(self.sessions) or len(self.session_ids) != len(self.sessions):
            _fail('invalid_input', 'Supply one subject ID and session ID per session.')
        for name, identities in [('subject_id', self.subject_ids), ('session_id', self.session_ids)]:
            for identity in identities:
                _text(identity, name)
            if len(set(identities)) != len(identities):
                _fail('duplicate_subject' if name == 'subject_id' else 'duplicate_session',
                      'Version 1 requires unique subject and session IDs.')
        sources, objects = set(), set()
        for session in self.sessions:
            path = getattr(session, 'file_path', None)
            source = os.path.realpath(path) if path is not None else None
            if id(session) in objects or (source is not None and source in sources):
                _fail('duplicate_source', 'A source recording cannot appear twice under different subject IDs.')
            objects.add(id(session))
            if source is not None:
                sources.add(source)

    def __len__(self):
        return len(self.sessions)

    def __iter__(self):
        return iter(self.sessions)

    def __getitem__(self, index):
        return self.sessions[index]

    def analyze(self, settings, *, channel=0, neuron=None, excluded_session_ids=(), excluded_trials=None):
        """Select channel/unit by position, or per-session {session_id: position}.

        Neuron selection is explicit when multiple saved units match a channel.
        excluded_trials maps session ID to {event key: [trial ID,...]}.
        """
        unknown = set(excluded_session_ids) - set(self.session_ids)
        if unknown or (excluded_trials and set(excluded_trials) - set(self.session_ids)):
            _fail('invalid_input', 'Unknown session exclusion identity.')
        settings = _settings(settings)
        results = []
        for session, subject_id, session_id in zip(self.sessions, self.subject_ids, self.session_ids):
            if session_id in excluded_session_ids:
                continue
            if getattr(session, 'data_loaded', True) is False:
                _fail('metadata_only', 'This session contains metadata only; load its samples or pass a selected channel to analyze_session.')
            choice = channel.get(session_id) if isinstance(channel, Mapping) else channel
            if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(session.channels):
                _fail('invalid_input', 'Select a valid channel position for each included session.')
            selected = session.channels[choice]
            catalog = session_catalogue(session)
            spike_times, links, unit_id = None, None, None
            if settings['method'] == 'peth':
                compatible = [i for i, n in enumerate(session.neurons)
                              if selected.number in getattr(n, 'channel_numbers', [])]
                pick = neuron.get(session_id) if isinstance(neuron, Mapping) else neuron
                if pick is None and len(compatible) == 1:
                    pick = compatible[0]
                if pick is None:
                    _fail('missing_spikes', 'Select one compatible saved unit for each included session.')
                if isinstance(pick, bool) or not isinstance(pick, int) or pick not in compatible:
                    _fail('spike_channel_mismatch', 'Selected saved unit does not belong to this channel.')
                selected_neuron = session.neurons[pick]
                spike_times = selected_neuron.timestamps
                links = [str(v) for v in selected_neuron.channel_numbers]
                unit_id = str(getattr(selected_neuron, 'unit_id', pick))
            results.append(analyze_session(session_id=session_id, subject_id=subject_id,
                data=selected.data, sample_rate=selected.sample_rate,
                unit=getattr(selected, 'unit', session.unit), channel_id=str(selected.number),
                events=catalog['events'], settings=settings,
                excluded_trial_ids=(excluded_trials or {}).get(session_id),
                spike_times=spike_times, spike_channel_ids=links, spike_unit_id=unit_id))
        if not results:
            _fail('no_valid_trials', 'All sessions are excluded.')
        result = aggregate_results(results)
        result['excluded_session_ids'] = [s for s in self.session_ids if s in excluded_session_ids]
        return result
