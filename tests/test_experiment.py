"""Scientific and identity regressions using synthetic data only."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from spikertools import Channel, Event, Events, Experiment, Neuron
from spikertools.experiment import (AnalysisError, aggregate_results, analyze_session,
                                   event_key, session_catalogue)


def event(key='touch', times=(1., 2.), ids=None):
    return dict(key=key, label='Touch', trials=[dict(id=str(i) if ids is None else ids[i], time=t)
                                              for i, t in enumerate(times)])


def analyze(data=None, *, subject='subject-1', session='session-1', events=None, settings=None, **kwargs):
    config = dict(method='event_average', event_keys=['touch'], tmin=-.1, tmax=.2)
    config.update(settings or {})
    arguments = dict(session_id=session, subject_id=subject,
                     data=np.ones(1000) if data is None else data,
                     sample_rate=100., unit='volts', channel_id='0',
                     events=[event()] if events is None else events, settings=config)
    arguments.update(kwargs)
    return analyze_session(**arguments)


def assert_error(code, callable_, *args, **kwargs):
    with pytest.raises(AnalysisError) as error:
        callable_(*args, **kwargs)
    assert error.value.code == code
    return error.value


def test_subject_weighting_does_not_pool_unequal_trial_counts():
    one = analyze(np.ones(2000), events=[event(times=(1, 2))])
    three = analyze(np.full(2000, 3.), subject='subject-2', session='session-2',
                    events=[event(times=tuple(range(1, 11)))])
    group = aggregate_results([one, three])
    condition = group['events'][0]
    np.testing.assert_allclose(condition['mean'], 2.)  # NOT (2*1+10*3)/12.
    np.testing.assert_allclose(condition['subject_sd'], np.sqrt(2))
    assert condition['subject_count'] == 2
    assert condition['trial_count'] == 12
    assert group['weighting'] == 'equal_subject'
    json.dumps(group, allow_nan=False)


def test_same_timestamp_occurrences_are_independently_excluded_and_input_unchanged():
    data = np.arange(1000, dtype=float)
    events = [event(times=(2, 1, 1), ids=['later', 'first', 'second'])]
    original_data, original_events = data.copy(), copy.deepcopy(events)
    result = analyze(data, events=events, excluded_trial_ids={'touch': ['first']})
    trials = result['events'][0]['trials']
    assert [t['id'] for t in trials] == ['later', 'first', 'second']
    assert [t['included'] for t in trials] == [True, False, True]
    assert trials[1]['reason'] == 'excluded'
    assert result['events'][0]['trial_count'] == 2
    np.testing.assert_array_equal(data, original_data)
    assert events == original_events
    assert_error('invalid_input', analyze, excluded_trial_ids={'touch': ['absent']})
    assert_error('ambiguous_event', analyze, events=[event(ids=['same', 'same'])])


def test_distinct_raw_identities_never_merge_by_label():
    keys = [event_key('/events/markers', 1), event_key('/events/notes', 1),
            event_key('/events/notes', 1, 'Touch')]
    assert len(set(keys)) == 3
    result = analyze(events=[event(key) for key in keys], settings={'event_keys': keys})
    assert [e['key'] for e in result['events']] == keys
    assert all(e['label'] == 'Touch' for e in result['events'])


def test_edge_windows_baseline_and_nonfinite_are_disclosed():
    data = np.ones(1000)
    data[100] = np.nan
    result = analyze(data, events=[event(times=(.01, 1, 2, 9.99))])
    assert [t['reason'] for t in result['events'][0]['trials']] == ['edge_window', 'nonfinite', None, 'edge_window']
    result = analyze(events=[event(times=(.2, 2))], settings={'baseline': [-.5, -.3]})
    assert result['events'][0]['trials'][0]['reason'] == 'edge_window'
    assert result['events'][0]['trial_count'] == 1
    assert_error('no_valid_trials', analyze, events=[event(times=(.01,))])
    assert_error('no_valid_trials', analyze, excluded_trial_ids={'touch': ['0', '1']})


def test_event_average_baseline_full_samples_then_output_binning():
    # At t=2, baseline samples are 150..179, output samples 190..219.
    result = analyze(np.arange(1000.), events=[event(times=(2,))],
                     settings={'baseline': [-.5, -.2], 'max_output_points': 3})
    np.testing.assert_allclose(result['events'][0]['time'], [-.055, .045, .145])
    np.testing.assert_allclose(result['events'][0]['mean'], [30., 40., 50.])


def test_nearest_sample_alignment_is_explicit_and_half_up():
    result = analyze(np.arange(1000.), events=[event(times=(1.025,))], sample_rate=20,
                     settings={'tmin': 0, 'tmax': .1})
    np.testing.assert_allclose(result['events'][0]['mean'], [21, 22])
    assert result['events'][0]['trials'][0]['alignment_error_seconds'] == pytest.approx(.025)


def test_rms_constant_sinusoid_and_before_trial_average():
    constant = analyze(np.full(2000, -3.), settings={'method': 'rms', 'rms_window': .1})
    np.testing.assert_allclose(constant['events'][0]['mean'], 3., atol=1e-12)
    t = np.arange(4000) / 1000
    sinusoid = analyze(2 * np.sin(2 * np.pi * 50 * t), sample_rate=1000,
                       settings={'method': 'rms', 'rms_window': .1})
    np.testing.assert_allclose(sinusoid['events'][0]['mean'], np.sqrt(2), atol=1e-12)
    signal = np.ones(1000)
    signal[150:250] = -1
    opposite = analyze(signal, settings={'method': 'rms', 'rms_window': .02})
    np.testing.assert_allclose(opposite['events'][0]['mean'], 1., atol=1e-12)
    baseline = analyze(np.full(2000, 3.), settings={'method': 'rms', 'baseline': [-.5, -.2]})
    np.testing.assert_allclose(baseline['events'][0]['mean'], 0., atol=1e-12)


def test_rms_requires_padding_even_when_visible_response_is_complete():
    result = analyze(events=[event(times=(.05, 1))],
                     settings={'method': 'rms', 'tmin': 0., 'tmax': .1, 'rms_window': .2})
    assert result['events'][0]['trials'][0]['reason'] == 'edge_window'
    assert result['processing']['rms_window_samples'] == 20


@pytest.mark.parametrize('width', [1, 3, 10, 101])
def test_rms_small_signal_survives_large_artifact_outside_its_window(width):
    signal = np.full(1000, 1e-8)
    signal[110] = 1.
    result = analyze(signal, events=[event(times=(1.,))], settings={
        'method': 'rms', 'rms_window': width / 100, 'tmin': 0., 'tmax': 2.})
    # Sample150's actual window is far beyond the impulse for every chosen width.
    assert result['events'][0]['mean'][150] == pytest.approx(1e-8, rel=1e-12, abs=1e-22)
    assert result['processing']['rms_algorithm'] == 'block_prefix_suffix_hypot'


def test_rms_large_finite_values_do_not_overflow_squared_samples():
    result = analyze(np.full(1000, 1e200), settings={'method': 'rms', 'rms_window': .1})
    np.testing.assert_allclose(result['events'][0]['mean'], 1e200, rtol=1e-12)


@pytest.mark.parametrize('method,field,value', [
    ('rms', 'rms_window_samples', 999),
    ('rms', 'rms_left_samples', 999),
    ('rms', 'rms_algorithm', 'unknown'),
    ('peth', 'bin_widths', [999]),
    ('peth', 'bin_edges', [-.1, .2]),
    ('event_average', 'alignment', 'unknown'),
])
def test_corrupt_derived_processing_metadata_fails(method, field, value):
    result = peth() if method == 'peth' else analyze(settings={'method': method})
    result['processing'][field] = value
    assert_error('invalid_result', aggregate_results, [result])


def test_corrupt_grid_and_source_units_fail_even_when_individually_plausible():
    result = analyze(settings={'max_output_points': 2})
    result['events'][0]['time'] = [-.09, .19]
    assert_error('invalid_result', aggregate_results, [result])
    result = analyze()
    result['processing']['source_unit'] = 'dimensionless'
    assert_error('invalid_result', aggregate_results, [result])


def peth(**kwargs):
    args = dict(settings={'method': 'peth', 'tmin': -.25, 'tmax': .3, 'bin_size': .2},
                events=[event(times=(1,))], spike_times=[.75, .8, .95, 1.15, 1.3],
                spike_channel_ids=['0'], spike_unit_id='42')
    args.update(kwargs)
    return analyze(**args)


def test_peth_known_rate_partial_final_bin_and_half_open_boundaries():
    result = peth()
    np.testing.assert_allclose(result['processing']['bin_widths'], [.2, .2, .15])
    # .75 and .8 in first bin; .95 in second; 1.15 in third; 1.3 excluded.
    np.testing.assert_allclose(result['events'][0]['mean'], [10, 5, 1 / .15])
    assert result['unit'] == 'spikes/s'
    assert result['processing']['spike_unit_id'] == '42'
    assert result['events'][0]['trial_count'] == 1
    empty = peth(spike_times=[])
    np.testing.assert_allclose(empty['events'][0]['mean'], 0)


def test_peth_exclusion_uses_actual_retained_trial_denominator_and_duration_only():
    result = peth(events=[event(times=(1, 2))], excluded_trial_ids={'touch': ['1']})
    np.testing.assert_allclose(result['events'][0]['mean'], [10, 5, 1 / .15])
    direct = analyze_session(session_id='s', subject_id='p', data=None, sample_rate=100,
        unit='volts', channel_id='0', events=[event(times=(1,))], settings=result['settings'],
        spike_times=[.75, .8, .95, 1.15, 1.3], spike_channel_ids=['0'], spike_unit_id='42', duration_seconds=10)
    assert direct['events'][0]['mean'] == result['events'][0]['mean']
    assert direct['events'][0]['trial_count'] == result['events'][0]['trial_count'] == 1
    assert_error('missing_spikes', peth, spike_times=None)
    assert_error('spike_channel_mismatch', peth, spike_channel_ids=['1'])
    assert_error('invalid_input', peth, spike_times=[10.])


def test_per_event_missing_contributors_are_counted_explicitly():
    one = analyze(events=[event(), event('second')], settings={'event_keys': ['touch', 'second']})
    two = analyze(subject='p2', session='s2', events=[event(), event('second')],
                  settings={'event_keys': ['touch', 'second']}, excluded_trial_ids={'touch': ['0', '1']})
    result = aggregate_results([one, two])
    assert result['subject_count'] == 2
    assert result['events'][0]['subject_count'] == 1
    assert result['events'][1]['subject_count'] == 2
    assert result['events'][0]['subject_sd'] is None
    assert result['events'][0]['excluded_subjects'] == [dict(subject_id='p2', session_id='s2',
        reason='no_valid_trials', excluded_trial_count=2, invalid_trial_count=0)]


def test_grid_units_calibration_and_duplicate_subject_validation():
    volts = analyze(np.full(1000, .002))
    millivolts = analyze(np.full(1000, 2.), unit='mV', subject='p2', session='s2')
    result = aggregate_results([volts, millivolts])
    np.testing.assert_allclose(result['events'][0]['mean'], .002)
    assert_error('duplicate_subject', aggregate_results, [volts, analyze(session='s2')])
    assert_error('duplicate_session', aggregate_results, [volts, analyze(subject='p2')])
    assert_error('incompatible_units', aggregate_results,
                 [volts, analyze(unit='ADC counts', subject='p2', session='s2')])
    assert_error('incompatible_grid', aggregate_results,
                 [volts, analyze(sample_rate=50, subject='p2', session='s2')])
    differing = peth(sample_rate=50, subject='p2', session='s2')
    assert aggregate_results([peth(), differing])['subject_count'] == 2


@pytest.mark.parametrize('kwargs,code', [
    ({'settings': {'event_keys': ['absent']}}, 'missing_event'),
    ({'sample_rate': float('nan')}, 'invalid_input'),
    ({'sample_rate': 0}, 'bounds'),
    ({'sample_rate': 1_000_001}, 'bounds'),
    ({'subject_id': ''}, 'invalid_input'),
    ({'unit': 'unknown'}, 'incompatible_units'),
    ({'settings': {'tmax': float('inf')}}, 'invalid_input'),
    ({'settings': {'tmax': 121}}, 'bounds'),
    ({'settings': {'max_output_points': 10001}}, 'bounds'),
    ({'settings': {'max_output_points': True}}, 'invalid_input'),
    ({'settings': {'unrecognized': 1}}, 'invalid_input'),
    ({'settings': {'baseline': [0, 0]}}, 'bounds'),
    ({'duration_seconds': 4}, 'invalid_input'),
    ({'events': [event(times=(float('nan'),))]}, 'invalid_input'),
    ({'events': [event(), event()]}, 'ambiguous_event'),
])
def test_finite_inputs_and_explicit_limits(kwargs, code):
    assert_error(code, analyze, **kwargs)


def test_tiny_bin_and_oversized_work_fail_before_allocating():
    assert_error('bounds', peth, settings={'method': 'peth', 'bin_size': 5e-324})
    assert_error('bounds', analyze, settings={'tmin': -120, 'tmax': 0}, sample_rate=1_000_000)
    assert_error('bounds', analyze, events=[event(times=[2] * 100_001)])


def test_aggregation_rejects_malformed_or_nonfinite_cache():
    for mutate in [lambda r: r['events'][0]['mean'].__setitem__(0, float('nan')),
                   lambda r: r['events'][0].__setitem__('trial_count', 1),
                   lambda r: r['events'][0]['trials'][0].__setitem__('reason', 'excluded'),
                   lambda r: r['processing'].__setitem__('unexpected', float('inf'))]:
        result = analyze()
        mutate(result)
        with pytest.raises(AnalysisError):
            aggregate_results([result])


def session(subject='p1', identity='s1', value=1., path=None):
    channel = Channel(np.full(1000, value), 100., number=0)
    return SimpleNamespace(subject_id=subject, session_id=identity, channels=[channel],
                           events=Events([Event('Touch', [1., 2.])]), neurons=[], unit='volts',
                           starting_time=0., file_path=path)


def test_experiment_collection_exclusions_and_duplicate_source():
    one, two = session(), session('p2', 's2', 3.)
    experiment = Experiment([one, two])
    assert len(experiment) == 2 and list(experiment) == [one, two] and experiment[1] is two
    key = session_catalogue(one)['events'][0]['key']
    config = dict(method='event_average', event_keys=[key], tmin=-.1, tmax=.1)
    result = experiment.analyze(config, excluded_session_ids=['s2'])
    assert result['subject_count'] == 1
    assert result['excluded_session_ids'] == ['s2']
    np.testing.assert_allclose(result['events'][0]['mean'], 1)
    result = experiment.analyze(config, excluded_trials={'s1': {key: ['row:0']}})
    assert result['events'][0]['trial_count'] == 3
    assert_error('no_valid_trials', experiment.analyze, config, excluded_session_ids=['s1', 's2'])
    assert_error('duplicate_source', Experiment, [one, one], subject_ids=['a', 'b'], session_ids=['a', 'b'])
    one.file_path, two.file_path = '/tmp/shared.nwb', '/tmp/../tmp/shared.nwb'
    assert_error('duplicate_source', Experiment, [one, two])


def test_experiment_unit_selection_requires_known_channel_linkage():
    one = session()
    neuron = Neuron('Unit', [1.01])
    neuron.channel_numbers, neuron.unit_id = [0], 42
    one.neurons = [neuron]
    key = session_catalogue(one)['events'][0]['key']
    config = dict(method='peth', event_keys=[key], tmin=-.1, tmax=.1)
    result = Experiment([one]).analyze(config)
    assert result['sessions'][0]['processing']['spike_unit_id'] == '42'
    duplicate = Neuron('Another', [2.01])
    duplicate.channel_numbers, duplicate.unit_id = [0], 43
    one.neurons.append(duplicate)
    assert_error('missing_spikes', Experiment([one]).analyze, config)
    assert Experiment([one]).analyze(config, neuron=1)['sessions'][0]['processing']['spike_unit_id'] == '43'
