"""Exercise the NWB reader and the same numerical path used by file workers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pynwb')
import h5py

from spikertools import AnalysisError, Experiment, Session
from spikertools.experiment import event_key, session_catalogue
from spikertools.nwb import read_nwb
from test_nwb import write_recording


def test_nwb_metadata_original_rows_nonzero_clock_and_calibration(tmp_path):
    path = tmp_path / 'subject-a.nwb'
    raw = write_recording(path, subject_id='subject-a', session_id='session-a',
        display=[json.dumps({'v': 1, 'prep': {'kind': 'muscle'}, 'filters': {'highPassHz': 20}}),
                 json.dumps({'v': 1, 'prep': {'kind': 'brain'}, 'filters': {'lowPassHz': 30}})])
    session = Session(path)
    assert session.subject_id == session.metadata['subject_id'] == 'subject-a'
    assert session.session_id == session.metadata['session_id'] == 'session-a'
    catalog = session_catalogue(session)
    assert catalog['channels'][0]['signal_role'] == 'brain'  # Electrode rows are reordered.
    assert catalog['channels'][0]['recorded_filters'] == {'lowPassHz': 30}
    assert catalog['channels'][1]['signal_role'] == 'muscle'
    event = catalog['events'][0]
    assert event['key'] == event_key('/events/markers', 7)
    assert event['trials'] == [dict(id='row:1', time=1.), dict(id='row:0', time=2.)]
    settings = dict(method='event_average', event_keys=[event['key']], tmin=0., tmax=.125)
    result = Experiment([session]).analyze(settings)
    assert session.analysis_catalogue() == catalog
    assert session.analyze_events(settings) == result['sessions'][0]
    # Stored count * global conversion * selected electrode conversion + offset.
    expected = (raw[1024:1152, 0] * .004 + 1 + raw[2048:2176, 0] * .004 + 1) / 2
    np.testing.assert_allclose(result['events'][0]['mean'], expected)
    assert result['unit'] == 'volts'
    # Display filter hints did not silently filter the raw calibrated signal.
    np.testing.assert_allclose(session.channels[0].data, raw[:, 0] * .004 + 1)
    peth = Experiment([session]).analyze(dict(method='peth', event_keys=[event['key']],
                                             tmin=0, tmax=.2, bin_size=.2))
    assert peth['events'][0]['mean'] == [5.]
    assert peth['sessions'][0]['processing']['spike_unit_id'] == '42'


def test_missing_metadata_requires_explicit_subject_and_session_ids(recording):
    session = Session(recording[0])
    assert session.subject_id is None and session.session_id is None
    with pytest.raises(AnalysisError, match='subject_id'):
        Experiment([session])
    assert len(Experiment([session], subject_ids=['known-subject'], session_ids=['known-session'])) == 1


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / 'session.nwb'
    return path, write_recording(path)


def test_stale_occurrences_after_debounce_fail_clearly(recording):
    session = Session(recording[0])
    session.events[0].debounce(2)
    with pytest.raises(AnalysisError, match='occurrence identities'):
        session_catalogue(session)


def test_optional_invalid_or_oversized_display_metadata_stays_absent(tmp_path):
    path = tmp_path / 'display.nwb'
    write_recording(path, display=['{"v":1,"prep":{"kind":"unknown"}}', 'x' * 513])
    catalog = session_catalogue(Session(path))
    assert [c['signal_role'] for c in catalog['channels']] == [None, None]


@pytest.mark.parametrize('timing', ['rate', 'timestamps'])
def test_metadata_read_never_reads_signal_or_allocates_full_channel_clocks(tmp_path, monkeypatch, timing):
    path = tmp_path / 'metadata.nwb'
    write_recording(path, timing=timing, subject_id='subject', session_id='session')
    expected = Session(path)
    original_getitem = h5py.Dataset.__getitem__
    timestamp_reads = []

    def guarded_read(dataset, index, *args, **kwargs):
        if dataset.name == '/acquisition/ElectricalSeries1/data':
            pytest.fail('Metadata-only read accessed the signal dataset')
        if dataset.name == '/acquisition/ElectricalSeries1/timestamps' and isinstance(index, slice):
            assert index.start is not None and index.stop is not None
            assert index.stop - index.start <= 1024
            timestamp_reads.append(index)
        return original_getitem(dataset, index, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded_read)
    monkeypatch.setattr('spikertools.nwb.CLOCK_CHUNK_SAMPLES', 1024)
    metadata = read_nwb(path, load_data=False)
    assert metadata['data'] is None
    assert metadata['sample_count'] == 8192
    assert metadata['duration_seconds'] == 8
    assert all(len(c.data) == len(c.time) == 0 for c in metadata['channels'])
    catalogue = session_catalogue(SimpleNamespace(**metadata))
    assert metadata['data_loaded'] is False and catalogue['data_loaded'] is False
    expected_catalogue = session_catalogue(expected)
    expected_catalogue['data_loaded'] = False
    assert catalogue == expected_catalogue
    assert [n.timestamps for n in metadata['neurons']] == [n.timestamps for n in expected.neurons]
    if timing == 'timestamps':
        assert len(timestamp_reads) == 8
    with pytest.raises(AnalysisError, match='metadata only'):
        Experiment([SimpleNamespace(**metadata)]).analyze(dict(
            method='event_average', event_keys=[catalogue['events'][0]['key']], tmin=0, tmax=.1))


def test_metadata_clock_validation_catches_chunk_boundary_discontinuity(tmp_path, monkeypatch):
    path = tmp_path / 'clock.nwb'
    write_recording(path, timing='timestamps')
    with h5py.File(path, 'r+') as file:
        times = file['acquisition/ElectricalSeries1/timestamps']
        times[1024] = times[1023]
    monkeypatch.setattr('spikertools.nwb.CLOCK_CHUNK_SAMPLES', 1024)
    with pytest.raises(ValueError, match='increasing finite times'):
        read_nwb(path, load_data=False)
