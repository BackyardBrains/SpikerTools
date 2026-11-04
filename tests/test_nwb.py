"""Numerical compatibility tests; every recording is generated, not customer data."""

import builtins
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

pytest.importorskip('pynwb')
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hdmf.common import MeaningsTable
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.event import EventsTable
from pynwb.file import Subject
from scipy.io import wavfile
from scipy.signal import welch

from spikertools import Session


def write_recording(path, *, timing='rate', names=('Left', 'Right'), series_names=('ElectricalSeries1',), mono=False,
                    subject_id=None, session_id=None, display=None):
    nwb = NWBFile('Synthetic experiment', 'test-identifier', datetime(2026, 1, 2, tzinfo=timezone.utc))
    if subject_id is not None:
        nwb.subject = Subject(subject_id=subject_id)
    if session_id is not None:
        nwb.session_id = session_id
    device = nwb.create_device(name='Synthetic device')
    group = nwb.create_electrode_group(name='electrodes', description='Synthetic electrodes', location='test', device=device)
    nwb.add_electrode_column('label', 'Channel name')
    for row, label in enumerate(names):
        nwb.add_electrode(id=100 + row, location='test', group=group, label=label)
    if display is not None:
        nwb.add_electrode_column('byb_display_v1', 'Optional BYB display metadata', data=display)
    rate, start, count = 1024., 5., 8192
    raw = np.round(1000 * np.sin(2 * np.pi * 64 * np.arange(count) / rate)).astype(np.int16)
    data = raw if mono else np.column_stack([raw, -raw])
    channel_rows = [1] if mono else [1, 0]
    for name in series_names:
        region = nwb.create_electrode_table_region(channel_rows, 'Deliberately reordered electrodes')
        times = dict(starting_time=start, rate=rate)
        if timing == 'timestamps':
            times = dict(timestamps=start + np.arange(count) / rate)
        nwb.add_acquisition(ElectricalSeries(name=name, data=data, electrodes=region,
                            conversion=.002, channel_conversion=np.array([2.] if mono else [2., .5]),
                            offset=1., **times))
    markers = EventsTable(name='markers', description='Synthetic markers')
    for column in ('event_type', 'source', 'channel', 'deleted'):
        markers.add_column(column, column)
    # Deliberately unsorted; the third marker is deleted.
    for ts, deleted in [(7., 0), (6., 0), (6.5, 1)]:
        markers.add_row(timestamp=ts, duration=.1, event_type=7, source=2, channel=0, deleted=deleted)
    for column, value, label in [('event_type', 7, 'Touch'), ('source', 2, 'keyboard')]:
        meanings = MeaningsTable(target=markers[column])
        meanings.add_row(value=value, meaning=label)
        markers.add_meanings_table(meanings)
    nwb.add_events_table(markers)
    for column in ('unit_name', 'band', 'threshold_low', 'threshold_high', 'threshold_count'):
        nwb.add_unit_column(column, column)
    nwb.add_unit(id=42, spike_times=[6.1, 7.1], electrodes=[1], unit_name='Fast unit', band=0,
                 threshold_low=-100., threshold_high=-10., threshold_count=2)
    nwb.add_unit(id=43, spike_times=[], electrodes=[0], unit_name='', band=1,
                 threshold_low=10., threshold_high=100., threshold_count=1)
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwb)
    return data


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / 'experiment.nwb'
    raw = write_recording(path)
    return path, raw


@pytest.mark.parametrize('timing', ['rate', 'timestamps'])
@pytest.mark.parametrize('mono', [True, False])
def test_signal_calibration_clock_names_and_psd(tmp_path, timing, mono):
    path = tmp_path / 'experiment.NWB'
    raw = write_recording(path, timing=timing, mono=mono)
    session = Session(wav_file_path=path)
    expected = raw.astype(float) * .002 * (2. if mono else np.array([2., .5])) + 1.
    np.testing.assert_allclose(session.data, expected)
    assert session.starting_time == 5.
    assert session.sample_rate == 1024.
    assert session.channels.names() == (['Right'] if mono else ['Right', 'Left'])
    channel = session.channels[0]
    assert channel.unit == 'volts'
    assert channel.metadata['electrode_row'] == 1
    assert channel.metadata['electrode_id'] == 101  # ID != row index.
    np.testing.assert_allclose(channel.time, np.arange(8192) / 1024.)
    frequency, power = welch(channel.data, fs=session.sample_rate, nperseg=1024)
    assert frequency[np.argmax(power)] == 64.
    assert np.sum(power) == pytest.approx(8., rel=.001)  # A=4 V => A²/2.
    assert session.datetime == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert session.metadata['identifier'] == 'test-identifier'
    # Everything remains usable after the NWB handle has closed.
    assert len(channel.filter(ftype='lp', cutoff=100).data) == 8192


def test_events_and_units_preserve_identity_and_time(recording):
    session = Session(recording[0])
    event = session.events['Touch']
    assert event.timestamps == [1., 2.]
    assert [r['timestamp'] for r in event.occurrences] == [6., 7.]
    assert event.occurrences[0]['source_name'] == 'keyboard'
    assert event.occurrences[0]['channel'] == 0
    assert event.occurrences[0]['duration'] == .1
    assert event.metadata['event_type'] == 7
    neuron, empty = session.neurons
    assert neuron.name == 'Fast unit'
    np.testing.assert_allclose(neuron.timestamps, [1.1, 2.1])
    assert neuron.unit_id == 42
    assert neuron.electrode_rows == [1]
    assert neuron.channel_numbers == [0]
    assert empty.channel_numbers == [1]
    assert empty.timestamps == []
    assert empty.name == 'Unit 43'
    assert neuron.thresh_low == -100.
    assert neuron.thresh_high == -10.
    assert neuron.threshold_unit == 'ADC counts'
    assert neuron.metadata['threshold_count'] == 2


def test_native_aqnwb_starting_time_shape_and_version(recording):
    path, _ = recording
    with h5py.File(path, 'r+') as f:
        f.attrs['nwb_version'] = '2.9.0'
        group = f['acquisition/ElectricalSeries1']
        attributes = dict(group['starting_time'].attrs)
        del group['starting_time']
        ds = group.create_dataset('starting_time', data=[5.])
        ds.attrs.update(attributes)
    session = Session(path)
    assert session.sample_rate == 1024.
    assert session.events['Touch'].timestamps == [1., 2.]
    assert session.neurons[0].unit_id == 42


def test_duplicate_labels_are_addressable(tmp_path):
    path = tmp_path / 'duplicates.nwb'
    write_recording(path, names=('Same', 'Same'))
    session = Session(path)
    assert session.channels.names() == ['Same', 'Same [channel 1]']
    assert session.channels['Same [channel 1]'].metadata['label'] == 'Same'


def test_event_tables_annotations_and_code_fallback(recording):
    path, _ = recording
    # Use schema-defined tables, then change them in the usual PyNWB export path.
    output = path.with_name('events.nwb')
    with NWBHDF5IO(path, 'r') as io:
        nwb = io.read()
        other = EventsTable(name='annotations', description='Other marker source')
        other.add_column('annotation', 'Text')
        other.add_row(timestamp=6., annotation='Touch')
        other.add_row(timestamp=7., annotation='_a_marker_not_a_neuron')
        nwb.add_events_table(other)
        codes = EventsTable(name='codes', description='No meanings')
        codes.add_column('event_type', 'Code')
        codes.add_row(timestamp=5.5, event_type=99)
        nwb.add_events_table(codes)
        with NWBHDF5IO(output, 'w') as dest:
            dest.export(src_io=io, nwbfile=nwb)
    session = Session(output)
    touch = [event for event in session.events if event.metadata['label'] == 'Touch']
    assert len(touch) == 2
    assert touch[0].name != touch[1].name
    assert session.events['99'].timestamps == [.5]
    assert session.events['_a_marker_not_a_neuron'].timestamps == [2.]
    assert len(session.neurons) == 2


def test_no_events_units_or_labels(recording):
    path, _ = recording
    with h5py.File(path, 'r+') as f:
        del f['events']
        del f['units']
        electrodes = f['general/extracellular_ephys/electrodes']
        del electrodes['label']
        electrodes.attrs['colnames'] = [n for n in electrodes.attrs['colnames'] if n != 'label']
    session = Session(path)
    assert len(session.events) == 0
    assert session.neurons == []
    assert session.channels.names() == ['Channel 0', 'Channel 1']


def test_multiple_series_require_selection(tmp_path):
    path = tmp_path / 'many.nwb'
    write_recording(path, series_names=('first', 'second'))
    with pytest.raises(ValueError, match='series_name'):
        Session(path)
    assert Session(path, series_name='second').metadata['series_name'] == 'second'
    with pytest.raises(ValueError, match='Available'):
        Session(path, series_name='missing')


@pytest.mark.parametrize('fault', ['gap', 'duplicate', 'nan', 'reverse'])
def test_irregular_timestamps_are_rejected(tmp_path, fault):
    path = tmp_path / 'bad.nwb'
    write_recording(path, timing='timestamps')
    with h5py.File(path, 'r+') as f:
        ts = f['acquisition/ElectricalSeries1/timestamps']
        if fault == 'gap':
            ts[7000:] += .5  # Validate beyond the first chunk/pair.
        elif fault == 'duplicate':
            ts[7000] = ts[6999]
        elif fault == 'nan':
            ts[7000] = np.nan
        else:
            ts[:] = ts[:][::-1]
    with pytest.raises(ValueError, match='timestamps'):
        Session(path)


@pytest.mark.parametrize('rate', [0., -1., np.inf, np.nan])
def test_invalid_rate_rejected(recording, rate):
    with h5py.File(recording[0], 'r+') as f:
        f['acquisition/ElectricalSeries1/starting_time'].attrs['rate'] = rate
    with pytest.raises(ValueError):
        Session(recording[0])


def test_invalid_channel_mapping_rejected(recording):
    with h5py.File(recording[0], 'r+') as f:
        f['acquisition/ElectricalSeries1/electrodes'][0] = 99
    with pytest.raises(ValueError, match='electrodes'):
        Session(recording[0])


def test_sidecar_rejected_for_nwb(recording):
    with pytest.raises(ValueError, match='sidecar'):
        Session(recording[0], 'events.txt')


def test_nwb_dependency_is_optional(tmp_path, recording):
    original_import = builtins.__import__
    def no_pynwb(name, *args, **kwargs):
        if name == 'pynwb':
            raise ImportError('deliberately unavailable')
        return original_import(name, *args, **kwargs)
    path = tmp_path / 'legacy.wav'
    wavfile.write(path, 1024, np.arange(100, dtype=np.int16))
    with patch('builtins.__import__', side_effect=no_pynwb):
        assert Session(path).format == 'wav'
        with pytest.raises(ImportError, match=r'spikertools\[nwb\]'):
            Session(recording[0])


def test_legacy_entry_points_and_values(tmp_path):
    from spikertools.core import Session as CoreSession
    from spikertools.models import Session as ModelsSession
    assert Session is CoreSession is ModelsSession
    path = tmp_path / 'BYB_2022-01-13_13.18.29.wav'
    raw = np.arange(60, dtype=np.int16).reshape(30, 2)
    wavfile.write(path, 10000, raw)
    path.with_name(path.stem + '-events.txt').write_text('1,0.1\n_ch0_neuron0,0.2\n')
    session = ModelsSession(wav_file_path=path)
    np.testing.assert_array_equal(session.data, raw)
    assert session.data.dtype == np.int16
    assert session.events['1'].timestamps == [.1]
    assert session.neurons[0].timestamps == [.2]
    assert session.datetime == datetime(2022, 1, 13, 13, 18, 29)
    with pytest.raises(ValueError, match='only available for NWB'):
        Session(path, series_name='anything')


def test_erp_values_timing_and_voltage_axis(recording):
    session = Session(recording[0])
    # 6.1 - 5 is slightly less than 1.1 in binary: nearest sample matters.
    session.events.create_event('fractional', [6.1 - 5.])
    with patch('matplotlib.pyplot.show'):
        session.plots.plot_erp(events='fractional', epoch_window=(-.125, .125), show=True)
    ax = plt.gca()
    line = ax.lines[0]
    center = round(1.1 * 1024)
    np.testing.assert_allclose(line.get_ydata(), session.channels[0].data[center - 128:center + 128])
    np.testing.assert_allclose(line.get_xdata(), np.arange(-128, 128) / 1024.)
    assert ax.get_ylabel() == 'Amplitude (volts)'
    plt.close('all')


def test_waveform_window_uses_actual_sample_times(recording):
    session = Session(recording[0])
    with patch('matplotlib.pyplot.show'):
        session.plots.plot_channels(time_window=(7.5, 9), show=True)
    line = plt.gca().lines[0]
    np.testing.assert_array_equal(line.get_ydata(), session.channels[0].data[7680:])
    np.testing.assert_array_equal(line.get_xdata(), np.arange(7680, 8192) / 1024.)
    plt.close('all')


def test_calibrated_power_axis(recording):
    session = Session(recording[0])
    with patch('matplotlib.pyplot.show'):
        session.plots.plot_average_power(events=list(session.events), freq_range=(0, 100), show=True)
    assert plt.gca().get_ylabel() == 'Power spectral density (dB re 1 V²/Hz)'
    line = plt.gca().lines[0]
    assert line.get_xdata()[np.argmax(line.get_ydata())] == 64.
    plt.close('all')


@pytest.mark.parametrize('plot', ['plot_channels', 'plot_erp', 'plot_peth', 'plot_spectrogram', 'plot_average_power', 'plot_epochs'])
def test_existing_plots_save_nwb_graphs(recording, tmp_path, plot):
    session = Session(recording[0])
    path = tmp_path / (plot + '.png')
    kwargs = {'save_path': path, 'show': False}
    if plot == 'plot_average_power':
        kwargs['epoch_window'] = (0., .5)
        kwargs['events'] = list(session.events)
    if plot == 'plot_epochs':
        kwargs['event'] = session.events['Touch']
    getattr(session.plots, plot)(**kwargs)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert path.stat().st_size > 1000


def test_normalized_signal_is_not_mislabeled_volts(recording):
    session = Session(recording[0])
    session.channels[0].normalize('zscore')
    with patch('matplotlib.pyplot.show'):
        session.plots.plot_erp(show=True)
    assert plt.gca().get_ylabel() == 'Amplitude (dimensionless)'
    plt.close('all')
