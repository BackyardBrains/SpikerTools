"""Synthetic numerical and copy-preservation tests; no customer files."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from spikertools.annotations import detect_spikes, write_annotated_copy
from spikertools.experiment import AnalysisError


def signal(length=4000):
    # Dense symmetric background stays median0/MAD5 after sparse excursions.
    return np.resize(np.arange(-10., 11.), length)


def detect(data, **settings):
    return detect_spikes(data, sample_rate=1000., channel_id='0', settings=settings)


def test_positive_negative_both_baseline_and_robust_noise():
    data = signal()
    data[100:103] = [40, 60, 50]
    data[200:203] = [-50, -70, -55]
    positive = detect(data)
    assert positive['sample_indices'] == [101]
    assert detect(data, polarity='negative')['sample_indices'] == [201]
    assert detect(data, polarity='both')['sample_indices'] == [101, 201]
    assert positive['baseline'] == 0
    assert positive['noise_sigma'] == pytest.approx(5 / .6744897501960817)
    offset = detect(data + 5)
    assert offset['sample_indices'] == [101]
    assert offset['baseline'] == positive['baseline'] + 5
    assert offset['threshold_positive'] == pytest.approx(positive['threshold_positive'] + 5)
    assert positive['provenance']['saturation'] == 'not_assessed'
    json.dumps(positive, allow_nan=False)


def test_strict_threshold_extrema_ties_refractory_and_chunk_boundary():
    data = signal(140000)
    data[65534:65539] = [40, 60, 60, 50, 40]
    data[65542] = 70
    data[65547] = 70
    result = detect(data, refractory_ms=10)
    assert result['sample_indices'] == [65542]  # Strongest; earliest tied peak.
    assert detect(data, refractory_ms=1)['sample_indices'] == [65535, 65542, 65547]
    data = signal()
    data[100] = 4 * 5 / .6744897501960817
    assert detect(data)['sample_indices'] == []  # Equality is not above threshold.


def test_zero_detections_zero_noise_nonfinite_and_unchanged_input():
    data = signal()
    before = data.copy()
    result = detect(data)
    assert result['spike_count'] == 0 and result['sample_indices'] == []
    assert len(result['preview']['time']) <= 2000
    np.testing.assert_array_equal(data, before)
    for bad, code in [(np.ones(1000), 'zero_noise'), (np.r_[signal(), np.nan], 'nonfinite_signal')]:
        with pytest.raises(AnalysisError) as error:
            detect(bad)
        assert error.value.code == code
    with pytest.raises(AnalysisError):
        detect(data, sigma_multiplier=.5)


pytest.importorskip('pynwb')
import h5py
from pynwb import NWBHDF5IO, NWBFile, validate
from pynwb.ecephys import ElectricalSeries
from pynwb.event import EventsTable
from hdmf.common import MeaningsTable
from spikertools import Session
from spikertools.nwb import read_nwb


def recording(path, *, events=True, units=True, negative_conversion=False):
    nwb = NWBFile('Annotation test', 'annotate-fixture', datetime(2026, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device('device')
    group = nwb.create_electrode_group('group', 'test electrodes', 'test', device)
    nwb.add_electrode_column('label', 'Name')
    nwb.add_electrode_column('byb_event_definitions_v1', 'Native appearance')
    for name in ('Left', 'Right'):
        nwb.add_electrode(location='test', group=group, label=name,
                          byb_event_definitions_v1='{"version":1,"events":[{"code":1,"colorArgb":42}]}')
    data = signal().astype(np.int16)
    data[100:103] = [40, 60, 50]
    electrodes = nwb.create_electrode_table_region([1, 0], 'Reordered channels')
    nwb.add_acquisition(ElectricalSeries(name='ElectricalSeries1', data=np.column_stack([data, -data]),
        electrodes=electrodes, starting_time=5., rate=1000., conversion=-.001 if negative_conversion else .001,
        channel_conversion=np.array([2., 1.]), offset=.1))
    if events:
        table = EventsTable(name='markers', description='Events')
        for col in ('event_type', 'deleted', 'source', 'channel', 'annotation'):
            table.add_column(col, col)
        table.add_row(timestamp=5.2, event_type=1, deleted=0, source=2, channel=0, annotation='')
        table.add_row(timestamp=5.3, event_type=1, deleted=0, source=2, channel=0, annotation='Free text')
        table.add_row(timestamp=5.4, event_type=2, deleted=1, source=2, channel=0, annotation='')
        meanings = MeaningsTable(target=table['event_type'])
        meanings.add_column('shortcut', 'Keyboard shortcut')
        meanings.add_row(value=1, meaning='Old one', shortcut='1')
        meanings.add_row(value=2, meaning='Unused', shortcut='2')
        table.add_meanings_table(meanings)
        nwb.add_events_table(table)
    if units:
        for name in ('unit_name', 'band', 'threshold_low', 'threshold_high', 'threshold_count'):
            nwb.add_unit_column(name, name)
        nwb.add_unit(id=17, spike_times=[5.1, 5.2], electrodes=[1], unit_name='Existing',
                     band=0, threshold_low=8., threshold_high=100., threshold_count=1)
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwb)
    with h5py.File(path, 'r+') as file:
        unknown = file.require_group('unknown_extension')
        unknown.create_dataset('values', data=np.arange(10))
        unknown.attrs['description'] = 'Must survive'
        unknown.attrs['raw_reference'] = file['acquisition/ElectricalSeries1/data'].ref
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def result_for(path, polarity='positive'):
    session = Session(path)
    return detect_spikes(session.channels[0].data, sample_rate=session.sample_rate,
                         channel_id='0', settings={'polarity': polarity})


def test_copy_roundtrip_preserves_raw_unknown_events_units_and_absolute_clock(tmp_path):
    source = recording(tmp_path / 'source.nwb')
    source_hash = sha(source)
    detection = result_for(source)
    output = tmp_path / 'copy.nwb'
    receipt = write_annotated_copy(source, output, detection=detection, channel_index=0,
        expected_source_sha256=source_hash, operation_id='operation-1', event_names=[
            dict(table='/events/markers', code=1, name='Touch'),
            dict(table='/events/markers', code=3, name='New unused', shortcut='3')])
    assert sha(source) == source_hash
    assert receipt['new_unit_id'] == '18'
    assert receipt['sha256'] == sha(output)
    with h5py.File(source) as original, h5py.File(output) as copied:
        for path in ('acquisition/ElectricalSeries1/data', 'events/markers/timestamp',
                     'events/markers/event_type', 'events/markers/annotation', 'events/markers/deleted',
                     'unknown_extension/values', 'general/extracellular_ephys/electrodes/byb_event_definitions_v1'):
            np.testing.assert_array_equal(copied[path][:], original[path][:])
        assert copied[copied['unknown_extension'].attrs['raw_reference']].name == '/acquisition/ElectricalSeries1/data'
        assert copied[copied['units/spike_times_index'].attrs['target']].name == '/units/spike_times'
        np.testing.assert_array_equal(copied['units/spike_times'][:2], [5.1, 5.2])
        assert copied['units/spike_times'][2] == pytest.approx(5.101)
        assert copied['units/electrodes'][-1] == 1  # Row, not channel-index 0.
        assert np.isnan(copied['units/threshold_low'][-1])
        assert copied['units/threshold_count'][-1] == 0
    loaded = Session(output)
    assert loaded.neurons[-1].timestamps == pytest.approx([.101])
    assert loaded.neurons[-1].channel_numbers == [0]
    definitions = {(d['table'], d['code']): d for d in loaded.event_definitions}
    assert definitions['/events/markers', 1] == dict(table='/events/markers', code=1, name='Touch', shortcut='1', occurrences=2)
    assert definitions['/events/markers', 3]['occurrences'] == 0
    assert len(loaded.events) == 2  # Existing blank/free-text groups, no fabricated trial.
    assert loaded.annotation_root_source_sha256 == source_hash
    assert validate(path=output) == []


def test_absent_event_table_creates_definitions_with_no_timestamps(tmp_path):
    source = recording(tmp_path / 'source.nwb', events=False, units=False)
    output = tmp_path / 'named.nwb'
    write_annotated_copy(source, output, operation_id='names', event_names=[
        dict(table='/events/markers', code=i, name=f'Condition {i}', shortcut=str(i)) for i in range(1, 7)])
    loaded = Session(output)
    assert len(loaded.events) == 0
    assert len(loaded.event_definitions) == 6
    assert all(d['occurrences'] == 0 for d in loaded.event_definitions)
    with h5py.File(output) as file:
        assert len(file['events/markers/timestamp']) == 0
    assert validate(path=output) == []


def test_observed_code_without_meanings_gets_name_and_requested_shortcut(tmp_path):
    source = recording(tmp_path / 'source.nwb', units=False)
    with h5py.File(source, 'r+') as file:
        del file['events/markers/meanings_tables']
        # AqNWB writes fixed UTF-8 fields and a one-element starting_time.
        series = file['acquisition/ElectricalSeries1']
        attrs = dict(series['starting_time'].attrs)
        del series['starting_time']
        series.create_dataset('starting_time', data=[5.]).attrs.update(attrs)
    output = tmp_path / 'named.nwb'
    write_annotated_copy(source, output, operation_id='names', event_names=[
        dict(table='/events/markers', code=i, name=f'Condition {i}', shortcut=str(i)) for i in range(1, 7)])
    loaded = Session(output)
    assert [d['shortcut'] for d in loaded.event_definitions] == list('123456')
    assert [d['occurrences'] for d in loaded.event_definitions] == [2, 0, 0, 0, 0, 0]
    assert validate(path=output) == []


def test_negative_calibration_and_zero_detection_copy(tmp_path):
    source = recording(tmp_path / 'negative.nwb', negative_conversion=True)
    detection = result_for(source, 'negative')
    assert detection['sample_indices'] == [101]
    output = tmp_path / 'detected.nwb'
    write_annotated_copy(source, output, detection=detection, channel_index=0, operation_id='negative')
    assert Session(output).neurons[-1].timestamps == pytest.approx([.101])
    with h5py.File(source, 'r+') as file:
        file['acquisition/ElectricalSeries1/data'][100:103, 0] = [1, -1, 1]
    empty = result_for(source)
    assert empty['spike_count'] == 0
    empty_out = tmp_path / 'zero.nwb'
    receipt = write_annotated_copy(source, empty_out, detection=empty, channel_index=0, operation_id='empty')
    assert receipt['new_unit_id'] is None
    assert len(Session(empty_out).neurons) == 1


def test_shortcut_conflicts_existing_path_source_hash_and_lineage(tmp_path):
    source = recording(tmp_path / 'source.nwb')
    output = tmp_path / 'one.nwb'
    with pytest.raises(AnalysisError, match='shortcut'):
        write_annotated_copy(source, output, operation_id='conflict', event_names=[
            dict(table='/events/markers', code=1, name='Changed', shortcut='z')])
    assert not output.exists()
    with pytest.raises(AnalysisError, match='shortcut'):
        write_annotated_copy(source, output, operation_id='conflict', event_names=[
            dict(table='/events/markers', code=3, name='New', shortcut='1')])
    with pytest.raises(AnalysisError, match='changed'):
        write_annotated_copy(source, output, operation_id='hash', expected_source_sha256='0' * 64,
            event_names=[dict(table='/events/markers', code=1, name='Changed')])
    write_annotated_copy(source, output, operation_id='first', event_names=[dict(table='/events/markers', code=1, name='Changed')])
    with pytest.raises(AnalysisError, match='already exists'):
        write_annotated_copy(source, output, operation_id='other', event_names=[dict(table='/events/markers', code=1, name='Changed')])
    second = tmp_path / 'two.nwb'
    with pytest.raises(AnalysisError, match='already recorded'):
        write_annotated_copy(output, second, operation_id='first', event_names=[dict(table='/events/markers', code=1, name='Changed')])
    receipt = write_annotated_copy(output, second, operation_id='second', event_names=[dict(table='/events/markers', code=1, name='Again')])
    assert receipt['root_source_sha256'] == sha(source)
    assert read_nwb(second, load_data=False)['annotation_root_source_sha256'] == sha(source)


@pytest.mark.parametrize('variant', ['soft', 'external', 'hard_alias', 'unknown_reference', 'region_reference', 'unknown_unit_column'])
def test_unsupported_aliases_references_and_unit_columns_fail_without_output(tmp_path, variant):
    source = recording(tmp_path / 'source.nwb')
    detection = result_for(source)
    with h5py.File(source, 'r+') as file:
        if variant == 'soft':
            file['bad'] = h5py.SoftLink('/units')
        elif variant == 'external':
            file['bad'] = h5py.ExternalLink('other.nwb', '/units')
        elif variant == 'hard_alias':
            file['bad'] = file['units/spike_times']
        elif variant == 'unknown_reference':
            file['unknown_extension'].attrs['changed_reference'] = file['units/spike_times'].ref
        elif variant == 'region_reference':
            file['unknown_extension'].attrs['region'] = file['units/spike_times'].regionref[:1]
        elif variant == 'unknown_unit_column':
            file['units'].create_dataset('unknown', data=[3])
    before = sha(source)
    output = tmp_path / 'copy.nwb'
    with pytest.raises(AnalysisError) as error:
        write_annotated_copy(source, output, detection=detection, channel_index=0, operation_id='reject')
    assert error.value.code == 'unsupported_layout'
    assert not output.exists() and sha(source) == before
