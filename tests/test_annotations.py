"""Synthetic numerical and copy-preservation tests; no customer files."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from spikertools.annotations import (detect_spikes, resolve_event_names,
                                    validate_annotated_copy, write_annotated_copy)
from spikertools.experiment import AnalysisError, analyze_session, event_key, session_catalogue


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
    assert definitions['/events/markers', 1] == dict(table='/events/markers', code=1, name='Touch', shortcut='1', occurrences=2, has_definition=True)
    assert definitions['/events/markers', 3]['occurrences'] == 0
    assert len(loaded.events) == 2  # Existing blank/free-text groups, no fabricated trial.
    assert loaded.annotation_root_source_sha256 == source_hash
    event = session_catalogue(loaded)['events'][0]
    peth = analyze_session(session_id='session', subject_id='subject', data=None,
        sample_rate=loaded.sample_rate, unit='volts', channel_id='0', duration_seconds=loaded.duration_seconds,
        events=[event], settings=dict(method='peth', event_keys=[event['key']], tmin=-.19, tmax=.21, bin_size=.1),
        spike_times=loaded.neurons[-1].timestamps, spike_channel_ids=['0'], spike_unit_id=receipt['new_unit_id'])
    assert peth['processing']['spike_unit_id'] == '18'
    assert sum(peth['events'][0]['mean']) == pytest.approx(10.)
    assert validate(path=output) == []


def test_explicit_timestamp_clock_is_preserved_when_saving_spikes(tmp_path):
    source = recording(tmp_path / 'clock.nwb')
    with h5py.File(source, 'r+') as file:
        series = file['acquisition/ElectricalSeries1']
        del series['starting_time']
        stamps = series.create_dataset('timestamps', data=5 + np.arange(4000) / 1000)
        stamps.attrs.update(unit='seconds', interval=1)
    output = tmp_path / 'copy.nwb'
    write_annotated_copy(source, output, detection=result_for(source), channel_index=0, operation_id='explicit-clock')
    with h5py.File(source) as before, h5py.File(output) as after:
        np.testing.assert_array_equal(before['acquisition/ElectricalSeries1/timestamps'][:], after['acquisition/ElectricalSeries1/timestamps'][:])
        assert 'starting_time' not in after['acquisition/ElectricalSeries1']
        assert after['units/spike_times'][-1] == pytest.approx(5.101)


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
    assert all(not d['has_definition'] for d in read_nwb(source, load_data=False)['event_definitions'])
    output = tmp_path / 'named.nwb'
    write_annotated_copy(source, output, operation_id='names', event_names=[
        dict(table='/events/markers', code=i, name=f'Condition {i}', shortcut=str(i)) for i in range(1, 7)])
    loaded = Session(output)
    assert [d['shortcut'] for d in loaded.event_definitions] == list('123456')
    assert [d['occurrences'] for d in loaded.event_definitions] == [2, 0, 0, 0, 0, 0]
    assert validate(path=output) == []


def test_rename_preserves_historical_shortcut_reuse_across_tables(tmp_path):
    source = recording(tmp_path / 'source.nwb')
    with h5py.File(source, 'r+') as file:
        file.copy('events/markers', 'events/other')
        other = file['events/other']
        other['meanings_tables/event_type_meanings'].attrs['target'] = other['event_type'].ref
    output = tmp_path / 'copy.nwb'
    write_annotated_copy(source, output, operation_id='rename-existing', event_names=[
        dict(table='/events/markers', code=1, name='Touch', shortcut='1')])
    definitions = {(d['table'], d['code']): d for d in Session(output).event_definitions}
    assert definitions['/events/markers', 1]['name'] == 'Touch'
    assert definitions['/events/other', 1]['name'] == 'Old one'
    assert definitions['/events/markers', 1]['shortcut'] == definitions['/events/other', 1]['shortcut'] == '1'
    with pytest.raises(AnalysisError, match='shortcut'):
        write_annotated_copy(source, tmp_path / 'conflict.nwb', operation_id='new-conflict', event_names=[
            dict(table='/events/markers', code=3, name='New', shortcut='1')])


def native_meanings_links(path):
    """Match nwb_io.cpp writeMeaningsTable's direct, absolute target links."""
    with h5py.File(path, 'r+') as file:
        table = file['events/markers']
        types = table['meanings_tables/event_type_meanings']
        del types.attrs['target']
        types['target'] = h5py.SoftLink(table['event_type'].name)
        file.copy(types, table['meanings_tables'], name='source_meanings')
        sources = table['meanings_tables/source_meanings']
        del sources['target']
        sources['target'] = h5py.SoftLink(table['source'].name)
    return path


def test_native_meanings_links_survive_rename_and_unused_definition(tmp_path):
    source = native_meanings_links(recording(tmp_path / 'native.nwb'))
    before = sha(source)
    output = tmp_path / 'named.nwb'
    write_annotated_copy(source, output, operation_id='native-names', event_names=[
        dict(table='/events/markers', code=1, name='Baseline'),
        dict(table='/events/markers', code=9, name='Blow', shortcut='9')])
    assert sha(source) == before
    with h5py.File(source) as original, h5py.File(output) as copied:
        for column in ('event_type', 'source'):
            group = copied[f'events/markers/meanings_tables/{column}_meanings']
            target = group.get('target', getlink=True)
            assert isinstance(target, h5py.SoftLink)
            assert target.path == f'/events/markers/{column}'
            assert group['target'].id == copied[f'events/markers/{column}'].id
        for name in ('id', 'timestamp', 'event_type', 'source', 'annotation', 'deleted'):
            np.testing.assert_array_equal(original[f'events/markers/{name}'][:], copied[f'events/markers/{name}'][:])
        np.testing.assert_array_equal(original['acquisition/ElectricalSeries1/data'][:], copied['acquisition/ElectricalSeries1/data'][:])
    definitions = {d['code']: d for d in Session(output).event_definitions}
    assert definitions[1]['name'] == 'Baseline' and definitions[1]['shortcut'] == '1'
    assert definitions[9]['name'] == 'Blow' and definitions[9]['occurrences'] == 0


def keyboard_recording(path, shortcuts=('1', '2', '3')):
    source = native_meanings_links(recording(path, units=False))
    with h5py.File(source, 'r+') as file:
        table = file['events/markers']
        table['event_type'][:] = [1, 2, 3]
        table['deleted'][:] = 0
        table['annotation'][:] = ['', '', '']
        meanings = table['meanings_tables/event_type_meanings']
        fields = dict(id=[0, 1, 2], value=[1, 2, 3], shortcut=list(shortcuts),
                      meaning=[{'1': 'Baseline', '2': 'Baseline end', '3': 'Saline'}[k] for k in shortcuts])
        for key, values in fields.items():
            attrs = dict(meanings[key].attrs)
            del meanings[key]
            dtype = h5py.string_dtype('utf-8') if key in ('shortcut', 'meaning') else np.int32
            meanings.create_dataset(key, data=values, dtype=dtype).attrs.update(attrs)
        file['acquisition/ElectricalSeries1/data'][:, 0] = np.arange(4000)
    return source


def test_shortcut_identity_selects_same_key_when_local_codes_are_permuted(tmp_path):
    for filename, shortcuts, baseline_code in [('one.nwb', ('1', '2', '3'), 1),
                                                ('two.nwb', ('3', '1', '2'), 2)]:
        source = keyboard_recording(tmp_path / filename, shortcuts)
        session = Session(source)
        legacy = session_catalogue(session)
        catalogue = session_catalogue(session, event_identity='keyboard_shortcut_v1')
        assert legacy['event_identity'] == 'table_code_annotation_v1'
        assert catalogue['event_identity'] == 'keyboard_shortcut_v1'
        event = next(e for e in catalogue['events'] if e['key'] == '["shortcut","1"]')
        assert event['label'] == 'Baseline' and event['identity_status'] == 'bound'
        assert event['source_events'] == [dict(table='/events/markers', code=baseline_code, annotation='',
            legacy_key=event_key('/events/markers', baseline_code, ''))]
        old = next(e for e in legacy['events'] if e['key'] == event_key('/events/markers', baseline_code, ''))
        assert event['trials'] == old['trials']
        result = analyze_session(session_id=filename, subject_id=filename, data=session.channels[0].data,
            sample_rate=session.sample_rate, unit='volts', channel_id='0', events=catalogue['events'],
            settings=dict(method='event_average', event_keys=[event['key']], tmin=0, tmax=.002))
        sample = 100 + baseline_code * 100
        np.testing.assert_allclose(result['events'][0]['mean'], .1 + np.arange(sample, sample + 2) * .002)


def test_shortcut_names_resolve_local_codes_and_reserve_observed_undefined_codes(tmp_path):
    source = keyboard_recording(tmp_path / 'permuted.nwb', ('3', '1', '2'))
    with h5py.File(source, 'r+') as file:
        # Code4 is observed but has no definition, and the row is deleted.
        file['events/markers/event_type'][-1] = 4
        file['events/markers/deleted'][-1] = 1
    settings = dict(event_identity='keyboard_shortcut_v1', event_names=[
        dict(shortcut='1', name='baseline'), dict(shortcut='9', name='blow')])
    changes = resolve_event_names(source, settings)
    assert changes == [dict(table='/events/markers', code=2, name='baseline', shortcut='1'),
                       dict(table='/events/markers', code=5, name='blow', shortcut='9')]
    before = sha(source)
    capability = validate_annotated_copy(source, event_names=changes, expected_source_sha256=before, operation_id='keys')
    assert capability['schema_version'] == 'spikertools.annotation-capability.v1'
    assert capability['source_sha256'] == before and capability['new_unit_id'] is None
    assert capability['event_names'] == changes and sha(source) == before
    assert list(tmp_path.iterdir()) == [source]
    output = tmp_path / 'named.nwb'
    receipt = write_annotated_copy(source, output, event_names=changes, expected_source_sha256=before, operation_id='keys')
    assert receipt['event_names'] == capability['event_names']
    loaded = Session(output)
    definitions = {d['shortcut']: d for d in loaded.event_definitions}
    assert definitions['1']['code'] == 2 and definitions['1']['name'] == 'baseline'
    assert definitions['9']['code'] == 5 and definitions['9']['occurrences'] == 0
    with h5py.File(source) as original, h5py.File(output) as copied:
        for column in ('event_type', 'timestamp', 'deleted'):
            np.testing.assert_array_equal(original[f'events/markers/{column}'][:], copied[f'events/markers/{column}'][:])
    assert sha(source) == before


def test_shortcut_notes_are_separate_unbound_events_stay_legacy_and_case_is_exact(tmp_path):
    source = native_meanings_links(recording(tmp_path / 'notes.nwb'))
    with h5py.File(source, 'r+') as file:
        file['events/markers/meanings_tables/event_type_meanings/shortcut'][:] = ['A', 'a']
        file['events/markers/meanings_tables/event_type_meanings/meaning'][0] = ''
    session = Session(source)
    events = session_catalogue(session, event_identity='keyboard_shortcut_v1')['events']
    assert [e['key'] for e in events] == ['["shortcut","A"]', '["shortcut","A","Free text"]']
    assert [e['label'] for e in events] == ['Key A', 'Free text']
    changes = resolve_event_names(source, dict(event_identity='keyboard_shortcut_v1', event_names=[dict(shortcut='a', name='Other')]))
    assert changes[0]['code'] == 2
    with h5py.File(source, 'r+') as file:
        file['events/markers/meanings_tables/event_type_meanings/shortcut'][0] = ''
    events = session_catalogue(Session(source), event_identity='keyboard_shortcut_v1')['events']
    assert all(e['identity_status'] == 'unbound' and e['shortcut'] is None for e in events)
    assert events[0]['key'] == event_key('/events/markers', 1, '')


def test_missing_shortcut_without_tables_creates_only_definition_and_ambiguous_tables_fail(tmp_path):
    source = recording(tmp_path / 'empty.nwb', events=False, units=False)
    settings = dict(event_identity='keyboard_shortcut_v1', event_names=[dict(shortcut='9', name='Blow')])
    changes = resolve_event_names(source, settings)
    assert changes == [dict(table='/events/markers', code=1, name='Blow', shortcut='9')]
    output = tmp_path / 'named.nwb'
    write_annotated_copy(source, output, event_names=changes, operation_id='unused')
    assert len(Session(output).events) == 0
    assert Session(output).event_definitions[0]['occurrences'] == 0
    source = recording(tmp_path / 'two-tables.nwb', units=False)
    with h5py.File(source, 'r+') as file:
        file.copy('events/markers', 'events/other')
        other = file['events/other']
        other['meanings_tables/event_type_meanings'].attrs['target'] = other['event_type'].ref
        other['meanings_tables/event_type_meanings/shortcut'][:] = ['', '']
    with pytest.raises(AnalysisError) as error:
        resolve_event_names(source, settings)
    assert error.value.code == 'ambiguous_event_table'


@pytest.mark.parametrize('shortcut,code', [('1', 'ambiguous_event_binding'), ('12', 'invalid_event_binding')])
def test_duplicate_or_malformed_shortcut_never_chooses_first_definition(tmp_path, shortcut, code):
    source = recording(tmp_path / 'ambiguous.nwb')
    with h5py.File(source, 'r+') as file:
        file['events/markers/meanings_tables/event_type_meanings/shortcut'][1] = shortcut
    settings = dict(event_identity='keyboard_shortcut_v1', event_names=[dict(shortcut='1', name='Baseline')])
    for call in (lambda: session_catalogue(Session(source), event_identity='keyboard_shortcut_v1'),
                 lambda: resolve_event_names(source, settings)):
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code
    # The default reader remains compatible, with no implicit cross-file remap.
    assert len(session_catalogue(Session(source))['events']) == 2


@pytest.mark.parametrize('settings', [
    {'event_names': [{'table': '/events/markers', 'code': 1, 'name': 'Baseline'}]},
    {'event_identity': 'keyboard_shortcut_v1', 'event_names': [{'code': 1, 'name': 'Baseline'}]},
    {'event_identity': 'keyboard_shortcut_v1', 'event_names': [{'shortcut': '1', 'name': 'Baseline'}, {'shortcut': '1', 'name': 'Duplicate'}]},
])
def test_resolver_rejects_legacy_payload_and_duplicate_requests(tmp_path, settings):
    source = recording(tmp_path / 'source.nwb')
    before = sha(source)
    with pytest.raises(AnalysisError):
        resolve_event_names(source, settings)
    assert sha(source) == before


def test_absent_and_empty_notes_share_shortcut_key_without_rewriting_legacy_identity(tmp_path):
    blank = keyboard_recording(tmp_path / 'blank.nwb')
    absent = keyboard_recording(tmp_path / 'absent.nwb')
    with h5py.File(absent, 'r+') as file:
        del file['events/markers/annotation']
        file['events/markers'].attrs['colnames'] = np.asarray(
            ['timestamp', 'event_type', 'deleted', 'source', 'channel'], dtype=h5py.string_dtype('utf-8'))
    blank_events = session_catalogue(Session(blank), event_identity='keyboard_shortcut_v1')['events']
    absent_events = session_catalogue(Session(absent), event_identity='keyboard_shortcut_v1')['events']
    assert [e['key'] for e in blank_events] == [e['key'] for e in absent_events]
    assert blank_events[0]['source_events'][0]['legacy_key'] == event_key('/events/markers', 1, '')
    assert absent_events[0]['source_events'][0]['legacy_key'] == event_key('/events/markers', 1, None)


@pytest.mark.parametrize('variant', ['wrong_target', 'chain', 'external', 'hard_alias', 'wrong_kind'])
def test_native_link_allowance_does_not_accept_arbitrary_links(tmp_path, variant):
    source = native_meanings_links(recording(tmp_path / 'source.nwb'))
    with h5py.File(source, 'r+') as file:
        meanings = file['events/markers/meanings_tables/event_type_meanings']
        del meanings['target']
        if variant == 'wrong_target':
            meanings['target'] = h5py.SoftLink('/events/markers/source')
        elif variant == 'chain':
            table = file['events/markers']
            table.move('event_type', 'actual_event_type')
            table['event_type'] = h5py.SoftLink('/events/markers/actual_event_type')
            meanings['target'] = h5py.SoftLink('/events/markers/event_type')
        elif variant == 'external':
            meanings['target'] = h5py.ExternalLink('foreign.nwb', '/events/markers/event_type')
        elif variant == 'hard_alias':
            meanings['target'] = file['events/markers/event_type']
        elif variant == 'wrong_kind':
            meanings['target'] = h5py.SoftLink('/events/markers/event_type')
            meanings.attrs['neurodata_type'] = 'DynamicTable'
    before = sha(source)
    with pytest.raises(AnalysisError) as error:
        validate_annotated_copy(source, operation_id='reject', event_names=[dict(table='/events/markers', code=1, name='Baseline')])
    assert error.value.code == 'unsupported_layout'
    assert sha(source) == before and list(tmp_path.iterdir()) == [source]


def test_exact_preflight_rejects_spike_specific_layout_and_source_changes(tmp_path):
    source = recording(tmp_path / 'source.nwb')
    detection = result_for(source)
    before = sha(source)
    capability = validate_annotated_copy(source, detection=detection, channel_index=0,
                                         expected_source_sha256=before, operation_id='detect')
    assert capability['new_unit_id'] == '18' and sha(source) == before
    with h5py.File(source, 'r+') as file:
        file['units'].create_dataset('unknown', data=[3])
    # Event-only edits preserve unknown unit data; detection must not extend it.
    validate_annotated_copy(source, operation_id='names', event_names=[dict(table='/events/markers', code=1, name='Baseline')])
    with pytest.raises(AnalysisError) as error:
        validate_annotated_copy(source, detection=detection, channel_index=0, operation_id='detect')
    assert error.value.code == 'unsupported_layout'
    with pytest.raises(AnalysisError) as error:
        validate_annotated_copy(source, expected_source_sha256=before, operation_id='names',
                                event_names=[dict(table='/events/markers', code=1, name='Baseline')])
    assert error.value.code == 'source_changed'


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
