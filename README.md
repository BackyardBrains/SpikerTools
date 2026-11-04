# SpikerTools

**SpikerTools** is a Python library designed to help students analyze SpikeRecorder files from SpikerBoxes. It provides easy-to-use functions for loading, processing, and visualizing neural and EEG data, enabling students to explore neuroscience concepts through hands-on data analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
  - [Loading Data](#loading-data)
  - [Plotting Session Overview](#plotting-session-overview)
  - [Event-Related Potential (ERP)](#event-related-potential-erp)
  - [Peri-Event Time Histogram (PETH)](#peri-event-time-histogram-peth)
  - [Spectrogram Analysis](#spectrogram-analysis)
  - [Average Power Spectrum](#average-power-spectrum)
- [For Teachers](#for-teachers)
  - [Sample Lesson Plan](#sample-lesson-plan)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load neural and EEG data from NWB recordings or legacy WAV files with events sidecars.
- Handle events and neuronal spikes with timestamped annotations.
- Filter and normalize signal data.
- Compute statistical measures like inter-spike intervals and firing rates.
- Visualize data with various plots:
  - Session overview with event markers.
  - Event-Related Potentials (ERPs).
  - Peri-Event Time Histograms (PETH) with raster plots.
  - Spectrograms with event overlays.
  - Average power spectra for frequency analysis.
  - Histograms of inter-event and inter-spike intervals.

## Installation

You can install SpikerTools using pip:

```bash
pip install spikertools
```

Note: If SpikerTools is not yet available on PyPI, you can install it directly from the source code:

```bash
git clone https://github.com/BackyardBrains/SpikerTools.git
python -m pip install ./SpikerTools
```

### NWB support (source installation)

The NWB reader in this branch is not yet a published PyPI release. Install the
source with its optional NWB dependencies using **Python 3.10 or newer**:

```bash
# From the SpikerTools source directory:
python -m pip install '.[nwb]'
```

WAV-only installations do not require PyNWB. NWB support uses PyNWB 4.2 or later
in the 4.x series; the compatibility tests run against 4.2.0.

```python
from spikertools import Session

session = Session('recording.nwb')  # pathlib.Path also works
print(session.channels.names())
print(session.events.names())
print(session.metadata['session_description'])

# The existing plotting functions work with NWB too.
session.plots.plot_channels(save_path='waveform.png', show=False)
session.plots.plot_erp(save_path='erp.png', show=False)
session.plots.plot_average_power(
    events=list(session.events), channel=session.channels[0],
    freq_range=(0, 100), epoch_window=(0, 1),
    save_path='spectrum.png', show=False,
)
# If the recording contains sorted spikes:
if session.neurons:
    session.plots.plot_peth(
        neuron=session.neurons[0], events=list(session.events),
        save_path='peth.png', show=False,
    )
```

These commands save ordinary PNG files for sharing. Uploading them to My files
and running server jobs are separate integrations; this library performs local
analysis and does not transfer login credentials or fetch protected URLs.

#### What the NWB reader preserves

- **Signal:** acquisition `ElectricalSeries`, time along the first dimension,
  with one or multiple channels. Values in `session.data` and `channel.data`
  are float64 **volts**: `stored * conversion * channel_conversion + offset`.
  WAV data retains its original values and dtype; no voltage calibration is
  invented for WAV. NWB signal plots label volts, or dimensionless after
  standard-deviation/z-score normalization.
- **Clock:** `starting_time` plus `rate`, including AqNWB's one-element
  `starting_time` dataset, or uniformly spaced explicit timestamps. Existing
  plots use seconds from the **first signal sample**: event and spike times
  are shifted by `session.starting_time`, and `channel.time` begins at zero.
  Add `session.starting_time` to recover the original NWB clock. Events outside
  the signal range are retained, including negative relative times; epoch
  plots omit trials without enough signal on both sides.
- **Channels:** electrode row mapping (not electrode IDs), names from `label`,
  calibration, electrode ID, location and group name. Duplicate display names
  get a stable suffix so every channel remains addressable by name; the saved
  label remains in `channel.metadata['label']`.
- **Events:** every top-level `/events` table, including SpikeRecorder's
  `markers`, code meanings and text annotations. Deleted rows are omitted;
  active times are sorted. Distinct table/type identities remain separate
  even when names collide. `event.metadata` identifies the original table and
  type. `event.occurrences` is a source snapshot containing original NWB
  timestamps and available ID, source, channel and duration columns. It is not
  rewritten by subsequent edits/debouncing of `event.timestamps`.
- **Sorted spikes:** standard `/units` becomes `session.neurons`, preserving
  unit IDs, names, spike times and electrode membership. Each neuron exposes
  `channel_numbers` for the selected signal (possibly empty), `electrode_rows`
  and `metadata`. SpikeRecorder's threshold values remain **ADC counts**,
  identified by `neuron.threshold_unit`, with band and threshold count retained.
- **Recording information:** `session.datetime` is the saved session start
  datetime, including timezone. `session.metadata` contains the identifier,
  session description, timestamp reference datetime, NWB version, selected
  series name, signal description and comments. All data is copied before the
  file closes, so plots do not depend on an open file handle.

When there are multiple acquisition signals, the reader selects
`ElectricalSeries1` if present; otherwise select one explicitly:

```python
session = Session('recording.nwb', series_name='MyElectricalSeries')
```

The initial reader supports local HDF5 NWB files with acquisition
`ElectricalSeries`, EventsTable and Units. It rejects irregular sample clocks
rather than resampling silently. Processing-module signals, legacy
`AnnotationSeries`, spike waveform `SpikeEventSeries`, NWB-Zarr, remote URLs,
writing/exporting NWB, and automatic merging of separate recordings are outside
this reader's scope. An events sidecar is rejected for NWB to avoid accidentally
combining two clocks. Early SpikeRecorder files that advertise core 2.9 while
containing EventsTable are read using the installed core schema, not cached
file namespaces; custom NWB extensions are not automatically registered.

Like the existing WAV loader, this is an eager, in-memory scientific reader.
Allow memory for float64 samples and the time axis. It is not an isolation or
resource-limiting boundary for untrusted uploaded files; a hosted analysis
service must enforce its own worker isolation and limits.

The calibration and time-axis conventions follow the
[PyNWB ElectricalSeries documentation](https://pynwb.readthedocs.io/en/stable/pynwb.ecephys.html)
and the [NWB format specification](https://nwb-schema.readthedocs.io/en/latest/format.html).

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - numpy
  - scipy
  - matplotlib
  - seaborn

Install the required packages using:

```bash
pip install numpy scipy matplotlib seaborn
```

## Usage Examples

### Loading Data

```python
from spikertools import Session

# Load your data file (WAV file and corresponding events file)
wav_file_path = 'data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29.wav'

# Initialize the Session
session = Session(wav_file_path)
```

### Plotting Channels from Session

```python
# Plot the session overview with event markers
session.plots.plot_channels()
```

### Event-Related Potential (ERP)

```python

p300_wav_file = 'data/eeg/p300/BYB_Recording_2019-06-11_13.23.58.wav'
s = Session(p300_wav_file)

# Define events for P300 (default is '1' and '2') and add colors
s.events[0].name = 'standard'
s.events[0].color = 'blue'

s.events[1].name = 'oddball'
s.events[1].color = 'red'

# Assign location as name to the P300 channel (Optional)
s.channels[0].name = 'P4'

# Optional: Filter the P300 channel to reduce high frequency noise
s.channels[0].filter(ftype='lp', cutoff=10, order=3)
```

### Peri-Event Time Histogram (PETH)

```python
# Plot PETH with raster for a neuron aligned to an event
session.plots.plot_peth(
    neuron=s.neurons[0],     # Neuron to plot (defaults to first)
    events=s.events,        # List of Event objects (defaults to all)
    epoch_window=(-0.5, 1.0),      # 500ms before to 1000ms after event
    bin_size=0.04,           # 40ms bins
    title="Peri-Event Time Histogram (PETH) with Raster Plots for Touch Pressure Events", # Add a custom title
    save_path=None, # Save the plot to a file
    show=True
)
```

### Spectrogram Analysis

```python
# Plot spectrogram of the EEG data with event markers
session.plots.plot_spectrogram(
    channel=session.channels[0],
    freq_range=(0, 50),
    events=session.events
)
```

### Average Power Spectrum

```python
# Plot average power spectra during 'Open' and 'Close' events
session.plots.plot_average_power(
    events=session.events,
    freq_range=(0, 30),
    epoch_window=(0, 5),
    channel=session.channels[0]
)
```

## Event experiments (numerical API v1)

`Experiment` groups independent subjects' recordings without concatenating files.
Version 1 requires **one session per unique, explicit subject**. NWB `subject_id`
and `session_id` are exposed when present; supply IDs for recordings that lack
them. A duplicate source path, session object, session ID or subject ID is rejected.
This is a source-library API; it is not a My files deployment or a new PyPI release.

```python
from spikertools import Session, Experiment

sessions = [Session("subject-a.nwb"), Session("subject-b.nwb")]
experiment = Experiment(sessions,
                        subject_ids=["subject-a", "subject-b"],
                        session_ids=["recording-a", "recording-b"])

catalogue = sessions[0].analysis_catalogue()
touch_key = catalogue["events"][0]["key"]  # Review event identities in every file.
settings = {
    "method": "event_average",       # or "rms", or "peth" with saved spikes
    "event_keys": [touch_key],
    "tmin": -0.2, "tmax": 0.8,      # seconds relative to each event
    "baseline": [-0.2, 0.0],        # optional; unavailable for PETH
    "max_output_points": 2000,
}
result = experiment.analyze(settings, channel=0)
curve = result["events"][0]
print(curve["subject_count"], curve["trial_count"])
# curve["time"], curve["mean"], curve["subject_sd"] are ordinary JSON lists.
```

For example, `excluded_trials={"recording-a": {touch_key: ["row:3"]}}` excludes
that occurrence, if `row:3` exists under the chosen event in that recording. An
exclusion referring to an absent trial is an error. `channel` is a channel
position or a `{session_id: position}` mapping. `excluded_session_ids` excludes
whole sessions; `excluded_trials` maps session IDs to event-key/trial-ID lists.
`Session.analyze_events(settings, ...)` returns the same individual-session result.

The catalogue contains channels, saved units and event occurrences. An NWB event
key is compact JSON `[table_path, event_type, annotation]`; labels are display text
and never merge identities. A trial ID is `row:<original event table row>`, scoped
to its event key and session. Equal timestamps remain distinct selectable trials.
Deleted NWB rows are absent. If an NWB event's timestamp list was subsequently
edited without updating its row identities, analysis fails and asks you to reload
the source. In-memory legacy events use their names and original list positions;
keep those lists unchanged while reusing saved selections.

For files recorded with common keyboard bindings, explicitly opt in to
`session_catalogue(session, event_identity="keyboard_shortcut_v1")`. Its events
use compact JSON `["shortcut", key]`, or `["shortcut", key, note]` for a nonempty
note. Empty and absent notes agree; notes never join unannotated trials. The
same key can therefore identify baseline even if one file stores local code1
and another stores local code2. Matching is case-sensitive and requires an
explicit, unique stored shortcut; it never guesses from names, codes or order.
Duplicate or malformed bindings raise an error. An unbound event keeps its
legacy key and has `identity_status="unbound"`; a group-analysis service must
require a binding before treating it as a shared condition.

The opt-in catalogue advertises `event_identity="keyboard_shortcut_v1"`. Each
event includes `identity_status`, `shortcut`, and `source_events` containing the
original `table`, `code`, `annotation` and `legacy_key`; trial IDs stay unchanged.
Save this mapping with analysis provenance. `analyze_session` accepts these event
objects but returns its existing numerical/trial schema. The default catalogue
still uses `table_code_annotation_v1`. Do not silently reinterpret old selections
or exclusions under new keys; preserve historical results and ask for a fresh
selection review when changing identity versions.

For PETH, set `method="peth"`, `baseline=None` and `bin_size` in seconds. Select
the saved unit with `neuron` (position or per-session mapping); only a single
compatible unit is automatically selected. Electrode/channel linkage is required.
Empty but present saved spike trains yield zero rates; absent spikes fail clearly.
Unit IDs are recorded independently for every session and do **not** identify the
same biological neuron across subjects. This API does not detect new spikes.

### Calculation and inclusion rules

- Event averages and RMS require identical native sample rates across sessions.
  Different rates fail explicitly; there is no hidden interpolation, resampling
  or anti-alias filter. PETH uses common time bins and can combine different rates.
- Signal epochs use the nearest event sample (half-sample ties round up). Every
  trial records the alignment error in seconds. Signal times are native offsets
  inside `[tmin,tmax)`; tiny rounding tolerance at exact sample boundaries is
  `1e-9` samples. Event and spike times are relative to the selected channel's
  first sample; the NWB reader subtracts the recorded start clock once.
- RMS is calculated **within each trial before averaging**. Its centered window
  has `round(rms_window * sample_rate)` samples (half ties up, minimum one). An
  even window has one more sample before the center; effective size/padding are
  returned in `processing`. RMS uses linear-work block prefix/suffix norms, avoiding
  cancellation after large artifacts. Subnormal values that would underflow during
  normalization fail explicitly with `precision_loss`. Baseline correction subtracts the trial's baseline
  mean: raw amplitude for event averages, RMS amplitude for RMS. It is not percent
  normalization or division by baseline.
- The entire response, baseline and RMS padding must exist. Edge or nonfinite
  trials are excluded with a reason. PETH includes spikes at the left bin boundary
  and excludes the right boundary, including `tmax`; a short last bin uses its
  true width. Only complete, retained trials enter the denominator.
- All native samples enter the calculations. After calculation, adjacent output
  samples are averaged into at most `max_output_points` display bins, with their
  mean times. This lossy display reduction is **not** a resampled signal for later
  spectral analysis. PETH instead requires its explicit bin count to fit the limit.
- Each subject's result is its own retained-trial mean; the group equally weights
  those subject means. `subject_sd` is descriptive sample standard deviation across
  subject means, or `null` for one subject. It is **not** a confidence interval,
  standard error or hypothesis test. Per-event subject/trial counts, contributing
  subject IDs, and subjects with no valid trials are all returned.
- Each session must retain at least one usable trial among selected events; each
  group event must have at least one contributing subject. All-excluded/invalid
  analysis fails with `no_valid_trials`, rather than a successful empty graph.
- Volts, V, mV and uV are calibrated to volts; dimensionless and ADC counts remain
  distinct units. Unknown or incompatible group units fail. Signal role and prior
  display filter hints are exposed from bounded optional `byb_display_v1` metadata;
  they are informational and do not apply filtering. The caller must confirm that
  the chosen channels/conditions are physiologically comparable. Equal units alone
  do not establish that comparability.

### Bounded file-worker adapter

Workers can load just one selected channel and use the same kernel without an
eager multi-channel `Session`:

```python
from spikertools.experiment import analyze_session, aggregate_results, event_key

key = event_key("/events/markers", 1)
subject_result = analyze_session(
    session_id="recording-a", subject_id="subject-a", channel_id="0",
    data=selected_channel, sample_rate=1000, unit="volts",
    events=[{"key": key, "label": "Touch",
             "trials": [{"id": "row:0", "time": 2.0}]}],
    settings={"method": "rms", "event_keys": [key],
              "tmin": -0.2, "tmax": 0.8, "rms_window": 0.05},
    excluded_trial_ids={},
)
group_result = aggregate_results([subject_result])
```

Optional PETH arguments are `spike_times`, `spike_unit_id`, `spike_channel_ids`
(which must contain `channel_id`) and `duration_seconds`. `data=None` is allowed
for PETH if duration is explicit. When data exists, duration must equal
`len(data)/sample_rate`. All IDs are strings and all times are finite seconds.
`AnalysisError` exposes a stable `.code` and structured `.details`, including
`missing_event`, `missing_spikes`, `spike_channel_mismatch`, `duplicate_subject`,
`incompatible_grid`, `incompatible_units`, `bounds` and `no_valid_trials`.

`spikertools.nwb.read_nwb(path, series_name=None, load_data=False)` provides the
reader's model dictionary without reading signal samples or allocating full
per-channel clocks. It includes `sample_count`, `duration_seconds`, calibration,
event row identities and saved spike times; explicit timestamp clocks are validated
in chunks of at most 65,536 samples. `data` is null, channel data/time arrays are
empty, and `data_loaded=False` marks these objects as catalogue-only. Passing such
a session to `Experiment.analyze` fails with `metadata_only`; use a bounded reader
for the selected calibrated channel and call `analyze_session` instead. PETH can
use `data=None` and explicit duration. Default `load_data=True` preserves the eager
local `Session` behavior. Events and saved spikes are still loaded into memory.
Metadata-only defaults cap event rows at 100,000, saved spike times at 10 million,
units at 256 and channels at 64; dataset shapes are checked before materialization.
Pass `max_event_rows`, `max_spike_times`, `max_units` and `max_channels` to set
stricter limits. Eager reads retain their existing unlimited behavior unless those
limits are explicitly set. The web service must additionally bound encoded text
metadata and total bytes.

Session output has `schema_version="spikertools.session-analysis.v1"`, the IDs,
sample rate/duration/unit, normalized `settings`, `processing` provenance and an
`events` list. Each event has `key`, `label`, `time`, `mean` (or null when empty),
`trial_count`, `trial_total` and `trials`. Every trial has `id`, `time`, `included`,
`reason` (null, `excluded`, `edge_window` or `nonfinite`) and alignment error.
Group output has `schema_version="spikertools.experiment-analysis.v1"`,
`weighting="equal_subject"`, settings/unit, validated `sessions`, and event means,
subject SD/count/IDs, trial counts and `excluded_subjects` with reason/counts.
Results contain finite JSON values and never modify source arrays or events.

Limits are 24 subjects, 64 event types, 100,000 occurrences per session, 50 million
signal samples, 10 million spikes, sample rates up to 1 MHz, 120-second response
and baseline windows within ±120 seconds, 2 million samples per native epoch/RMS
window, 200 million sample-work units per session, 10,000 output points per trace,
and 10 million combined event-trace values. These are library ceilings; a web
service should additionally bound file bytes, process memory, CPU and job time.
Authenticate source access separately; this library does not fetch remote URLs,
grant sharing rights or execute user-provided scripts.

## Explicit spike detection and annotated copies

```python
from spikertools.annotations import detect_spikes, write_annotated_copy

session = Session("recording.nwb")
preview = detect_spikes(session.channels[0].data, sample_rate=session.sample_rate,
                        channel_id="0")
# Inspect preview['preview'], thresholds and spike_count before explicitly saving.
receipt = write_annotated_copy("recording.nwb", "annotated.nwb", detection=preview,
    channel_index=0, operation_id="reviewed-detection-1", event_names=[
        {"table": "/events/markers", "code": 1, "name": "Touch"}])
```

The default detector uses positive excursions above the median baseline plus
4 times robust noise sigma: median absolute deviation divided by
0.6744897501960817. Settings support `polarity` (`positive`, `negative`, `both`),
`sigma_multiplier` (1–20) and `refractory_ms` (0.1–100, default 1). Each continuous
threshold excursion contributes its strongest sample. Refractory suppression
keeps strongest absolute excursions first, breaking ties by earliest sample.
There is no filtering or automatic clustering; “Detected spikes” does not mean
an isolated neuron. Zero/invalid noise and nonfinite samples fail explicitly.
Saturation is not inferred from calibrated voltages. The preview contains at most
2,000 consecutive real samples; the complete train stays available separately
for saving. Do not send full trains or raw recordings as preview JSON.

The writer exclusively creates a new file and checks the selected channel's data
hash, calibration and clock. Services should pass `expected_source_sha256` from
their preview and independently enforce access, quota and durable publication.
Raw arrays, existing units, events, subject identity and unknown fields are
preserved. New standard NWB units contain absolute spike times, electrode row
references and detector provenance. Zero detections append no empty unit.
SpikeRecorder currently restores its own ADC threshold bands and re-detects them;
it does not display these arbitrary saved units as editable spike trains.
The saved trains work in My files PETH and Python. No physical voltage threshold
is misrepresented as native ADC configuration.

Event names and optional shortcuts use native `event_type_meanings` definitions.
Existing shortcuts are preserved; conflicting requests fail. New definitions can
include a `shortcut`, including in an explicitly selected absent events table.
They add no occurrence or timestamp. The separate `event_definitions` catalogue
contains table, code, name, shortcut and occurrence count, including zero counts.
Its `has_definition` flag distinguishes a meanings row from an observed numeric
code that has not been named yet. Historical cross-table shortcut reuse survives
renaming; only a new conflicting assignment is rejected.
Appearance and free-text occurrence annotations remain unchanged. The native
recorder reads these definitions; its UI may merge equal names across codes.
Copies retain `annotation_root_source_sha256` across generations so applications
can reject original/copy duplicate subjects without inventing subject IDs.

Use the read-only resolver when naming the same keyboard events across files:

```python
from spikertools.annotations import resolve_event_names, validate_annotated_copy

changes = resolve_event_names("recording.nwb", {
    "event_identity": "keyboard_shortcut_v1",
    "event_names": [{"shortcut": "1", "name": "Baseline"},
                    {"shortcut": "9", "name": "Blow"}],
})
capability = validate_annotated_copy("recording.nwb", event_names=changes,
    expected_source_sha256=source_sha256, operation_id="reviewed-names-1")
# Present the resolved changes for review before calling write_annotated_copy
# with the same arguments and a new output path.
```

The resolver returns ordinary local writer operations. A missing shortcut gets
the lowest unused positive code, reserving both defined and observed codes,
including deleted occurrences. It adds only a definition with zero trials. A
missing event table can become `/events/markers`; multiple candidate tables are
an explicit ambiguity. Legacy code-based payloads are rejected by this resolver.

`validate_annotated_copy` accepts the writer's exact arguments except output path
and uses its same bounded source/request plan, including detection-specific
requirements. Its `spikertools.annotation-capability.v1` receipt includes source
and root hashes, source bytes, maximum output bytes, proposed unit ID and resolved
event names. It creates no file. Save repeats validation and still checks output
creation, free-space failures, source races and final byte limits. A capability
is neither access permission nor a guarantee against those operational failures.

Detection is bounded to 20 million selected samples, 1 million excursions and
100,000 retained spikes. Exact full-channel median/MAD requires linear memory.
Copies are limited to 100 MiB total, 8 MiB added, 64 event definitions and 128
units. Unsupported custom unit columns, region references, aliases into changed
data and unsafe links fail rather than silently losing content. Standard internal
electrode-group device links and native event/source MeaningsTable target links
are preserved. Targets must be the exact canonical unchanged columns with
hard-linked path components and correct NWB types. No external/virtual dataset access
or arbitrary HDF5 link traversal is permitted. Operation IDs use 1–128 ASCII
letters, digits, underscores or hyphens; repeated IDs and existing outputs fail.

## For Teachers

SpikerTools is designed to be an educational tool that integrates seamlessly into classroom activities. Here's how you can incorporate it into your teaching:

- **Hands-On Data Analysis**: Provide students with real neural or EEG data recordings and guide them through loading and analyzing the data using SpikerTools.
- **Visualization of Neural Activity**: Use the plotting functions to help students visualize neural spikes, event-related potentials, and frequency content of EEG signals.
- **Concept Reinforcement**: Reinforce concepts like neuronal firing rates, inter-spike intervals, and the effects of stimuli on neural activity.
- **Customizable Plots**: Encourage students to explore different parameters and customize plots to deepen their understanding.
- **Interdisciplinary Learning**: Integrate programming skills with neuroscience, promoting interdisciplinary education.

### Sample Lesson Plan

1. **Introduction to Neural Signals:**
   - Discuss the basics of neural spikes and EEG signals.
   - Explain the significance of events and stimuli in neural recordings.
   
2. **Data Loading and Preprocessing:**
   - Show students how to load data into SpikerTools.
   - Demonstrate filtering and normalization techniques.

3. **Data Visualization:**
   - Guide students through plotting session overviews and ERPs.
   - Analyze spectrograms to understand frequency components.

4. **Data Analysis:**
   - Calculate firing rates and inter-spike intervals.
   - Compare neural responses to different stimuli.

5. **Discussion and Interpretation:**
   - Interpret the results and discuss their implications.
   - Encourage students to ask questions and explore further.

## Contributing

We welcome contributions to enhance SpikerTools. If you have ideas for new features, improvements, or bug fixes, please:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear messages.
4. Submit a pull request describing your changes.

Please ensure that your code follows best practices and includes appropriate tests.

## License

SpikerTools is released under the **MIT License**.

## Development tests

```bash
python -m pip install -e '.[nwb]' pytest build
MPLBACKEND=Agg python -m pytest -q
python -m build
```

Tests generate small NWB files and check calibrated samples, voltage power,
clock offsets, event deletion/identity, electrode mapping, sorted spikes, graph
output and WAV compatibility. CI also runs the WAV suite without PyNWB installed.
