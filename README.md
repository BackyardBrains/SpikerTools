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
