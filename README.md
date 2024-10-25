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

- Load neural and EEG data from WAV files.
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
git clone https://github.com/yourusername/SpikerTools.git
cd SpikerTools
pip install .
```

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

### Plotting Session Overview

```python
# Plot the session overview with event markers
session.plots.plot_session()
```

### Event-Related Potential (ERP)

```python
# Make the event markers have meaningful names and add different colors to differntiate on the same plot
for event in session.events:
    if event.name.strip() == '1':
        event.name = 'Standard'
        event.color = 'blue'
    elif event.name.strip() == '2':
        event.name = 'Oddball'
        event.color = 'red'

# Filter the channel to reduce noise (optional)
session.channels[0].filter(ftype='lp', cutoff=10, order=3)

# Plot ERPs for both events on the same axis
session.plots.plot_erp(
    events=session.events,
    epoch_window=(-0.5, 1.0),
    channel=session.channels[0]
)
```

### Peri-Event Time Histogram (PETH)

```python
# Plot PETH with raster for a neuron aligned to an event
session.plots.plot_peth(
    neuron=neuron,           # Neuron to plot
    events=event_list,       # List of Event objects
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
