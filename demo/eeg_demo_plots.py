from spikertools import Session
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import welch

# --- Configuration ---

# Paths to EEG data files
p300_wav_file = 'data/eeg/p300/BYB_Recording_2019-06-11_13.23.58.wav'
alpha_wave_wav_file = 'data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25.wav'

# Channel configurations
p300_channel_index = 0  # Using channel 0 for P300
alpha_channel_index = 0  # Using channel 0 for Alpha Wave
p300_channel_name = 'P4'
alpha_channel_name = 'Ch0'

# --- Plotting P300 Session Overview ---

print("=== Plotting P300 Session Overview ===")

# Initialize the P300 EEG session
p300_session = Session(p300_wav_file)

# Define events for P300
p300_session.events[0].name = 'standard'
p300_session.events[0].color = 'blue'

p300_session.events[1].name = 'oddball'
p300_session.events[1].color = 'red'

# Assign name to the P300 channel
p300_session.channels[p300_channel_index].name = p300_channel_name

# Optional: Filter the P300 channel to reduce high frequency noise
p300_session.channels[p300_channel_index].filter(ftype='lp', cutoff=5, order=3)

# --- Plotting EEG Traces Around Events (2 Minutes) ---

print("\n=== Plotting EEG Traces Around Events (2 Minutes) ===")
p300_session.plots.plot_channels(
    channels=[p300_channel_index],
    time_window=(0, 120),  # 2 minutes window
    save_path=None,
    title="2m of EEG Traces and Events",
    show=True
)

# --- Plotting Event-Related Potentials (ERP) for P300 ---
print("\n=== Plotting Event-Related Potentials (ERP) ===")

# Plot ERP for 'standard' and 'oddball' events
p300_session.plots.plot_erp(
    event_names=['standard', 'oddball'],
    epoch_window=(-0.5, 1.0),  # 500ms before to 1000ms after event
    channel_index=p300_channel_index,
    save_path=None,
    show=True
)

# --- Plotting Power Spectral Density (PSD) for Alpha Wave ---

print("\n=== Plotting Power Spectral Density (PSD) for Alpha Wave ===")

# Initialize the Alpha Wave EEG session
alpha_session = Session(alpha_wave_wav_file)

# Define events for Alpha Wave
alpha_session.events[0].name = 'Close'
alpha_session.events[0].color = 'purple'

alpha_session.events[1].name = 'Open'
alpha_session.events[1].color = 'orange'

# Assign name to the Alpha Wave channel
alpha_session.channels[alpha_channel_index].name = alpha_channel_name

# Optional: Filter the Alpha Wave channel to isolate the alpha band (8-12 Hz)
alpha_session.channels[alpha_channel_index].filter(ftype='bp', cutoff=[8, 12], order=3)  # Alpha band

# --- Plotting Spectrogram for Alpha Wave (First 30 Seconds) ---

print("\n=== Plotting Spectrogram for Alpha Wave (First 30 Seconds) ===")

# Plot spectrogram for channel 0 limited to the first 30 seconds
alpha_session.plots.plot_spectrogram(
    channel_index=alpha_channel_index,
    freq_range=(0, 50),  # Frequency range to display
    event_names=['Open', 'Close'],
    time_window=(0, 30),  # First 30 seconds
    save_path=None,
    show=True
)

