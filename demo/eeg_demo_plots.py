from spikertools import Session

# Paths to EEG data files
p300_wav_file = 'data/eeg/p300/BYB_Recording_2019-06-11_13.23.58.wav'

# --- Plotting P300 Session Overview ---

print("=== Plotting P300 Session Overview ===")

# Initialize the P300 EEG session
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

# --- Plotting EEG Traces Around Events (2 Minutes) ---

print("\n=== Plotting 30s EEG Traces at Arbitrary start of 10s ===")
s.plots.plot_channels(
    time_window=(10, 40),  # 2 minutes window
    title="30s of EEG Traces and Events"
)

# --- Plotting Event-Related Potentials (ERP) for P300 ---
print("\n=== Plotting Event-Related Potentials (ERP) ===")

# Plot ERP for 'standard' and 'oddball' events
s.plots.plot_erp(
    epoch_window=(-0.5, 1.0),  # 500ms before to 1000ms after event
)

# Wave Based Analysis - Using Alpha Wave

alpha_wave_wav_file = 'data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25.wav'

print("\n=== Plotting Power Spectral Density (PSD) for Alpha Wave ===")

# Initialize the Alpha Wave EEG session
alpha_session = Session(alpha_wave_wav_file)

# Define events for Alpha Wave
alpha_session.events[0].name = 'Close'
alpha_session.events[0].color = 'purple'

alpha_session.events[1].name = 'Open'
alpha_session.events[1].color = 'orange'

# Assign name to the Alpha Wave channel
alpha_session.channels[0].name = 'O1'

# Optional: Filter the Alpha Wave channel to isolate the alpha band (8-12 Hz)
alpha_session.channels[0].filter(ftype='bp', cutoff=[8, 12], order=3)  # Alpha band

# --- Plotting Spectrogram for Alpha Wave (First 30 Seconds) ---

print("\n=== Plotting Spectrogram for Alpha Wave (First 30 Seconds) ===")

# Plot spectrogram for channel 0 limited to the first 30 seconds
alpha_session.plots.plot_spectrogram(
    channel=alpha_session.channels[0], #Channel 1
    freq_range=(0, 50),  # Frequency range to display
    events = alpha_session.events,
    time_window=(0, 30),  # First 30 seconds
    save_path=None,
    title="Alpha Wave Spectrogram (Note: increase in alpha following eyes closed)",
    show=True
)

