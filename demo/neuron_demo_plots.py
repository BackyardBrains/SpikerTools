from spikertools import Session

# Paths to Neuron data files
spike_wav_file = 'data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29.wav'

# Initialize the Neuron Session
s = Session(spike_wav_file)

# Define events for Neurons
s.events[0].name = 'light'
s.events[0].color = 'blue'

s.events[1].name = 'medium'
s.events[1].color = 'red'

s.events[2].name = 'heavy'
s.events[2].color = 'green'

# Assign name to the Neuron channel
s.channels[0].name = 'SpikerBox'

#Optional. Name the neuron chanels to specify locations.
s.neurons[0].name = 'femur'

# --- Plotting Neuron Channel Overview ---

print("\n=== Plotting Neuron Channel Overview ===")
s.plots.plot_channels(
    channels=s.channels, #Default is all channels anyway.
    time_window=None,  # Plot the entire duration
    title="Touch Pressure Experiment on Cockroach Leg",
    save_path=None,
    show=True
)

# --- Plotting Peri-Event Time Histograms (PETHs) with Raster Plots ---
print("\n=== Plotting Peri-Event Time Histograms (PETHs) with Raster Plots ===")

# Define a list of event objects to generate PETHs and Raster Plots for
event_list = s.events  #  All 3: 'light', 'medium', 'heavy'

# Select the first neuron in the channel
neuron = s.neurons[0] if s.neurons else None

# Plot PETH with Raster for multiple events on a single plot
s.plots.plot_peth(
    neuron=neuron,           # Neuron to plot
    events=event_list,       # List of Event objects
    epoch_window=(-0.5, 1.0),      # 500ms before to 1000ms after event
    bin_size=0.04,           # 40ms bins
    title="Peri-Event Time Histogram (PETH) with Raster Plots for Touch Pressure Events",
    save_path=None,
    show=True
)

