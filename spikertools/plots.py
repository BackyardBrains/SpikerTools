# spikertools/plots.py

import matplotlib.pyplot as plt
import numpy as np
import logging
import seaborn as sns  # Import seaborn
   
# Set global plot style using seaborn
sns.set_style('whitegrid')  # Use seaborn's 'whitegrid' style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

class Plots:
    def __init__(self, session):
        """
        Initializes the Plots class with a reference to a Session instance.
        
        Parameters:
        - session (Session): The Session object containing data and events.
        """
        self.session = session
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def plot_channels(self, channels=None, time_window=None, title=None, save_path=None, show=True):
         """
         Plots EEG channel data within a specified time window.
         
         Parameters:
         - channels (list of Channel): List of Channel objects to plot. Defaults to all channels.
         - time_window (tuple): Absolute time window for plotting (start_time, end_time) in seconds.
         - title (str): Title of the plot.
         - save_path (str): Path to save the plot.
         - show (bool): Whether to display the plot.
         """
         if channels is None:
             channels = self.session.channels
         
         plt.figure(figsize=(12, 6))
         
         for channel in channels:
             # Extract data within the time window
             if time_window:
                 start_time, end_time = time_window
                 start_idx = int(start_time * channel.sample_rate)
                 end_idx = int(end_time * channel.sample_rate)
                 data = channel.data[start_idx:end_idx]
                 times = np.linspace(start_time, end_time, end_idx - start_idx)
             else:
                 data = channel.data
                 times = np.arange(len(data)) / channel.sample_rate
             
             plt.plot(times, data, label=channel.name, color=channel.color)
         
         # Plot event markers if present
         for event in self.session.events:
             for timestamp in event.timestamps:
                 if time_window:
                     if time_window[0] <= timestamp <= time_window[1]:
                         plt.axvline(x=timestamp, color=event.color, linestyle='--', alpha=0.7)
                 else:
                     plt.axvline(x=timestamp, color=event.color, linestyle='--', alpha=0.7)
         
         plt.xlabel('Time (s)')
         plt.ylabel('Amplitude')
         if title:
             plt.title(title)
         else:
             plt.title('EEG Channel Overview')
         
         # Remove duplicate labels in legend
         handles, labels = plt.gca().get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         if by_label:
             plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
         
         plt.tight_layout()
         
         if save_path:
             plt.savefig(save_path)
             self.logger.info(f"Session plot saved to {save_path}.")
         if show:
             plt.show()
         else:
             plt.close()

    def plot_spectrogram(self, channel, freq_range=(0, 50), events=None, time_window=None, title=None, save_path=None, show=True):
        """
        Plots the spectrogram of a selected channel with optional event markers.
        
        Parameters:
        - channel: Channel to analyze.
        - freq_range (tuple): Frequency range to display (min_freq, max_freq).
        - events (list): List of events to mark on the spectrogram.
        - time_window (tuple, optional): Time window in seconds as (start, end). Defaults to entire duration.
        - title (str, optional): Title of the spectrogram. If None, a default title is used.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        
        Returns:
        - None
        """
        from scipy.signal import spectrogram

        if channel is None:
            channel = self.session.channels[0]

        sample_rate = channel.sample_rate

        if time_window:
            start_time, end_time = time_window
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            data_segment = channel.data[start_idx:end_idx]
            if len(data_segment) == 0:
                self.logger.error(f"No data in the specified time window: {time_window} seconds.")
                return
            f, t, Sxx = spectrogram(data_segment, fs=sample_rate, nperseg=1024, noverlap=512)
            # Adjust time axis to absolute time
            t = t + start_time
        else:
            f, t, Sxx = spectrogram(channel.data, fs=sample_rate, nperseg=1024, noverlap=512)

        # Limit frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        # Set plot title
        if title is None:
            title = 'Spectrogram'
        plt.title(title)

        plt.colorbar(label='Power/Frequency (dB/Hz)')

        # Plot event markers
        if events:
            for event in events:                
                for timestamp in event.timestamps:
                    if time_window:
                        if start_time <= timestamp <= end_time:
                            plt.axvline(x=timestamp, color=event.color, linestyle='--', label=event.name)
                    else:
                        plt.axvline(x=timestamp, color=event.color, linestyle='--', label=event.name)

        # Remove duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Spectrogram plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_average_power(self, events=None, freq_range=(0, 50), epoch_window=(0, 5), channel=None, save_path=None, show=True):
         """
         Plots the average power spectrum over epochs for specified events.
         
         Parameters:
         - events (list of Event): List of Event objects to compare.
         - freq_range (tuple): Frequency range for analysis (min_freq, max_freq) in Hz.
         - epoch_window (tuple): Absolute time window around each event (start, end) in seconds.
         - channel (Channel): Channel to analyze.
         - save_path (str): Path to save the plot.
         - show (bool): Whether to display the plot.
         """
         from scipy.signal import welch

         # Check if channel is None
         if channel is None:
            channel = self.session.channels[0]

         # If channel is a list, check its length
         if isinstance(channel, list):
            if len(channel) == 1:
                # Extract the single channel from the list
                channel = channel[0]
            else:
                self.logger.warning("Too many channels specified, using first channel.")
                channel = channel[0]
        
         sample_rate = channel.sample_rate
         start_samples = int(epoch_window[0] * sample_rate)
         end_samples = int(epoch_window[1] * sample_rate)
 
         plt.figure(figsize=(10, 6))
 
         for event in events:
             if not isinstance(event, object):
                 self.logger.warning(f"Expected SpikeEvent object, got {type(event)}. Skipping.")
                 continue
             
             epochs = []
             for timestamp in event.timestamps:
                 idx = int(timestamp * sample_rate)
                 start_idx = idx + start_samples
                 end_idx = idx + end_samples
                 if start_idx < 0 or end_idx > len(channel.data):
                     continue
                 epoch = channel.data[start_idx:end_idx]
                 epochs.append(epoch)
             if not epochs:
                 continue
             # Concatenate epochs for PSD calculation
             epochs_data = np.concatenate(epochs)
             f, Pxx = welch(epochs_data, fs=sample_rate, nperseg=1024)
             # Limit frequency range
             freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
             plt.plot(f[freq_mask], 10 * np.log10(Pxx[freq_mask]), label=event.name, color=event.color)
 
         plt.xlabel('Frequency [Hz]')
         plt.ylabel('Power Spectral Density (dB/Hz)')
         plt.title('Average Power Spectrum')
         plt.legend()
         plt.grid(True)
         plt.tight_layout()
         if save_path:
             plt.savefig(save_path)
             self.logger.info(f"Average power spectrum plot saved to {save_path}.")
         if show:
             plt.show()
         else:
             plt.close()

    def plot_epochs(self, event=None, epoch_window=(-0.5, 1.0), channel=None, save_path=None, show=True):
        """
        Plots individual epochs of the channel data around specified events.
        
        Parameters:
        - event (Event): Event to plot.
        - epoch_window (tuple): Time window around the event (pre, post) in seconds.
        - channel (Channel): Channel to analyze.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        
        if event is None:
            self.logger.error("No event specified for epochs plot.")
            return
        
        # Check if channel is None
        if channel is None:
            self.logger.warning("No channel specified for ERP plot, using first channel.")
            channel = self.session.channels[0]

        # If channel is a list, check its length
        if isinstance(channel, list):
            if len(channel) == 1:
                # Extract the single channel from the list
                channel = channel[0]
            else:
                self.logger.warning("Too many channels specified, using first channel.")
                channel = channel[0]

        sample_rate = channel.sample_rate
        pre_samples = int(abs(epoch_window[0]) * sample_rate)
        post_samples = int(epoch_window[1] * sample_rate)
        epoch_length = pre_samples + post_samples
        time_axis = np.linspace(epoch_window[0], epoch_window[1], epoch_length)

        
        epochs = []
        for timestamp in event.timestamps:
            idx = int(timestamp * sample_rate)
            start_idx = idx - pre_samples
            end_idx = idx + post_samples
            if start_idx < 0 or end_idx > len(channel.data):
                continue
            epoch = channel.data[start_idx:end_idx]
            epochs.append(epoch)
        if not epochs:
            self.logger.warning("No epochs to plot.")
            return

        plt.figure(figsize=(10, 6))
        for i, epoch in enumerate(epochs):
            plt.plot(time_axis, epoch + i * np.std(epoch), color=channel.color)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Epochs around event "{event.name}"')
        plt.axvline(x=0, color='black', linestyle='--', label='Event Onset')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Epochs plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_raster(self, neuron_name, event_name=None, epoch_window=(-0.5, 1.0), other_events=None, save_path=None, show=True):
        """
        Plots a raster plot of neuronal spikes aligned to events.
        
        Parameters:
        - neuron_name (str): Name of the neuron.
        - event_name (str): Name of the event to align spikes.
        - epoch_window (tuple): Time window around the event (pre, post) in seconds.
        - other_events (list): List of other event names to mark in the plot.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        neuron = next((n for n in self.session.neurons if n.name == neuron_name), None)
        if not neuron:
            self.logger.error(f"Neuron '{neuron_name}' not found.")
            return

        if event_name:
            event = next((e for e in self.session.events if e.name == event_name), None)
            if not event:
                self.logger.error(f"Event '{event_name}' not found.")
                return
            event_timestamps = event.timestamps
        else:
            event_timestamps = [0]  # Align to time zero

        plt.figure(figsize=(10, 6))
        for trial_idx, event_time in enumerate(event_timestamps):
            epoch_start = event_time + epoch_window[0]
            epoch_end = event_time + epoch_window[1]
            spike_times = [t - event_time for t in neuron.timestamps if epoch_start <= t <= epoch_end]
            plt.vlines(spike_times, trial_idx + 0.5, trial_idx + 1.5, color=neuron.color)

            # Plot other events within the epoch
            if other_events:
                for other_event_name in other_events:
                    other_event = next((e for e in self.session.events if e.name == other_event_name), None)
                    if other_event:
                        other_event_times = [t - event_time for t in other_event.timestamps if epoch_start <= t <= epoch_end]
                        plt.scatter(other_event_times, [trial_idx + 1] * len(other_event_times), marker='o', color=other_event.color)

        plt.xlabel('Time (s)')
        plt.ylabel('Trial')
        plt.title(f'Raster Plot of Neuron "{neuron_name}" aligned to Event "{event_name}"')
        plt.axvline(x=0, color='black', linestyle='--', label='Event Onset')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Raster plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_peth(self, neuron=None, events=None, epoch_window=(-1.0, 1.0), bin_size=0.05, title=None, save_path=None, show=True):
        """
        Plots the Peri-Event Time Histogram (PETH) with Raster Plot for the specified neuron and events.
        
        Parameters:
        - neuron (Neuron, optional): The neuron object to plot spikes for. Defaults to first neuron.
        - events (list of Event, optional): List of event objects to align spikes to. Defaults to all events.
        - epoch_window (tuple): Time window around the event (start, end) in seconds.
        - bin_size (float): Size of each histogram bin in seconds.
        - title (str, optional): Title of the PETH plot. If None, a default title is used.
        - save_path (str, optional): Path to save the combined PETH and Raster plot.
        - show (bool): Whether to display the plot. Defaults to True.
        
        Returns:
        - None
        """
        # Default assignments
        if neuron is None:
            if not self.session.neurons:
                self.logger.error("No neurons available in the session.")
                return
            neuron = self.session.neurons[0]
            self.logger.info("No neuron specified. Using the first neuron in the session.")
        
        if events is None:
            events = self.session.events
            self.logger.info("No events specified. Using all events in the session.")
        
        if not events:
            self.logger.error("No events provided for PETH plotting.")
            return
        
        # Initialize figure with two subplots: PETH and Raster
        fig, (ax_peth, ax_raster) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                                gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot PETH
        for event in events:
            # Align spikes relative to the event
            aligned_spikes = []
            for timestamp in event.timestamps:
                aligned = np.array(neuron.timestamps) - timestamp
                # Select spikes within the window
                spikes_in_window = aligned[(aligned >= epoch_window[0]) & (aligned <= epoch_window[1])]
                aligned_spikes.extend(spikes_in_window)
            
            if not aligned_spikes:
                self.logger.warning(f"No spikes found for event '{event.name}' within the specified window.")
                continue
            
            # Create histogram bins
            bins = np.arange(epoch_window[0], epoch_window[1] + bin_size, bin_size)
            counts, _ = np.histogram(aligned_spikes, bins=bins)
            
            # Normalize to firing rate (spikes per second)
            firing_rate = counts / (len(event.timestamps) * bin_size)
            
            # Plot as stairs
            ax_peth.step(bins[:-1], firing_rate, where='post', label=event.name, color=event.color)
        
        # Customize PETH plot
        ax_peth.set_ylabel('Firing Rate (Hz)')
        if title:
            ax_peth.set_title(title)
        else:
            ax_peth.set_title('Peri-Event Time Histogram (PETH)')
        ax_peth.legend()
        ax_peth.grid(True)
        
        # Plot Raster
        y_ticks = []
        y_labels = []
        for idx, event in enumerate(events):
            for event_num, timestamp in enumerate(event.timestamps):
                aligned_spikes = np.array(neuron.timestamps) - timestamp
                spikes_in_window = aligned_spikes[(aligned_spikes >= epoch_window[0]) & (aligned_spikes <= epoch_window[1])]
                ax_raster.scatter(spikes_in_window, 
                                  np.full_like(spikes_in_window, idx * len(event.timestamps) + event_num),
                                  marker='|', color=event.color, s=100)
            y_ticks.append(idx * len(event.timestamps) + len(event.timestamps)/2 - 0.5)
            y_labels.append(event.name)
        
        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_labels)
        ax_raster.set_xlabel('Time relative to event (s)')
        ax_raster.set_ylabel('Events')
        ax_raster.grid(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            self.logger.info(f"PETH and Raster plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_intervals_histogram(self, data_type='event', name=None, bins=50, save_path=None, show=True):
        """
        Plots a histogram of inter-event or inter-spike intervals.
        
        Parameters:
        - data_type (str): 'event' or 'neuron'.
        - name (str): Name of the event or neuron.
        - bins (int): Number of bins for the histogram.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        if data_type == 'event':
            obj = next((e for e in self.session.events if e.name == name), None)
            if not obj:
                self.logger.error(f"Event '{name}' not found.")
                return
            intervals = obj.inter_event_intervals()
            xlabel = 'Inter-Event Interval (s)'
            title = f'Inter-Event Interval Histogram for "{name}"'
        elif data_type == 'neuron':
            obj = next((n for n in self.session.neurons if n.name == name), None)
            if not obj:
                self.logger.error(f"Neuron '{name}' not found.")
                return
            intervals = obj.inter_spike_intervals()
            xlabel = 'Inter-Spike Interval (s)'
            title = f'Inter-Spike Interval Histogram for "{name}"'
        else:
            self.logger.error("Invalid data_type. Must be 'event' or 'neuron'.")
            return

        # Fixing the conditional check
        if intervals.size == 0:
            self.logger.warning(f"No intervals found for {data_type} '{name}'.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(intervals, bins=bins, color=obj.color, alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Intervals histogram saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_erp(self, events=None, epoch_window=(-0.5, 1.0), channel=None, title=None, save_path=None, show=True):
        """
        Plots Event-Related Potentials (ERP) for specified events.
        
        Parameters:
        - events (list of Event): List of Event objects to plot ERP for.
        - epoch_window (tuple): Relative time window around each event (start, end) in seconds.
        - channel (Channel): Channel object to analyze.
        - title (str): Title of the ERP plot.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
         # Check if channel is None
        if channel is None:
            self.logger.warning("No channel specified for ERP plot, using first channel.")
            channel = self.session.channels[0]

        # If channel is a list, check its length
        if isinstance(channel, list):
            if len(channel) == 1:
                # Extract the single channel from the list
                channel = channel[0]
            else:
                self.logger.warning("Too many channels specified, using first channel.")
                channel = channel[0]
        
        sample_rate = channel.sample_rate
        pre_samples = int(abs(epoch_window[0]) * sample_rate)
        post_samples = int(epoch_window[1] * sample_rate)
        epoch_length = pre_samples + post_samples
        time_axis = np.linspace(epoch_window[0], epoch_window[1], epoch_length)
 
        plt.figure(figsize=(12, 6))
        
        for event in events:
    
            epochs = []
            for timestamp in event.timestamps:
                idx = int(timestamp * sample_rate)
                start_idx = idx - pre_samples
                end_idx = idx + post_samples
                if start_idx < 0 or end_idx > len(channel.data):
                    continue
                epoch = channel.data[start_idx:end_idx]
                epochs.append(epoch)
            if not epochs:
                continue
            # Average across epochs
            erp = np.mean(epochs, axis=0)
            plt.plot(time_axis, erp, label=event.name, color=event.color)
 
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        if title:
            plt.title(title)
        else:
            plt.title('Event-Related Potential (ERP)')
        plt.axvline(x=0, color='black', linestyle='--', label='Event Onset')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"ERP plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

