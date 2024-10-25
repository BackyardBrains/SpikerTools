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

    def plot_session(self, channels=None, time_window=None, save_path=None, show=True):
        """
        Plot an overview of the entire session with channels and event markers.
        
        Parameters:
        - channels (list of int, optional): List of channel indices to include. Defaults to all channels.
        - time_window (tuple of float, optional): Time window in seconds as (start, end). Defaults to entire duration.
        - save_path (str, optional): File path to save the plot. If None, displays the plot interactively.
        - show (bool): Whether to display the plot. Defaults to True.
        
        Returns:
        - None
        """
        if channels is None:
            channels = list(range(len(self.session.channels)))  # All channels

        sample_rate = self.session.sample_rate
        if time_window:
            start_time, end_time = time_window
        else:
            total_samples = len(self.session.channels[0].data)
            start_time = 0
            end_time = total_samples / sample_rate

        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        time_axis = np.linspace(start_time, end_time, end_idx - start_idx)

        plt.figure(figsize=(12, 8))

        offset = 0
        y_offsets = []
        for idx in channels:
            channel = self.session.channels[idx]
            data_segment = channel.data[start_idx:end_idx]
            data_offset = data_segment + offset
            plt.plot(time_axis, data_offset, label=channel.name, color=channel.color)
            y_offsets.append(offset)
            offset += np.max(data_segment) - np.min(data_segment) + np.std(data_segment)

        # Plot event markers
        for event in self.session.events:
            for timestamp in event.timestamps:
                if start_time <= timestamp <= end_time:
                    plt.axvline(x=timestamp, color=event.color, linestyle='--', label=event.name)

        plt.title('Session Overview with Event Markers')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Session plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_spectrogram(self, channel_index=0, freq_range=(0, 50), event_names=None, save_path=None, show=True):
        """
        Plots the spectrogram of a selected channel with optional event markers.
        
        Parameters:
        - channel_index (int): Index of the channel to analyze.
        - freq_range (tuple): Frequency range to display (min_freq, max_freq).
        - event_names (list): List of event names to mark on the spectrogram.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        from scipy.signal import spectrogram
        
        channel = self.session.channels[channel_index]
        f, t, Sxx = spectrogram(channel.data, fs=channel.sample_rate, nperseg=1024, noverlap=512)

        # Limit frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        plt.colorbar(label='Power/Frequency (dB/Hz)')

        # Plot event markers
        if event_names:
            for event_name in event_names:
                event = next((e for e in self.session.events if e.name == event_name), None)
                if event:
                    for timestamp in event.timestamps:
                        plt.axvline(x=timestamp, color=event.color, linestyle='--', label=event_name)

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

    def plot_average_power(self, event_names, freq_range=(0, 50), epoch_window=(0, 5), channel_index=0, save_path=None, show=True):
        """
        Plots the average power spectrum over epochs for specified events.
        
        Parameters:
        - event_names (list): List of event names to compare.
        - freq_range (tuple): Frequency range for analysis (min_freq, max_freq).
        - epoch_window (tuple): Time window around each event (start, end) in seconds.
        - channel_index (int): Index of the channel to analyze.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        from scipy.signal import welch

        channel = self.session.channels[channel_index]
        sample_rate = channel.sample_rate
        start_samples = int(epoch_window[0] * sample_rate)
        end_samples = int(epoch_window[1] * sample_rate)

        plt.figure(figsize=(10, 6))

        for event_name in event_names:
            event = next((e for e in self.session.events if e.name == event_name), None)
            if not event:
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
            plt.plot(f[freq_mask], 10 * np.log10(Pxx[freq_mask]), label=event_name, color=event.color)

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

    def plot_epochs(self, event_name, epoch_window=(-0.5, 1.0), channel_index=0, save_path=None, show=True):
        """
        Plots individual epochs of the channel data around specified events.
        
        Parameters:
        - event_name (str): Name of the event to align epochs.
        - epoch_window (tuple): Time window around the event (pre, post) in seconds.
        - channel_index (int): Index of the channel to analyze.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        channel = self.session.channels[channel_index]
        sample_rate = channel.sample_rate
        pre_samples = int(abs(epoch_window[0]) * sample_rate)
        post_samples = int(epoch_window[1] * sample_rate)
        epoch_length = pre_samples + post_samples
        time_axis = np.linspace(epoch_window[0], epoch_window[1], epoch_length)

        event = next((e for e in self.session.events if e.name == event_name), None)
        if not event:
            self.logger.error(f"Event '{event_name}' not found.")
            return

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
        plt.title(f'Epochs around event "{event_name}"')
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

    def plot_peth(self, neuron_name, event_name, epoch_window=(-0.5, 1.0), bin_size=0.01, show_raster=True, show_peth=True, save_path=None, show=True):
        """
        Plots a Peri-Event Time Histogram (PETH) with optional raster plot.
        
        Parameters:
        - neuron_name (str): Name of the neuron.
        - event_name (str): Name of the event to align spikes.
        - epoch_window (tuple): Time window around the event (pre, post) in seconds.
        - bin_size (float): Bin size for the histogram in seconds.
        - show_raster (bool): Whether to include the raster plot.
        - show_peth (bool): Whether to include the PETH.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        neuron = next((n for n in self.session.neurons if n.name == neuron_name), None)
        if not neuron:
            self.logger.error(f"Neuron '{neuron_name}' not found.")
            return

        event = next((e for e in self.session.events if e.name == event_name), None)
        if not event:
            self.logger.error(f"Event '{event_name}' not found.")
            return

        event_timestamps = event.timestamps
        all_spike_times = []

        for event_time in event_timestamps:
            epoch_start = event_time + epoch_window[0]
            epoch_end = event_time + epoch_window[1]
            spike_times = [t - event_time for t in neuron.timestamps if epoch_start <= t <= epoch_end]
            all_spike_times.extend(spike_times)

        fig, axs = plt.subplots(2 if show_raster and show_peth else 1, 1, figsize=(10, 6), sharex=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        if show_raster:
            # Raster plot
            for trial_idx, event_time in enumerate(event_timestamps):
                epoch_start = event_time + epoch_window[0]
                epoch_end = event_time + epoch_window[1]
                spike_times = [t - event_time for t in neuron.timestamps if epoch_start <= t <= epoch_end]
                axs[0].vlines(spike_times, trial_idx + 0.5, trial_idx + 1.5, color=neuron.color)
            axs[0].set_ylabel('Trial')
            axs[0].set_title(f'Raster Plot of Neuron "{neuron_name}" aligned to Event "{event_name}"')
            axs[0].axvline(x=0, color='black', linestyle='--')

        if show_peth:
            # PETH
            bins = np.arange(epoch_window[0], epoch_window[1] + bin_size, bin_size)
            counts, _ = np.histogram(all_spike_times, bins=bins)
            firing_rates = counts / (len(event_timestamps) * bin_size)
            axs[-1].bar(bins[:-1], firing_rates, width=bin_size, color=neuron.color, align='edge')
            axs[-1].set_xlabel('Time (s)')
            axs[-1].set_ylabel('Firing Rate (Hz)')
            axs[-1].set_title(f'PETH of Neuron "{neuron_name}" aligned to Event "{event_name}"')
            axs[-1].axvline(x=0, color='black', linestyle='--')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"PETH plot saved to {save_path}.")
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
    
    def plot_erp(self, event_names, epoch_window=(-1.0, 1.0), channel_index=0, save_path=None, show=True):
        """
        Plots the Event-Related Potentials (ERPs) by averaging epochs around specified events.
        
        Parameters:
        - event_names (list of str): List of event names to include in the ERP plot.
        - epoch_window (tuple): Time window around the event (pre, post) in seconds.
        - channel_index (int): Index of the channel to analyze.
        - save_path (str): Path to save the plot.
        - show (bool): Whether to display the plot.
        """
        channel = self.session.channels[channel_index]
        sample_rate = channel.sample_rate
        pre_samples = int(abs(epoch_window[0]) * sample_rate)
        post_samples = int(epoch_window[1] * sample_rate)
        epoch_length = pre_samples + post_samples
        time_axis = np.linspace(epoch_window[0], epoch_window[1], epoch_length)
        
        plt.figure(figsize=(10, 6))
        
        for event_name in event_names:
            event = next((e for e in self.session.events if e.name == event_name), None)
            if not event:
                self.logger.error(f"Event '{event_name}' not found.")
                continue  # Skip this event and proceed to the next
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
                self.logger.warning(f"No epochs to plot for event '{event_name}'.")
                continue
            # Compute the ERP by averaging the epochs
            erp = np.mean(epochs, axis=0)
            # Plot the ERP
            plt.plot(time_axis, erp, color=event.color, label=event_name)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Event-Related Potentials (ERPs)')
        plt.axvline(x=0, color='black', linestyle='--', label='Event Onset')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"ERP plot saved to {save_path}.")
        if show:
            plt.show()
        else:
            plt.close()

