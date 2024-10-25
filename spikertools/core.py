# spikertools/core.py

import numpy as np
from scipy.io import wavfile
from spikertools.plots import Plots
import os
import re
from datetime import datetime

class Session:
    def __init__(self, wav_file_path, events_file_path=None):
        self.wav_file = wav_file_path
        self.sample_rate, self.data = wavfile.read(wav_file_path)
        print(f"Loaded WAV file: {wav_file_path}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Data length: {len(self.data)} samples")

        if events_file_path is None:
            # Infer events file path
            events_file = wav_file_path.replace('.wav', '-events.txt')
        else:
            events_file = events_file_path

        print(f"Looking for events file: {events_file}")

        # Initialize events and neurons before loading
        self.events = []
        self.neurons = []

        # Initialize the Plots class
        self.plots = Plots(self)
        
        if os.path.exists(events_file):
            self._load_events(events_file)
            print(f"Loaded {len(self.events)} events and {len(self.neurons)} neurons.")
        else:
            print("No events file found.")

        # Initialize other attributes
        self.channels = self._initialize_channels()
        self.datetime = self._extract_datetime()

    def _load_events(self, events_file):
        with open(events_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    continue
                name, timestamp = parts
                name = name.strip()
                try:
                    timestamp = float(timestamp.strip())
                except ValueError:
                    print(f"Invalid timestamp: {timestamp} in line: {line}")
                    continue
                self._add_event(name, timestamp)

    def _add_event(self, name, timestamp):
        # Normalize the event name by stripping leading/trailing spaces
        name = name.strip()
        
        # Assign colors to events and neurons
        color = 'k'  # Default color is black
        color_map = {
            '1': 'r', '2': 'g', '3': 'b', '4': 'c', '5': 'm',
            'Open': 'orange', 'Close': 'purple',
        }
        if name in color_map:
            color = color_map[name]

        # Check if event already exists
        for event in self.events:
            if event.name == name:
                event.timestamps.append(timestamp)
                return
        
        # Handle threshold events for neurons
        if 'thresh' in name.lower():
            # Use regex to parse threshold events
            match = re.match(r'_(neuron\d+)thresh(\w+)-(\d+)', name, re.IGNORECASE)
            if match:
                neuron_id, thresh_type_partial, thresh_value_str = match.groups()
                thresh_value = -int(thresh_value_str)  # Assuming thresholds are negative

                # Find the neuron that corresponds to this threshold
                target_neuron = None
                for neuron in self.neurons:
                    if neuron_id in neuron.name:
                        target_neuron = neuron
                        break
                
                if target_neuron:
                    if 'hig' in thresh_type_partial.lower():
                        target_neuron.thresh_high = thresh_value
                        #print(f"Set high threshold for {target_neuron.name} to {thresh_value}")
                    elif 'low' in thresh_type_partial.lower():
                        target_neuron.thresh_low = thresh_value
                        #print(f"Set low threshold for {target_neuron.name} to {thresh_value}")
                    else:
                        print(f"Unknown threshold type in event name: {name}")
                else:
                    print(f"Neuron '{neuron_id}' not found for threshold event: {name}")
                return  # Threshold event processed; exit the method
        
        # Check if neuron already exists for spike events
        if name.startswith('_'):
            for neuron in self.neurons:
                if neuron.name == name:
                    neuron.timestamps.append(timestamp)
                    return
            
            # If not, create a new neuron
            neuron = Neuron(name, timestamps=[timestamp], color=color)
            self.neurons.append(neuron)
            #print(f"Added neuron: '{name}' with initial spike at timestamp: {timestamp}")
        else:
            # If not a neuron, treat as a regular event
            event = Event(name, timestamps=[timestamp], color=color)
            self.events.append(event)
            #print(f"Added event: '{name}' at timestamp: {timestamp}")

    def _initialize_channels(self):
        # Initialize channels based on the data
        channels = []
        if self.data.ndim == 1:
            channel = Channel(self.data, sample_rate=self.sample_rate, number=0)
            channels.append(channel)
        else:
            for i in range(self.data.shape[1]):
                channel_data = self.data[:, i]
                channel = Channel(channel_data, sample_rate=self.sample_rate, number=i)
                channels.append(channel)
        return channels

    def _extract_datetime(self):
        # Extract datetime from the filename
        basename = os.path.basename(self.wav_file)
        match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2}\.\d{2})', basename)
        if match:
            date_str, time_str = match.groups()
            date_parts = [int(part) for part in date_str.split('-')]
            time_parts = [int(part) for part in time_str.split('.')]
            return datetime(*date_parts, *time_parts)
        return None


class Channel:
    def __init__(self, data=[], sample_rate=None, number=0, name='', color='k'):
        self.data = data  # NumPy array
        self.sample_rate = sample_rate
        self.number = number
        self.name = name or f'Channel {number}'
        self.color = color
        self.filters_applied = []

        if self.sample_rate:
            self._t = np.arange(0, len(self.data) / self.sample_rate, 1 / self.sample_rate)
        else:
            self._t = np.arange(len(self.data))

    @property
    def time(self):
        return self._t

    def filter(self, ftype='hp', cutoff=300, order=2):
        from scipy import signal

        nyquist = 0.5 * self.sample_rate
        if isinstance(cutoff, list) and ftype in ['bp', 'br']:
            cutoff = [c / nyquist for c in cutoff]
        elif isinstance(cutoff, (int, float)):
            cutoff = cutoff / nyquist
        else:
            raise ValueError("Invalid cutoff format.")

        if ftype == 'hp':
            b, a = signal.butter(order, cutoff, btype='high')
        elif ftype == 'lp':
            b, a = signal.butter(order, cutoff, btype='low')
        elif ftype == 'bp':
            b, a = signal.butter(order, cutoff, btype='bandpass')
        elif ftype == 'br':
            b, a = signal.butter(order, cutoff, btype='bandstop')
        elif ftype == 'n':  # Notch filter
            Q = 30.0  # Quality factor
            b, a = signal.iirnotch(cutoff, Q)
        else:
            raise ValueError("Unsupported filter type.")

        self.data = signal.filtfilt(b, a, self.data)
        self.filters_applied.append({'type': ftype, 'cutoff': cutoff, 'order': order})
        return self

    def normalize(self, method='zscore', norm_value=None):
        if method == 'mean':
            self.data = self.data - np.mean(self.data)
        elif method == 'std':
            self.data = self.data / np.std(self.data)
        elif method == 'zscore':
            self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        elif method == 'scalar':
            if norm_value is None:
                raise ValueError("norm_value must be provided for scalar normalization.")
            self.data = self.data * norm_value
        else:
            raise ValueError("Unsupported normalization method.")
        return self

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def std(self):
        return np.std(self.data)

class Event:
    def __init__(self, name, timestamps=None, color='k'):
        self.name = name
        self.timestamps = timestamps or []
        self.color = color

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def inter_event_intervals(self):
        if len(self.timestamps) < 2:
            return []
        return np.diff(sorted(self.timestamps))

    def event_rate(self):
        if len(self.timestamps) < 2:
            return np.nan
        total_time = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / total_time if total_time > 0 else np.nan

class Neuron:
    def __init__(self, name, timestamps=None, color='k'):
        self.name = name
        self.timestamps = timestamps or []
        self.color = color
        self.thresh_high = None  # Added high threshold property
        self.thresh_low = None   # Added low threshold property

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def inter_spike_intervals(self):
        if len(self.timestamps) < 2:
            return []
        return np.diff(sorted(self.timestamps))

    def firing_rate(self):
        if len(self.timestamps) < 2:
            return np.nan
        total_time = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / total_time if total_time > 0 else np.nan


