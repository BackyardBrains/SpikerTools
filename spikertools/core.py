# spikertools/core.py

from spikertools.models import Event, Neuron, Channel, Session, Events, Channels
from spikertools.plots import Plots
import numpy as np
from scipy.io import wavfile
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
        self.events = Events([])  # Changed from list to Events
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
        if self.events.has_event(name):
            self.events[name].timestamps.append(timestamp)
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
        channels = Channels([])  # Initialize empty Channels container (this uses models.Channels)
        if self.data.ndim == 1:
            channel = Channel(self.data, sample_rate=self.sample_rate, number=0) # This will now use models.Channel
            channels.append(channel)
        else:
            for i in range(self.data.shape[1]):
                channel_data = self.data[:, i]
                channel = Channel(channel_data, sample_rate=self.sample_rate, number=i) # This will now use models.Channel
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


