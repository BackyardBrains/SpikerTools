import numpy as np
from scipy.io import wavfile
import os
import re
from datetime import datetime

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

class Channel:
    def __init__(self, data=[], sample_rate=None, number=0, name='', color='k'):
        self.data = data  # NumPy array
        self.sample_rate = sample_rate
        self.number = number
        self._name = name or f'Channel {number}'
        self.color = color
        self.filters_applied = []
        self._container = None  # Reference to the Channels container

        if self.sample_rate:
            self._t = np.arange(0, len(self.data) / self.sample_rate, 1 / self.sample_rate)
        else:
            self._t = np.arange(len(self.data))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if self._name == new_name:
            return # No change needed

        old_name = self._name
        if self._container:
            # Ask the container to validate and perform the rename in its map
            # This will raise an error if new_name is invalid (e.g., duplicate)
            self._container._request_rename_channel(self, old_name, new_name)
        
        # If the container part was successful (or no container), update the internal name
        self._name = new_name

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

    def smooth(self, n=5):
        """
        Smooth the channel data using a moving average filter over n samples.
        Args:
            n (int): Number of samples for the moving average window.
        Returns:
            self: The Channel object with smoothed data.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        kernel = np.ones(n) / n
        self.data = np.convolve(self.data, kernel, mode='valid')
        return self

class Channels:
    """A container class for Channel objects that supports both numeric indexing and name-based lookup."""
    
    def __init__(self, channels_list): # Changed param name for clarity
        self._channels = []
        self._name_map = {}
        self._number_map = {}
        for channel_obj in list(channels_list): # Iterate over a copy
            channel_obj._container = self # Set container reference
            
            # Validate name and number uniqueness during initialization
            if channel_obj.name in self._name_map: # .name uses property getter
                raise ValueError(f"Duplicate channel name '{channel_obj.name}' during initialization.")
            if channel_obj.number in self._number_map:
                raise ValueError(f"Duplicate channel number {channel_obj.number} during initialization.")

            self._channels.append(channel_obj)
            self._name_map[channel_obj.name] = channel_obj
            self._number_map[channel_obj.number] = channel_obj
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._channels[key]
        elif isinstance(key, str):
            return self._name_map[key]
        elif isinstance(key, slice):
            return self._channels[key]
        else:
            raise KeyError("Invalid key type. Use int, str, or slice.")
    
    def __len__(self):
        return len(self._channels)
    
    def __iter__(self):
        return iter(self._channels)
    
    def names(self):
        """Returns a list of all channel names."""
        return list(self._name_map.keys())
    
    def numbers(self):
        """Returns a list of all channel numbers."""
        return list(self._number_map.keys())
    
    def has_channel(self, name):
        """Checks if a channel with the given name exists."""
        return name in self._name_map
    
    def has_channel_number(self, number):
        """Checks if a channel with the given number exists."""
        return number in self._number_map
    
    def append(self, channel):
        """Adds a new channel to the collection."""
        if channel.name in self._name_map:
            raise KeyError(f"Channel with name '{channel.name}' already exists.")
        if channel.number in self._number_map:
            raise KeyError(f"Channel number {channel.number} already exists.")
        
        self._channels.append(channel)
        self._name_map[channel.name] = channel
        self._number_map[channel.number] = channel
        channel._container = self  # Set the container reference

    def rename(self, old_name_key: str, new_name: str):
        """
        Rename a channel while preserving its data.
        
        Parameters:
        - old_name_key: Current name of the channel
        - new_name: New name for the channel
        """
        if old_name_key not in self._name_map:
            raise KeyError(f"Channel '{old_name_key}' not found for renaming.")
        
        channel_to_rename = self._name_map[old_name_key]
        # This assignment will trigger the Channel's name.setter property
        channel_to_rename.name = new_name

    def _request_rename_channel(self, channel_obj, old_name_in_channel, new_name_proposed):
        """
        Called by a Channel's name setter to update the container's map.
        Validates new_name_proposed and updates the internal _name_map.
        """
        # Ensure the channel_obj is indeed managed by this container under old_name_in_channel
        if old_name_in_channel not in self._name_map or self._name_map[old_name_in_channel] is not channel_obj:
            # This indicates an inconsistency, or the channel isn't properly in this container under that old name.
            # It might be a new channel whose name is being set after default initialization,
            # and old_name_in_channel was its default (e.g., "Channel 0").
            # If old_name_in_channel is not in the map, there's nothing to pop for it.
            # If it is in the map but points to a different object, that's a bigger issue.
            # For simplicity, we assume if old_name_in_channel is in the map, it must be channel_obj.
             if old_name_in_channel in self._name_map and self._name_map[old_name_in_channel] is not channel_obj:
                raise RuntimeError(
                    f"Internal inconsistency: Channel's reported old name '{old_name_in_channel}' "
                    f"is mapped to a different channel object in the container."
                )
            # If old_name_in_channel is not in _name_map, proceed to check new_name_proposed.
            # Nothing to pop for old_name_in_channel.

        # Check if new_name_proposed is already taken by *another* channel
        if new_name_proposed in self._name_map and self._name_map[new_name_proposed] is not channel_obj:
            raise KeyError(
                f"Cannot rename to '{new_name_proposed}': name already used by a different channel."
            )

        # Perform the update in the map
        if old_name_in_channel in self._name_map and self._name_map[old_name_in_channel] is channel_obj:
            self._name_map.pop(old_name_in_channel)
        
        self._name_map[new_name_proposed] = channel_obj
        # The channel_obj._name will be updated by its own setter after this method returns successfully.

    def create_channel(self, data, sample_rate, number, name=None, color='k'):
        """
        Create a new channel with the given parameters.
        
        Parameters:
        - data: Channel data array
        - sample_rate: Sample rate of the channel
        - number: Channel number
        - name: Name for the channel (default: 'Channel {number}')
        - color: Color for the channel (default: 'k' for black)
        
        Returns:
        - The newly created Channel object
        """
        if name is None:
            name = f'Channel {number}'
            
        if name in self._name_map:
            raise KeyError(f"Channel '{name}' already exists")
        if number in self._number_map:
            raise KeyError(f"Channel number {number} already exists")
            
        channel = Channel(data=data, sample_rate=sample_rate, number=number, name=name, color=color)
        self.append(channel)
        return channel

class Events:
    """A container class for Event objects that supports both numeric indexing and name-based lookup."""
    
    def __init__(self, events):
        self._events = list(events)
        self._name_map = {event.name: event for event in events}
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._events[key]
        elif isinstance(key, str):
            return self._name_map[key]
        elif isinstance(key, slice):
            return self._events[key]
        else:
            raise KeyError("Invalid key type. Use int, str, or slice.")
    
    def __len__(self):
        return len(self._events)
    
    def __iter__(self):
        return iter(self._events)
    
    def names(self):
        """Returns a list of all event names."""
        return list(self._name_map.keys())
    
    def has_event(self, name):
        """Checks if an event with the given name exists."""
        return name in self._name_map
    
    def append(self, event):
        """Adds a new event to the collection."""
        self._events.append(event)
        self._name_map[event.name] = event

    def get_timestamps(self, names):
        """
        Get all timestamps from events with the specified names.
        
        Parameters:
        - names: Can be a single event name or a list/set of event names
        
        Returns:
        - List of timestamps from all matching events
        """
        if isinstance(names, str):
            names = {names}
        elif isinstance(names, (list, tuple)):
            names = set(names)
            
        timestamps = []
        for name in names:
            if name in self._name_map:
                timestamps.extend(self._name_map[name].timestamps)
        return sorted(timestamps)

    def rename(self, old_name, new_name):
        """
        Rename an event while preserving its data.
        
        Parameters:
        - old_name: Current name of the event
        - new_name: New name for the event
        """
        if old_name not in self._name_map:
            raise KeyError(f"Event '{old_name}' not found")
        if new_name in self._name_map:
            raise KeyError(f"Event '{new_name}' already exists")
            
        event = self._name_map[old_name]
        event.name = new_name
        del self._name_map[old_name]
        self._name_map[new_name] = event

    def create_event(self, name, timestamps, color='k'):
        """
        Create a new event with the given name and timestamps.
        
        Parameters:
        - name: Name for the new event
        - timestamps: List of timestamps for the event
        - color: Color for the event (default: 'k' for black)
        
        Returns:
        - The newly created Event object
        """
        if name in self._name_map:
            raise KeyError(f"Event '{name}' already exists")
            
        event = Event(name, timestamps=timestamps, color=color)
        self.append(event)
        return event

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
        from spikertools.plots import Plots
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
                    elif 'low' in thresh_type_partial.lower():
                        target_neuron.thresh_low = thresh_value
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
        else:
            # If not a neuron, treat as a regular event
            event = Event(name, timestamps=[timestamp], color=color)
            self.events.append(event)

    def _initialize_channels(self):
        # Initialize channels based on the data
        channels = Channels([])  # Initialize empty Channels container
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