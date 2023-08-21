import numpy as np
from scipy import signal
import math

class Channel:
    def __init__(self, data=[], name='', number=0, color='k', sampleRate=None, filters=None ):

        class Filters:
            def __init__(self):
                self.hardware = []
                self.software = []
                self.analysis = []

        self._data = data  # data extracted from WAV file, default is empty list
        self._name = name
        self._number = number
        self._color = color
        self._filters = filters
        self._sampleRate = sampleRate
        if filters is None:
            self._filters = Filters()
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data_in):
        self._data = data_in
        return self._data
    
    @property
    def filters(self):
        return self._filters
    @filters.setter  
    def filters(self, key):
        self._filters = key
        return self._filters

    
    @property
    def sampleRate (self):
        return self._sampleRate 
    @sampleRate.setter
    def sampleRate(self, fs_in):
        self._sampleRate  = float(fs_in)
        self._t = np.arange(0, len(self._data) / self.sampleRate, 1 / self.sampleRate )
        return self._sampleRate

    @property
    def name(self):
        return self._name if self._name is not None else ""
    @name.setter
    def name(self, key):
        if not isinstance(key, str):
            raise ValueError("Invalid name format. Please provide a string value.")
        self._name = key

    @property
    def number(self):
        return self._number
    @number.setter
    def number(self, key):
        if not isinstance(key, int):
            raise ValueError("Invalid number format. Please provide an integer value.")
        self._number = key
        return  self._number

    @property
    def time(self):
        return self._t

    @property
    def color(self):
        return self._color
    @color.setter
    def color(self, color_in):
        self._color = color_in
        return self._color

    @property
    def mean(self):
        return np.mean(self._data)

    @property
    def std(self):
        return np.std(self._data)
    @property
    def std(self, interval=None):
        if interval:
            spec_interval = self._data[interval[0] * self.fs:interval[1] * self.fs]
            return np.std(spec_interval)
        else:
            return np.std(self._data)


    # tool functions for channel objects

    # filtering function (lowpass, highpass, notch, bandpass, band reject)
    # modifies self._data and self._filterfreqs
    # Updates: Grouped 'hp' and 'lp' in a single condition, similarly grouped 'bp' and 'br' in a single condition.
    def filter(self, ftype='hp', cutoff=[300], filter_order=2):
        if cutoff > self._sampleRate / 2:
            raise ValueError(f"Filter frequency should not exceed Nyquist: {self._sampleRate / 2} ")

        assert isinstance(cutoff, (int, float)) or ftype in ['bp', 'br'] and isinstance(cutoff, list), "Cutoff should be either int or float or list(only in case of 'bp' or 'br')"

        if ftype in ['hp', 'lp']:
            b, a = signal.butter(filter_order, cutoff, ftype, fs=self._sampleRate)
            out = signal.filtfilt(b, a, self._data)
            self._data = out
            if ftype == 'hp':
                self._filters.analysis = [ cutoff,  self._sampleRate ]
            else:
                self._filters.analysis = [ 0 , cutoff ]
        elif ftype in ['bp', 'br']:
            b, a = signal.butter(filter_order, cutoff, 'bandpass' if ftype == 'bp' else 'bandstop', fs=self.fs)
            out = signal.filtfilt(b, a, self._data)
            self._filters.analysis = cutoff
            self._data = out

        elif ftype == 'n':  # notch filter
            Q = math.sqrt((cutoff + 1) * (cutoff - 1)) / 2
            b_notch, a_notch = signal.iirnotch(cutoff, Q, self.fs)
            out = signal.filtfilt(b_notch, a_notch, self._data)
            self._filters.analysis = [cutoff, cutoff] 
            self._data = out
        else:
            raise ValueError("Incorrect filter type specified!")
        
        return self

 
    # downsampling function, applies an anti-aliasing filter first, then downsamples
    # modifies self._fs, self._data, and if anti-aliasing filter is less than lowpass filter, self._filterfreq, and the time vector
    # Update: Removed redundant function call
    def decimate(self, decim_factor):
        self._data = signal.decimate(self._data, decim_factor)
        self._sampleRate /= decim_factor
        self._filters.software = min(self._filters.software, self._sampleRate / 2)
        self._t = np.arange(0, len(self._data) / self._sampleRate, 1 / self._sampleRate)
        return self

    # normalization function
    # Update: simplifies the conditionals
    def normalize(self, norm_type, norm_value=None):
        if norm_type == 'mean': # Mean subtracted
            self._data -= np.mean(self._data)
        elif norm_type == 'std': # Dived through subtracted
            self._data /= np.std(self._data)
        elif norm_type == 'zscore': # Mean Subtracted, Dived through by STD.
            self._data -= self.mean
            self._data /= np.std(self._data)
        elif norm_type == 'scalar': #multply by a scaler
            assert isinstance(norm_value, (int, float)), "Must specify number for scalar"
            self._data *= norm_value
        else:
            raise ValueError("Incorrect normalization type specified")
        return self
