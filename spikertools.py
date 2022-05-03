#Spiker Tools

#from operator import sub
#from xml.sax.handler import property_declaration_handler
import numpy as np
from scipy import signal  
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
from datetime import datetime
import random
import time
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#from pydub import AudioSegment
#from tinytag import TinyTag
import collections
import os.path


""" Events Class """

class Events(list):
    def __init__(self):
         self.items = []

    def append(self, item):
         self.items.append (item)

    def __repr__(self):
        return self.items

    def __str__(self):
        return self.eventNames

    def __len__(self):
        return len(self.items)

    def __contains__(self, key):
        return any(key is item or key == item for item in self.items)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        self.idx += 1
        try:
            return self.items[self.idx-1]
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.

    def __getitem__(self, key):
        #print("__getitem__ Events" )
        if type(key) == str:
            for count, value in enumerate(self.items):
                if str(value) == key:
                    return (self.items[count])
        if type(key) == int:
            return (self.items[key])

    def __setitem__(self, key, newvalue):
        #print("__setitem__ Events" )
        if key in self.items:
            self.items[key].timestamps.append(float(newvalue))
        else:
            e = Event(key)
            e.timestamps.append(float(newvalue))
            self.items.append(e)

    @property
    def eventNames(self):
        return str(self.items)
 
class Event:
    def __init__(self, key = None):
        if isinstance(key, str):
            self._name = key
        else: 
            self._name = "event"
            print("Empty [Event] class created.")
        self._timestamps = []

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, Event):
            if other.name == self.name:
                return True
            else:
                return False
        if isinstance(other, str):
            if other == self.name:
                return True
            else:
                return False
        else:
            return False

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, key):
        self._name = key
    
    @property
    def timestamps(self):
        return self._timestamps

"""Channel Class """
class Channel:
    def __init__(self, data = None, fs= None, filterfreqs= None, label= None, color= None):
        self._data = data if data is not None else [] #data extracted from WAV file, default is empty list
        self._fs = fs if fs is not None else 0 #sampling frequency extracted from WAV file, default is 10,000 Hz
        self._filterfreqs = filterfreqs if filterfreqs is not None else [0,10000] # bandpass filter cutoff frequencies, set by user, default is 0 to 10000 Hz
        self._label = label if label is not None else "channel" #channel string label set by user, default is channel 
        self._color = color if color is not None else 'k' #set by user
        self._t = np.arange(0, (len(self._data)/self._fs), (1/self._fs)) #time vector, elaborated from sample rate and duration of data
        self._index = 0

    #getter functions for channel attributes

    @property
    def data(self):
        return self._data
    @property
    def fs(self):
        return self._fs
    @property
    def time(self):
        return self._t 
    @property
    def originalfiltersettings(self):
        return self._filterfreqs
    @property
    def label(self):
        return self._label
    @property
    def color(self):
        return self._color 
    @property
    def index(self):
        return self._index
    #setter functions for channel attributes
    @data.setter
    def data(self, data_in):
        self._data = data_in
        return self._data
    @fs.setter
    def fs(self, fs_in):
        self._fs = fs_in 
        self._t = np.arange(0, (len(self._data)/self._fs), (1/self._fs))
        return self._fs 
    @originalfiltersettings.setter
    def originalfiltersettings(self, filterfreqs_in):
        self._filterfreqs = filterfreqs_in
        return self._filterfreqs
    @label.setter
    def label (self, label_in):
        self._label = label_in
        return self._label
    @color.setter
    def color (self, color_in):
        self._color = color_in
        return self._color
    @index.setter
    def index (self, index_in):
        self._index = index_in
        colors = ["k", "b","g", "m", "r"]
        self._color = colors[index_in]
        return self._index  
    
    #tool functions for channel objects

    #filtering function (lowpass, highpass, notch, bandpass, band reject)
    #modifies self._data and self._filterfreqs
    def filt(self, cutoff, ftype, filter_order = 2):
        if (cutoff > ((self._fs)/2)):
            raise Exception(f"Filter frequency should not exceed Nyquist: {(self._fs)/2} ")
        if (ftype == 'hp'): #highpass filter
            assert (isinstance(cutoff, float) or isinstance(cutoff,int)), "Must specify integer or float."
            b_hpf, a_hpf = signal.butter(filter_order, cutoff, 'highpass', fs=self._fs)
            out = signal.filtfilt(b_hpf, a_hpf, self._data)
            self._filterfreqs[0] = cutoff
            self._data = out
        elif (ftype == 'lp'):#lowpass fitler
            assert (isinstance(cutoff, float) or isinstance(cutoff,int)), "Must specify integer or float."
            b_lpf, a_lpf = signal.butter(filter_order, cutoff, 'lowpass', fs=self._fs)
            out = signal.filtfilt(b_lpf, a_lpf, self._data)
            self._filterfreqs[1]= cutoff
            self._data = out
        elif (ftype == 'n'): #notch filter
            assert (isinstance(cutoff, float) or isinstance(cutoff,int)), "Must specify integer or float."
            Q = (math.sqrt((cutoff+1)*(cutoff-1)))/2
            b_notch, a_notch = signal.iirnotch(cutoff, Q ,self._fs)
            out = signal.filtfilt(b_notch, a_notch, self._data)
            self._data = out
        elif (ftype == 'bp'): #bandpass filter
            assert isinstance(cutoff, list), "Must specify 2-element list"
            b_bpf, a_bpf = signal.butter(filter_order, cutoff, 'bandpass', fs=self._fs)
            out = signal.filtfilt(b_bpf, a_bpf, self._data)
            self._filterfreqs = cutoff
            self._data = out
        elif (ftype == 'br'): #band reject filter
            assert isinstance(cutoff, list), "Must specify 2-element list"
            b_brf, a_brf = signal.butter(filter_order, cutoff, 'bandstop', fs = self._fs)
            out = signal.filtfilt(b_brf, a_brf, self._data)
            self._data = out
        else: 
            raise Exception("Incorrect filter type specified!")
        return self

    @property
    def std(self, interval=[0,0]):
        if (interval == [0,0]):
            return np.std(self._data)
        else: 
            spec_interval = self._data[(interval[0]*self._fs), (interval[1]*self._fs)]
            return np.std(spec_interval)
    
    
    #downsampling function, applies an anti-aliasing filter first, then downsamples
    #modifies self._fs, self._data, and if anti-aliasing filter is less than lowpass filter, self._filterfreq, and the time vector
    def decimate(self, decim_factor):
        out_data = signal.decimate(self._data, decim_factor)
        self._data = out_data
        new_fs = self._fs/decim_factor
        self._fs = new_fs
        if (self._filterfreqs[1] > (new_fs)/2):
            self._filterfreqs[1]=new_fs/2 
        self._t = np.arange(0, (len(self._data)/self._fs), (1/self._fs))
        return self

    def normalize(self, norm_type, norm_value = None):
        if (norm_type == 'mean'):
            avg = np.mean(self._data)
            out_data = self._data - avg
            self._data = out_data 
        elif (norm_type == "std"):
            std_coeff = 1/(self.get_std())
            out_data = np.multiply(self._data, std_coeff)
            self._data = out_data 
        elif (norm_type == "scalar"):
            assert (isinstance(norm_value, float) or isinstance(norm_value, int)), "Must specify number for scalar"
            #print(type(norm_value))
            out_data = np.multiply(self._data, norm_value)
            self._data = out_data
        else: 
            raise Exception("Incorrect normalization type specified")
        return self

# fs = 44100       # sampling rate, Hz, must be integer
# duration = 1   # in seconds, may be float
# f1 = 10        # sine frequency, Hz, may be float
# f2 = 4       
# # generate samples, note conversion to float32 array
# data = (np.sin(2*np.pi*np.arange(fs*duration)*(f1)/fs)) + (np.sin(2*np.pi*np.arange(fs*duration)*(f2)/fs))

# chan1 = Channel(data, fs)
# plt.plot(chan1._t, data)
# chan1.filt(7, 'hp', 2)
# plt.figure()
# plt.plot(chan1._t, chan1.get_data())

# """Session Class"""

class Session: 

    def __init__(self, sessionpath = ""):
        if (sessionpath != ""):
            self._sessionpath = sessionpath
            if sessionpath.endswith(".wav") or sessionpath.endswith(".m4a"):
                self._eventspath = sessionpath[:-4] + '-events.txt' 
                if not os.path.exists(self._eventspath):
                    self._eventspath = None
                    print("No event file found")
        else:
            self._sessionpath = None
            self._eventspath = None 
            print("Empty Session object created")
        
        self._channels = []
        self._events = Events()

        #reading data from file
        if (self._sessionpath != None):
            try:
                if sessionpath.endswith(".wav"):
                    sample_rate, data = wavfile.read(self._sessionpath)
                    self._samplerate = sample_rate
                    

                    path_to_date = self._sessionpath
                    path_to_date = path_to_date.split('_')
                    date = path_to_date[-2]
                    date = date.split("-")
                    date = [int(x) for x in date]
                    time = path_to_date[-1]
                    time = time.split(".")[:-1]
                    time = [int(y) for y in time]
                    self._datetime = datetime(year=date[0], month=date[1], day=date[2], hour=time[0], minute=time[1], second=time[2])

                    if (np.ndim(data) == 1):
                        add_channel = Channel(data = data, fs= self._samplerate)
                        add_channel.index = 0
                        self._channels.append(add_channel)
                    else: 
                        for i in range(np.ndim(data)): 
                            add_channel = Channel(data = np.transpose(data)[i], fs= self._samplerate)
                            add_channel.index = i
                            self.channels.append(add_channel) 
                elif sessionpath.endswith(".m4a"):
                    tag = TinyTag.get(sessionpath)
                    print("done this")
                    sample_rate = tag.samplerate
                    data = AudioSegment.from_file(sessionpath, "m4a")
                    print("done that")
                    self._samplerate = sample_rate
                    if (np.ndim(data) == 1):
                        add_channel = Channel(data = data, fs= self._samplerate)
                        add_channel.index = 0
                        self._channels.append(add_channel)
                    else: 
                        for i in range(np.ndim(data)): 
                            add_channel = Channel(data = np.transpose(data)[i], fs= self._samplerate)
                            add_channel.index = i
                            self.channels.append(add_channel) 
            except BaseException as err:
                print(f"Unexpected {err}, {type(err)}")
        if (self._eventspath != None):
            try: 
                with open(self._eventspath) as event_file:
                    timestamps = event_file.readlines()
                    timestamps = timestamps[2:]
                    for timestamp in timestamps:
                        eventname = timestamp[0]
                        if eventname not in self._events:
                            e = Event(eventname)
                            e.timestamps.append( float(timestamp.split(',')[1])) 
                            self._events.append( e )
                        else:
                            self._events[eventname].timestamps.append( float(timestamp.split(',')[1])) 
                                                       
            except BaseException as err:
                print(f"Unexpected {err}, {type(err)}")
            except: 
                print("This event file doesn't exist in your working directory.")
            

    #getter object for Session class
  
    @property
    def channels(self): #returns list of channel objects
      '''
      Returns a list of Channel objects corresponding to the channels in a Session object.
      '''
      return self._channels
    
    @property
    def sessionpath(self): #retruns data path of session
      '''
      Returns path of Session data (string object).
      '''
      return self._sessionpath
    
    @property
    def eventspath(self): #if it exists, events path of session is returned
      '''
      Returns path of Session events (string object).
      If there are no session objects, prints a warning message and returns nothing. 
      '''
      if (self._eventspath == None):
          print("No event path is specified for this session.")
      else: 
          return self._eventspath

    @property
    def sessionID(self): #sessionID is returned 
      '''
      Returns ID of session. 
      The session ID is a numbe/string assigned by the user when it is set (see set_sessionID).
      '''
      try:
           self._sessionID
      except AttributeError:
          self.set_sessionID()

      return self._sessionID

    @property
    def subject(self): #subject is returned
      '''
      Returns name of subject as a string object. 
      The subject is assigned by the user when it is set (see set_subject).
      '''
      try:
           self._subject
      except AttributeError:
          self.set_subject()
      return self._subject

    @property
    def datetime(self): #date and time of session is returned
      '''
      Returns datetime object witht he date and time of the session. 
      The date and time is set by the user (see set_datetime).
      '''
      try:
          self._datetime
      except AttributeError:
          self._datetime
      return self._datetime
    
    @property
    def samplerate(self): 
      '''
      Returns the sampling rate of the channel data. 
      The sampling rate is set by the user at the initialization of the Session object. 
      It can be modified via the downsampling method (see _decim)
      '''
      return self._samplerate
    
    @property
    def filtering(self):
      '''
      Returns the filter frequencies of the channel data. 
      These are specified by the user
      '''
      return self._filterfreqs 

    @property
    def events(self):
      '''
      Returns a dictionary containing the events of a Session if they exist.  
      '''
      return self._events

    
    #setter object for Session class
    def set_nchannels(self, nchannels): #returns number of channels
      '''
      Sets the number channels in a Session object.
      
      Keword Arguments: 
      nchannels -- number of channels (int)

      Returns:
      number of channels       
      '''
      self._nchannels = nchannels
      return self._nchannels     
    def set_channels(self, channels): #returns list of channel objects
      '''
      Set channels in a Session object.

      Keyword Arguments:
      channels -- a list of channel objects

      Return:
      a list of all channels attached to the Session object
      '''
      if (len(channels)!= self._nchannels):
          self._nchannels = len(channels)
      self._channels = channels
      channeldata = []
      for chan in channels:
          channeldata.append(chan.get_data())
      self._channeldata = channeldata 
      return self._channels
    def set_sessionpath(self, sessionpath, construct = True):
      '''
      Set the path to Session data file (string object).

      Keyword Argument:
      sessionpath -- the path to the data file (string)
      construct -- determines if a Session object should be constructed (bool)

      Return:
      the sessionpath set for the Session object
      '''
      self._sessionpath = sessionpath 
      if construct: 
           self.__init__(self._sessionpath)
      return self._sessionpath
    def set_eventspath(self, eventspath, construct = False):
      '''
      Set the path to Session events file (string object).

      Keyword Arguments:
      eventspath -- the path to the events file (string)
      construct -- determines if a Session object should be constructed (boolean)

      Return:
      the set eventspath for the Session object
      '''
      self._eventspath = eventspath
      if construct:
          self.__init__(sessionpath=self._sessionpath, eventspath = self._eventspath)
      return self._sessionpath
    @sessionID.setter
    def sessionID(self, sessionID = None):
      '''
      Set the Session ID for the Session object.

      Keyword Arguments:
      sessionID -- the ID for the session (int)

      Return:
      the set Session ID
      '''
      self._sessionID = sessionID
      return self._sessionID
    @subject.setter
    def subject(self, subject= None):
      '''
      Set the subject number for the Session object.

      Keyword Arguments:
      subject -- subject number (int)

      Return:
      the subject number
      '''
      self._subject = subject
      return self._subject
    @datetime.setter
    def datetime(self, spec_datetime = None):
      '''
      Set the year, month, day, hour, minute and second for the Session object.

      Keyword Arguments:
      spec_datetime -- set the datetime manually (string)

      Return:
      the datetime set for the Session
      '''
      self._datetime = spec_datetime
      return self._datetime
    
    def set_samplerate(self, samplerate):
      '''
      Set the sample rate for the Session object.

      Keyword Arguments:
      samplerate -- the sample rate (int)

      Return:
      the sample rate set for the Session object
      '''
      self._samplerate = samplerate
      return self._samplerate
    def set_filterfreqs(self, filterfreqs):
      '''
      Set the low and high filter frequencies for the Session object.

      Keyword Arguments:
      filterfreqs -- the low and high filter frequenciea (list: int)

      Return:
      the filter frequenciese set for the Session object
      '''
      self._filterfreqs = filterfreqs
      return self._filterfreqs 
    def set_events(self, events):
      '''
      Set the events for the Session object.

      Keyword Arguments:
      events -- the events to attatch to the Session object (dict: obj)

      Return:
      the events set for the Session object
      '''
      self._events = events
      return self._events
    
    def rename_event(self, old_event, new_event):
        self._events[new_event]=self._events.pop(old_event)
        
    def _filt (self, cutoff, ftype, filter_order = 2, channel_index = None):
      '''
      Filter the channel data in the Session object inplace.

      Keyword Arguments:
      cutoff -- cutoff frequency (int)
      ftype -- low or high-pass filter (string)
      filter_order -- filter order (int)
      channel_index -- if set, filters only the chosen channel_index (int)
      '''
      if (channel_index == None): 
          for chan in self._channels:
              filted_chan = chan.filt(cutoff, ftype, filter_order)
              self._channels[channel_index] = filted_chan
      else: 
          chan_to_filt = self._channels[channel_index]
          filtered_chan = chan_to_filt.filt(cutoff, ftype, filter_order)
          self._channels[channel_index] = filtered_chan 
    def _get_std(self, interval=[0,0], channel_index=None):
      '''
      Get the standard deviation of the data in the Session object.

      Keyword Arguments:
      interval -- if set, the std will only be calculated on the interval of the data (list: int)
      channel_index -- if set, calculates std only on chosen channel index (int)

      Return:
      the standard deviation calculated
      '''
      if (channel_index == None):
          std_vec = []
          for chan in self._channels:
              std_vec.append(chan.get_std(interval))
          return std_vec
      else: 
          return self._channels[channel_index].get_std(interval)
    def _decim(self, decim_factor, channel_index=None):
      '''
      Downsample the data in the Session object inplace.

      Keyword Arguments:
      decim_factor -- the factor to decimate by (int)
      channel_index -- if set, decimates only the chosen index (int)
      '''
      if (channel_index == None): 
          for chan in self._channels:
              decim_chan = chan.decim(decim_factor)
              self._channels[channel_index] = decim_chan
      else: 
          chan_to_decim = self._channels[channel_index]
          decimated_chan = chan_to_decim.decim(decim_factor)
          self._channels[channel_index] = decimated_chan
    def _normalize(self, norm_type, norm_value = None, channel_index = None):
      '''
      Normalize the data in the Session object inplace.

      Keyword Arguments:
      norm_type -- setting to normalize the data based on the mean, std or a scalar (str)
      norm_value -- scalar to use to normalize the data; use with norm_type of scalar (float or int)
      channel_index -- if set, only normalize the chosen channel (int)
      '''
      if (channel_index == None):
          for chan in self._channels:
              norm_chan = chan.normalize(norm_type, norm_value)
              self._channels[channel_index] = norm_chan
      else: 
          chan_to_norm = self._channels[channel_index]
          normalized_chan = chan_to_norm.normalize(norm_type, norm_value)
          self._channels[channel_index] = normalized_chan 
              

    # plotting functions      

    # plot helper functions
    def plot_events(self, left_bound, right_bound, chosen_channel_fs, max_data, min_data, offset):
        color_index = 0
        #n_colors = len(self._events)
        event_plots = []
        event_labels = []
        event_colors = []
        for event in self.events:
            color = event.color
            time_markers = event.timestamps
            time_markers_interval = []
            event_label = f"Event {event}"
            event_labels.append(event_label)
            for marker in time_markers: 
                marker_samp = marker*self.channels[0].fs
                if (left_bound <= marker_samp) and (right_bound >= marker_samp):
                    time_markers_interval.append(marker)
            markerlength = 10*(max_data - min_data)
            y = [0.5*(color_index/(len(self._events)))+0.25]*len(time_markers_interval)
            event_plot = plt.scatter(time_markers_interval, y, c = color, marker = "|")
            plt.ylim((0,1))
            #event_plot = plt.eventplot(time_markers_interval, lineoffsets=offset, linelengths= markerlength, linewidths = 1, colors = color, label ='Event')
            time_axis_lim= self.channels[0].time[-1]
            plt.xlim(0,time_axis_lim)
            event_plots.append(event_plot)
            color_index = color_index + 1
        return event_labels, event_plots, event_colors

    def monte_carlo_avg(self, spec_channel, onset_event, pre_onset, post_onset):
        channel = self._channels[spec_channel]
        channel_data = channel.get_data() 
        onsets = self._events[onset_event]
        num_mc_epochs = len(onsets)
        num_mc_sim = 100

        window_size_samples = int((pre_onset + post_onset)*self._samplerate)
        mc_avgs = np.zeros((num_mc_sim,window_size_samples))
        
        for sim in range(num_mc_sim):
            these_mc_onsets = []
            for i in range(num_mc_epochs):
                cur_size_mc_onsets = len(these_mc_onsets)
                while cur_size_mc_onsets == len(these_mc_onsets):
                    this_random = (len(channel_data)*random.random())/self._samplerate
                    if this_random not in these_mc_onsets:
                        if (this_random>pre_onset) and (this_random <((len(channel_data)/self._samplerate)-post_onset)):
                            these_mc_onsets.append(this_random)
        
            these_epochs_data_mc = []
            for p in range(len(these_mc_onsets)):
                onset_timestamp = these_mc_onsets[p]
                sel_start_samp = int((onset_timestamp - pre_onset)*self._samplerate)
                this_epoch_data_mc = channel_data[sel_start_samp:(sel_start_samp + window_size_samples)]
                these_epochs_data_mc.append(this_epoch_data_mc)
        
            avg_raw_epoch_mc = np.mean(these_epochs_data_mc, axis=0)
            mc_avgs[sim,:] = avg_raw_epoch_mc
        avg_raw_epoch_mc = np.mean(mc_avgs,0)
        mc_std = np.std(mc_avgs,0)
        mc_avg_epoch = avg_raw_epoch_mc
        mc_plus_epoch = avg_raw_epoch_mc + (2*mc_std)
        mc_minus_epoch = avg_raw_epoch_mc - (2*mc_std)
        
        return mc_avg_epoch, mc_plus_epoch, mc_minus_epoch





    def plot_interval(self, channelindex, bounds, offset=0, events = False, event_marker_factor=2, show = True, make_fig = True, legends=False):
      '''
      Plot an interval of the data.

      Keyword Arguments:
      channelindex -- channel to plot (int)
      left_bound -- start plot interval from left_bound (float)
      right_bound -- end plot interval at right_bound (float)
      offset -- offset between channels in case data is multichannel, offset from horizontal axis in case data is single channel (float) Default value is 0.
      events -- if True, shows events on the plot (Boolean) Default value is False.
      event_marker_factor -- controls the length of the event markers (float)
      show -- if True, shows plot (Boolean). Default value is True.
      make_fig -- if True, makes a new Figure object (Boolean). Default value is True.
      legends -- if True, shows legend of plots on figure (Boolean). Default value is False

      '''
      left_bound = bounds[0]
      right_bound = bounds[1]
      if make_fig:
            plt.figure()
      if (isinstance(channelindex, list)):
          sp_ind = 1
          for chan_ind in channelindex:
                plt.subplot(len(channelindex), 1, sp_ind)
                sp_ind = sp_ind + 1
                chosen_channel = self._channels[chan_ind]
                chosen_channel_fs = chosen_channel.get_fs()
                left_bound_sample = left_bound*chosen_channel_fs
                right_bound_sample = right_bound*chosen_channel_fs
                full_time_axis = chosen_channel.get_time()
                time_axis = full_time_axis[left_bound_sample: right_bound_sample]
                full_data_axis = chosen_channel.get_data()
                data_axis = full_data_axis[left_bound_sample: right_bound_sample]
                data_axis = list(np.asarray(data_axis) + offset)
                min_data = np.min(data_axis)
                max_data = np.max(data_axis)
                plt.ylim(min_data*1.1, max_data*1.1)
                plt.plot(time_axis, data_axis, color= self._channels[chan_ind].get_color() )
                event_labels = [f'Channel {chan_ind}']
                if events:
                    labels, plots, colors = self.plot_events(left_bound_sample, right_bound_sample, chosen_channel_fs, max_data, min_data, offset)
                    event_labels = event_labels + labels
                plt.xlabel("Time(sec)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                if legends:
                    plt.legend(event_labels)
          if show:
                plt.show()


      else:       
            chosen_channel = self._channels[channelindex]
            chosen_channel_fs = chosen_channel.get_fs()
            left_bound = left_bound*chosen_channel_fs
            right_bound = right_bound*chosen_channel_fs
            full_time_axis = chosen_channel.get_time()
            time_axis = full_time_axis[left_bound: right_bound]
            full_data_axis = chosen_channel.get_data()
            data_axis = full_data_axis[left_bound: right_bound]
            data_axis = list(np.asarray(data_axis) + offset)
            min_data = np.min(data_axis)
            max_data = np.max(data_axis)
            plt.ylim(min_data*1.1, max_data*1.1)
            plt.plot(time_axis, data_axis, color= self._channels[channelindex].get_color() )
            event_labels = ['data']
            if events:
               labels, plots, colors = self.plot_events(left_bound, right_bound, chosen_channel_fs, max_data, min_data, offset)
               event_labels = event_labels + labels
            plt.xlabel("Time(sec)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            if legends:
                plt.legend(event_labels)
            if show:
                plt.show()
    
      
    
        
    def plot_overview(self, show_events=True, figsize = (11,8.5)):
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        if show_events:
            plot_size = len(self.channels) + 3
        else:
            plot_size = len(self.channels) + 1
        plt.subplot(plot_size, 1, 1)
        session_overview = f"""
        File Name: {self.sessionpath}
        Date and Time : {self.datetime}
        Sample Rate: {self.samplerate}
        Session duration (in samples): {len(self.channels[0].time)} samples
        Session duration (in hh:mm:ss): {time.strftime('%H:%M:%S', time.gmtime(len(self.channels[0].time)/self.samplerate))} 
        Session ID: {self.sessionID}
        Subject: {self.subject}
        """
        plt.text(0,0,session_overview, fontsize=12)
        plt.title(f"Session Overview: {self.sessionID}", fontweight='bold', loc = 'center', fontsize=18)
        plt.tight_layout()
        plt.axis("off")
        plot_ind = 2
        chan_ind = 0
        for chan in self.channels:
            plt.subplot(plot_size, 1, plot_ind)
            plt.plot(chan.time, chan.data, color = chan.color)
            plt.xlim(0, chan.time[-1])
            #plt.axis("off")
            #plt.subplot(plot_size, 1, plot_ind+1)
            channel_overview = f"Channel {chan_ind}:  Mean: {round(np.mean(chan.data), 2)} | Standard Dev: {round(chan.std,2)}"
            #plt.annotate(0,0,channel_overview, fontsize=8, wrap=True)
            plt.title(channel_overview, loc='left', fontsize=14)
            plt.axis("off")
            plot_ind = plot_ind + 1
            chan_ind = chan_ind + 1
            if not show_events:
                ax = plt.gca()
                bar = AnchoredSizeBar(ax.transData, 1, '1 s', 4)
                ax.add_artist(bar)
                ax.axis('off')
        if show_events:
            plt.subplot(plot_size, 1, plot_ind)
            e_labels, e_plots, e_colors = self.plot_events(0,len(self.channels[0].data),self.samplerate,np.max(self.channels[0].data),np.min(self.channels[0].data),0)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            #plt.legend(e_labels,bbox_to_anchor=(0, 0))
            #plt.xlabel("Time (seconds)")
            plt.title("Events", loc = 'left', fontsize=14)
            plt.tight_layout()
            plt.yticks([])
            ax = plt.gca()
            bar = AnchoredSizeBar(ax.transData, 1, '1 s', 4)
            ax.add_artist(bar)
            ax.axis('off')
            #plt.annotate()
            plt.subplot(plot_size,1,plot_ind+1)
            col_index= 0
            for ev in self.events:
                inter_event_interval = np.mean([(ev.timestamps[i+1]-ev.timestamps[i]) for i in range(len(ev.timestamps)-1)])
                plt.text(0,0.75-((0.75/len(self.events))*col_index),f"       Event [{ev}] (n = {len(ev.timestamps)}): meInter-Event Interval = {round(inter_event_interval,2)}s \n", color = ev.color, fontsize=14)
                col_index=col_index+1 
            plt.axis("off")

        
        plt.tight_layout()
        plt.show()

        pass

    def plot_eltraces(self, spec_event, bounds, spec_channel = 0, spec_color = 'k', alpha = 0.2, show=True, makefig = True, monte_carlo=False):
        lbound = bounds[0]
        rbound = bounds[1]
        if makefig:
            plt.figure()
        timemarkers = self._events[spec_event]
        spec_channel_data = self._channels[spec_channel].get_data()
        time_axis = np.arange(0, -lbound + rbound, (1/self._samplerate))
        for timemarker in timemarkers:
            data_axis = spec_channel_data[math.floor((timemarker + lbound)*self._samplerate): math.floor((timemarker + rbound)*self._samplerate)]
            if len(data_axis) != len(time_axis):
                pass
            else:
                plt.plot(time_axis, data_axis, color = spec_color, alpha = alpha)
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude")
        if monte_carlo:
            mc_avg, mc_plus, mc_minus = self.monte_carlo_avg(spec_channel=spec_channel, onset_event=spec_event,pre_onset=-lbound,post_onset=rbound)
            plt.plot(time_axis, mc_avg, color = "blue", linewidth = 2)
            plt.plot(time_axis, mc_plus, color = "blue")
            plt.plot(time_axis, mc_minus, color = "blue")
            plt.fill_between(time_axis, mc_minus,mc_plus,color="blue", alpha=0.2)
        
        if show:
            plt.show()
    
    #Original ETA
    def plot_elavg(self, spec_event, timewindow=[-1, 1], spec_channel = 0, spec_color = 'k', showtraces = False, alpha = 0.2, show=True, makefig=True, monte_carlo=False):
        lbound = timewindow[0]
        rbound = timewindow[1]
        if makefig:
            plt.figure()
        timemarkers = self._events[spec_event]
        spec_channel_data = self.get_channel(spec_channel).get_data()
        #spec_channel_data = list(spec_channel_data)
        time_axis = np.arange(0, -lbound + rbound, (1/self._samplerate))
        #time_axis = list(time_axis)
        avg_data = []
        for timemarker in timemarkers:
            data_axis = spec_channel_data[int((timemarker + lbound)*self._samplerate): int((timemarker + rbound)*self._samplerate)]
            if len(data_axis) != len(time_axis):
                pass
            else:
                avg_data.append(data_axis)
                if showtraces:
                    plt.plot(time_axis, data_axis, color = spec_color, alpha = alpha)
            
        avg_trace = np.mean(avg_data, axis=0)
        #print(f"Len Avg Trace: {len(avg_trace)}")
        #print(f"Len time: {len(time_axis)}")
        if len(avg_trace) > len(time_axis):
            avg_trace=avg_trace[0:len(time_axis)-1]
            print("this")
        if len(avg_trace) < len(time_axis):
            time_axis = time_axis[0:len(avg_trace)-1]
            print("that")
        plt.plot(time_axis, avg_trace, color = self.get_channel(spec_channel).get_color())
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude")
        if monte_carlo:
            mc_avg, mc_plus, mc_minus = self.monte_carlo_avg(spec_channel=spec_channel, onset_event=spec_event,pre_onset=-lbound,post_onset=rbound)
            plt.plot(time_axis, mc_avg, color = "blue", linewidth = 2)
            plt.plot(time_axis, mc_plus, color = "blue")
            plt.plot(time_axis, mc_minus, color = "blue")
            plt.fill_between(time_axis, mc_minus,mc_plus,color="blue", alpha=0.2)
        if show:
            plt.show()
        return
    
    #Event Triggered Average
    def plot_eta(self, events, timewindow=[-1, 1], channel = 0,  showtraces = False, alpha = 0.2, show=True, makefig=True, monte_carlo=False, ax=None):
        lbound = timewindow[0]
        rbound = timewindow[1]
        if np.isscalar(events):
            events = [events]
        if makefig:
            plt.figure()
        n = 0
        for event in events:  
            if type(event) == str:
                event = self._events[event]  
            timemarkers = event.timestamps
            #spec_channel_data = list(spec_channel_data)
            time_axis = np.arange(lbound, rbound, (1/self._samplerate))
            #time_axis = list(time_axis)
            avg_data = []
            for timemarker in timemarkers:
                data_axis = self.channels[channel].data[int((timemarker + lbound)*self._samplerate): int((timemarker + rbound)*self._samplerate)]
                if len(data_axis) != len(time_axis):
                    #print("Uneven Row")
                    pass
                else:
                    avg_data.append(data_axis)
                    if showtraces:
                        plt.plot(time_axis, data_axis, color = self._events.color(event), alpha = alpha)
                
            avg_trace = np.mean(avg_data, axis=0)
            #print(f"Len Avg Trace: {len(avg_trace)}")
            #print(f"Len time: {len(time_axis)}")
            if len(avg_trace) > len(time_axis):
                avg_trace=avg_trace[0:len(time_axis)-1]
                print("this")
            if len(avg_trace) < len(time_axis):
                time_axis = time_axis[0:len(avg_trace)-1]
                print("that")
            if ax is None:
                fig, ax = plt.subplots(1)
            else:
                plt.sca(ax)
            plt.plot(time_axis, avg_trace, color = event.color)
            n = n + 1
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude")
        if monte_carlo:
            mc_avg, mc_plus, mc_minus = self.monte_carlo_avg(spec_channel=channel, onset_event=event,pre_onset=-lbound,post_onset=rbound)
            plt.plot(time_axis, mc_avg, color = "blue", linewidth = 2)
            plt.plot(time_axis, mc_plus, color = "blue")
            plt.plot(time_axis, mc_minus, color = "blue")
            plt.fill_between(time_axis, mc_minus,mc_plus,color="blue", alpha=0.2)
        if show:
            plt.show()
        return
    
    def plot_joydiv(self, spec_event, bounds, spec_channel = 0, spec_color = 'k', alpha = 0.2, show=True, makefig=True):
        lbound = bounds[0]
        rbound = bounds[1]
        if makefig:
            fig = plt.figure()
        timemarkers = self._events[spec_event]
        spec_channel_data = self.get_channel(spec_channel).get_data()
        time_axis = np.arange(0, -lbound + rbound, (1/self._samplerate))
        plot_index = 1
        ylim_top =  np.max(spec_channel_data)
        ylim_bottom = np.min(spec_channel_data)
        for timemarker in timemarkers:
            data_axis = spec_channel_data[int((timemarker + lbound)*self._samplerate): int((timemarker + rbound)*self._samplerate)]
            plt.subplot(len(timemarkers),1,plot_index)
            plt.plot(time_axis, data_axis, color = spec_color, alpha = alpha)
            plt.ylabel("Amplitude")
            #plt.autoscale(False)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylim(ylim_bottom, ylim_top)
            ax.axes.yaxis.set_visible(False)
            if plot_index == len(timemarkers):
                ax.axes.xaxis.set_visible(True)
                ax.spines['bottom'].set_visible(True)
            plot_index = plot_index + 1
        fig.supxlabel('Time')
        fig.supylabel('Traces')
        fig.tight_layout()
        if show:
            plt.show()
        return


    def plot_raster(self, spec_channel, bounds = (0, None)):
        lbound = bounds[0]
        rbound = bounds[1]
        chosen_channel = self._channels[spec_channel]
        chosen_channel_fs = chosen_channel.get_fs()
        lbound = lbound*chosen_channel_fs
        if rbound == None:
            rbound = len(self.get_channel(spec_channel).get_data()) - 1
        rbound = rbound*chosen_channel_fs
        full_time_axis = chosen_channel.get_time()
        time_axis = full_time_axis[lbound: rbound]
        full_data_axis = chosen_channel.get_data()
        data_axis = full_data_axis[lbound: rbound]
        #data_axis = list(np.asarray(data_axis))
        color_index = 0
        #n_colors = len(self._events)
        event_plots = []
        event_labels = []
        for event in self._events:
            color = 'C' + str(color_index)
            time_markers = self._events[event]
            time_markers_interval = []
            event_label = f"Event {event}"
            event_labels.append(event_label)
            for marker in time_markers: 
                marker_samp = marker*chosen_channel_fs
                if (lbound <= marker_samp) and (rbound >= marker_samp):
                    time_markers_interval.append(marker)
            event_plot = plt.eventplot(time_markers_interval, linewidths = 1, colors = color, label ='Event')
            event_plots.append(event_plot)
            color_index = color_index + 1
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude")
        plt.legend(event_labels)
        plt.show()

    def plot_mag_spectrum(self, spec_channel, bounds=(0,None)):
        '''
        Plot the Magnitude Spectrum of the data.

        Keyword Arguments:
        spec_channel -- specific channel to calculate psd for (int)
        lbound -- calculate the Magnitude Spectrum for data starting from lbound (float)
        rbound -- calculate the Magnitude Spectrum for data upto rbound (float) 
        '''
        lbound = bounds[0]
        rbound = bounds[1]
        plt.figure()
        chosen_channel = self._channels[spec_channel]
        chosen_channel_fs = chosen_channel.get_fs()
        lbound = lbound*chosen_channel_fs
        if rbound == None:
            rbound = len(self.get_channel(spec_channel).get_data()) - 1
        rbound = rbound*chosen_channel_fs
        full_time_axis = chosen_channel.get_time()
        time_axis = full_time_axis[lbound: rbound]
        full_data_axis = chosen_channel.get_data()
        data_axis = full_data_axis[lbound: rbound]
        plt.magnitude_spectrum(data_axis, Fs = chosen_channel_fs)
        plt.title("Magnitude Spectrum of the Signal")
        plt.xlabel("Time(sec)")
        plt.ylabel("Amplitude of Spectrum")
        plt.show()

    def plot_spectrogram(self, spec_channel, freq_res = 1, time_res=0.5, bounds = (0, None), freq_bounds=None, amp_bounds = None, makefig=True, show=True):
        '''
        Plot the Spectrogram of the data.

        Keyword Arguments:
        spec_channel -- specific channel to calculate psd for (int)
        lbound -- calculate spectrogram for data starting from lbound (float)
        rbound -- calculate spectrogram for data upto rbound (float) 
        '''
        lbound = bounds[0]
        rbound = bounds[1]
        if makefig:
            plt.figure()
        chosen_channel = self._channels[spec_channel]
        chosen_channel_fs = chosen_channel.get_fs()
        lbound = lbound*chosen_channel_fs
        if rbound == None:
            rbound = len(self.get_channel(spec_channel).get_data()) - 1
        rbound = rbound*chosen_channel_fs
        full_time_axis = chosen_channel.get_time()
        time_axis = full_time_axis[lbound: rbound]
        full_data_axis = chosen_channel.get_data()
        data_axis = full_data_axis[lbound: rbound]
        nfft = math.floor(chosen_channel_fs/freq_res) 
        n_overlap = nfft - time_res*chosen_channel_fs 
        if amp_bounds != None:
            vmin = amp_bounds[0]
            vmax = amp_bounds[1]
        else:
            vmin = None
            vmax = None
        plt.specgram(data_axis, Fs = chosen_channel_fs, NFFT=nfft, noverlap=n_overlap, vmin=vmin, vmax=vmax)
        if freq_bounds!=None:
            plt.ylim(freq_bounds)
        plt.title("Spectrogram of the Signal")
        plt.xlabel("Time(sec)")
        plt.ylabel("Frequency")
        if show:
            plt.show()

    def plot_psd(self, spec_channel, bounds = (0, None), freq_bounds=None, amp_bounds=None, freq_res =1, time_res=0.5, show=True, makefig=True):
        '''
        Plot the Power Spectral Density of the data.

        Keyword Argumetns:
        spec_channel -- specific channel to calculate psd for (int)
        lbound -- calculate psd for data starting from lbound (float)
        rbound -- calculate psd for data upto rbound (float) 
        '''
        lbound = bounds[0]
        rbound = bounds[1]
        if makefig:
            plt.figure()
        chosen_channel = self._channels[spec_channel]
        chosen_channel_fs = chosen_channel.get_fs()
        lbound = lbound*chosen_channel_fs
        if rbound == None:
            rbound = len(self.get_channel(spec_channel).get_data()) - 1
        rbound = rbound*chosen_channel_fs
        full_time_axis = chosen_channel.get_time()
        time_axis = full_time_axis[lbound: rbound]
        full_data_axis = chosen_channel.get_data()
        data_axis = full_data_axis[lbound: rbound]
        nfft = math.floor(chosen_channel_fs/freq_res) 
        n_overlap = nfft - time_res*chosen_channel_fs 
        plt.psd(data_axis, Fs = chosen_channel_fs, NFFT = nfft, noverlap=n_overlap)
        plt.title("Power Spectral Density Plot of the Signal")
        plt.xlabel("Frequency(Hz)")
        plt.ylabel("Amplitude (dB/Hz)")
        if amp_bounds!=None:
            plt.ylim(amp_bounds)
        if freq_bounds!=None:
            plt.xlim(freq_bounds)
        if show:
            plt.show()
    
    def plot_peth(self, spec_channel, bounds, onset_event, spike_event, nbins, figsize = None):
        if isinstance(onset_event, list):
            fig = plt.figure(figsize=figsize)
            n_onset_events = len(onset_event)
            ons_index = 2
            all_spikes_onsets = []
            for onset in onset_event:
                num_trials = len(self._events[onset])
                trials = self._events[onset]
                spikes = self._events[spike_event]
                left_bound = bounds[0]
                right_bound = bounds[1]
                trial_index = 1
                all_spikes = []
                plt.subplot(1+n_onset_events,1, ons_index)
                for trial in trials:
                #print("TRIAL")
                    spikes_per_event =  []
                    for spike in spikes:
                        if (spike >= trial + left_bound) and (spike <= trial + right_bound):
                        #print(spike)
                            spikes_per_event.append(spike)
                    trial_axis = [trial_index]*len(spikes_per_event)

                    spikes_per_event_adj = [spp - trial for spp in spikes_per_event]
                    all_spikes = all_spikes + spikes_per_event_adj
                    plt.scatter(spikes_per_event_adj, trial_axis, color = f"C{ons_index-2}", marker = "|")
                    trial_index = trial_index + 1
                plt.xlim([bounds[0], bounds[1]]) 
                plt.axvline(0, color = "r")
                plt.xlabel("Interval (seconds)")   
                plt.ylabel("Trials")
                plt.title(f"Raster for {onset}")
                ons_index = ons_index + 1
                all_spikes_onsets.append(all_spikes)

            plt.subplot(1+n_onset_events,1,1)
            plt.title("Peri-Event Time Histogram")
            for ons_iter in range(len(onset_event)):
                plt.hist(all_spikes_onsets[ons_iter], bins = nbins, range = bounds, histtype=u'step')
            plt.ylabel("Count")
            plt.legend(onset_event)
            plt.axvline(0, color = "r")
            
            fig.tight_layout()
            plt.show()
            pass
        else:
            fig = plt.figure(figsize=figsize)
            num_trials = len(self._events[onset_event])
            trials = self._events[onset_event]
            spikes = self._events[spike_event]
            left_bound = bounds[0]
            right_bound = bounds[1]
            trial_index = 1
            all_spikes = []
            plt.subplot(2,1,2)
            for trial in trials:
                #print("TRIAL")
                spikes_per_event =  []
                for spike in spikes:
                    if (spike >= trial + left_bound) and (spike <= trial + right_bound):
                        #print(spike)
                        spikes_per_event.append(spike)
                trial_axis = [trial_index]*len(spikes_per_event)

                spikes_per_event_adj = [spp - trial for spp in spikes_per_event]
                all_spikes = all_spikes + spikes_per_event_adj
                plt.scatter(spikes_per_event_adj, trial_axis, color = "k", marker = "|")
                trial_index = trial_index + 1
            plt.xlim([bounds[0], bounds[1]]) 
            plt.axvline(0, color = "r")
            plt.xlabel("Interval (seconds)")   
            plt.ylabel("Trials")
            plt.title("Raster")

            plt.subplot(2,1,1)
            plt.title("Peri-Event Time Histogram")
            plt.hist(all_spikes, bins = nbins, range = bounds, histtype=u"step")
            plt.ylabel("Count")
            plt.axvline(0, color = "r")

            fig.tight_layout()
            plt.show()
                    
class Sessions:
    def __init__(self, sessions):
        self._sessions = sessions
    
    def plot_interval(self, channel, bounds, offset=0, events = False, event_marker_factor=2, show = True, make_fig = True, legends=False, join = True):
        if make_fig:
            fig = plt.figure()
        if join:
            for sesh in self._sessions:
                sesh.plot_interval(channel, bounds, offset=offset, events = events, event_marker_factor= event_marker_factor, show = False, make_fig = False, legends=False)
            if legends:
                plt.legend([sesh.get_sessionID() for sesh in self._sessions])
            if show:
                plt.show()
        else:
            n_sessions = len(self._sessions)
            sesh_ind = 1
            for sesh in self._sessions:
                plt.subplot(n_sessions,1, sesh_ind)
                sesh.plot_interval(channel, bounds, offset=offset, events = events, event_marker_factor= event_marker_factor, show = False, make_fig = False, legends=legends)
                sesh_ind = sesh_ind + 1
                plt.title(sesh.get_sessionID())
            if show:
                plt.tight_layout()
                plt.show()
    
    def plot_psd(self, spec_channel, bounds = (0, None), freq_bounds=None, amp_bounds=None, freq_res =1, time_res=0.5, show=True, makefig=True, join = True):
        if makefig:
            fig = plt.figure()
        if join:
            for sesh in self._sessions:
                sesh.plot_psd(spec_channel, bounds = bounds, freq_bounds=freq_bounds, amp_bounds=amp_bounds, freq_res =freq_res, time_res=time_res, show=False, makefig=False)
            plt.legend([sesh.get_sessionID() for sesh in self._sessions])
            if show:
                plt.show()
        else:
            n_sessions = len(self._sessions)
            sesh_ind = 1
            for sesh in self._sessions:
                plt.subplot(n_sessions,1, sesh_ind)
                sesh.plot_psd(spec_channel, bounds = bounds, freq_bounds=freq_bounds, amp_bounds=amp_bounds, freq_res =freq_res, time_res=time_res, show=False, makefig=False)
                sesh_ind = sesh_ind + 1
                plt.title(sesh.get_sessionID())
            if show:
                plt.tight_layout()
                plt.show()
    
    def plot_peth(self, spec_channel, bounds, onset_event, spike_event, nbins, figsize = None):
        fig = plt.figure(figsize=figsize)
        nsessions = len(self._sessions)
        sesh_index = 2
        all_spikes_sessions = []
        for sesh in self._sessions:
            num_trials = len(sesh._events[onset_event])
            trials = sesh._events[onset_event]
            spikes = sesh._events[spike_event]
            left_bound = bounds[0]
            right_bound = bounds[1]
            trial_index = 1
            all_spikes = []
            plt.subplot(1+nsessions,1, sesh_index)
            for trial in trials:
            #print("TRIAL")
                spikes_per_event =  []
                for spike in spikes:
                    if (spike >= trial + left_bound) and (spike <= trial + right_bound):
                    #print(spike)
                        spikes_per_event.append(spike)
                trial_axis = [trial_index]*len(spikes_per_event)        
                spikes_per_event_adj = [spp - trial for spp in spikes_per_event]
                all_spikes = all_spikes + spikes_per_event_adj
                plt.scatter(spikes_per_event_adj, trial_axis, color = "k", marker = "|")
                trial_index = trial_index + 1
            plt.xlim([bounds[0], bounds[1]]) 
            plt.axvline(0, color = "r")
            plt.xlabel("Interval (seconds)")   
            plt.ylabel("Trials")
            plt.title(f"Raster {sesh.get_sessionID()}")
            sesh_index = sesh_index + 1
            all_spikes_sessions.append(all_spikes)

        plt.subplot(1+nsessions,1,1)
        plt.title("Peri-Event Time Histogram")
        for sesh_iter in range(len(self._sessions)):
            plt.hist(all_spikes_sessions[sesh_iter], bins = nbins, range = bounds, histtype=u"step")
        plt.ylabel("Count")
        plt.legend([sesh.get_sessionID() for sesh in self._sessions])
        plt.axvline(0, color = "r")
        
        fig.tight_layout()
        plt.show()

    def plot_spectrogram(self, spec_channel, freq_res = 1, time_res=0.5, bounds = (0, None), freq_bounds=None, amp_bounds = None, makefig=True, show=True):
        n_sessions = len(self._sessions)
        sesh_ind = 1
        if makefig:
            plt.figure()
        for sesh in self._sessions:
            plt.subplot(n_sessions,1, sesh_ind)
            sesh.plot_spectrogram(spec_channel, freq_res = freq_res, time_res= time_res, bounds = bounds, freq_bounds= freq_bounds, amp_bounds = amp_bounds, makefig=False, show=False)
            sesh_ind = sesh_ind + 1
            plt.title(sesh.get_sessionID())
        plt.tight_layout()
        if show:
            plt.show()
    
    def plot_eta(self, events, timewindow, channel = 0, spec_color = 'k', showtraces = False, alpha = 0.2, show=True, makefig=True, monte_carlo=False):
        n_sessions = len(self._sessions)
        sesh_ind = 1
        if makefig:
            plt.figure()
        for sesh in self._sessions:
            plt.subplot(n_sessions,1, sesh_ind)
            sesh.plot_eta(events = events, timewindow = timewindow, channel = channel, showtraces = showtraces, alpha = alpha, show=False, makefig=False, monte_carlo=monte_carlo)
            sesh_ind = sesh_ind + 1
            plt.title(sesh.sessionID)
        plt.tight_layout()
        if show:
            plt.show()

    def plot_eltraces(self, events, bounds, spec_channel = 0, spec_color = 'k', alpha = 0.2, show=True, makefig = True, monte_carlo=False):
        n_sessions = len(self._sessions)
        sesh_ind = 1
        if makefig:
            plt.figure()
        for sesh in self._sessions:
            plt.subplot(n_sessions,1, sesh_ind)
            sesh.plot_eltraces(spec_event, bounds, spec_channel = spec_channel, spec_color = spec_color, alpha = alpha, show=False, makefig=False, monte_carlo=monte_carlo)
            sesh_ind = sesh_ind + 1
            plt.title(sesh.get_sessionID())
        plt.tight_layout()
        if show:
            plt.show()
        
    def plot_joydiv(self, spec_event, bounds, spec_channel = 0, spec_color = 'k', alpha = 0.2, show=True, makefig=True):
        n_sessions = len(self._sessions)
        sesh_ind = 1
        if makefig:
            plt.figure()
        for sesh in self._sessions:
            plt.subplot(n_sessions,1, sesh_ind)
            sesh.plot_joydiv(spec_event, bounds , spec_channel = spec_channel, spec_color = spec_color, alpha = alpha, show=False, makefig=False)
            sesh_ind = sesh_ind + 1
            plt.title(sesh.get_sessionID())
        plt.tight_layout()
        if show:
            plt.show()
        pass