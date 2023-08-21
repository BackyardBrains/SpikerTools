from datetime import datetime
from scipy.io import wavfile
from collections import UserList
import numpy as np
import os

from spikertools import Event, Events
from spikertools import Channel


class Session: 
    def __init__(self, sessionpath = None):
        from spikertools import Neuron, Neurons
        from spikertools import SessionPlots

        class Paths:
            def __init__(self, sessionFile):
                self.sessionFile = sessionFile
                self.eventFile = None

        self._paths = Paths(sessionFile=sessionpath)

        self._datetime = None
        self._sessionID = None
        self._subject = None
        self._channels = []
        self._events = Events()
        self._neurons = Neurons()
        self._plot = SessionPlots( self )

        if (sessionpath != None):

            if sessionpath.endswith(".wav") or sessionpath.endswith(".m4a"):
                self.paths.eventFile  = sessionpath[:-4] + '-events.txt' 
                if not os.path.exists(self.paths.eventFile):
                    self.paths.eventFile = None
                    print("No event file found")
        #else:
            #Empty Session object created
        
        #reading data from file
        if (self.paths.sessionFile != None):
            try:
                if self.paths.sessionFile.endswith(".wav"):
                    sampleRate, data = wavfile.read(self.paths.sessionFile)

                    path_to_date = self.paths.sessionFile
                    path_to_date = path_to_date.split('_')
                    date = path_to_date[-2]
                    date = date.split("-")
                    date = [int(x) for x in date]
                    time = path_to_date[-1]
                    time = time.split(".")[:-1]
                    time = [int(y) for y in time]
                    self._datetime = datetime(year=date[0], month=date[1], day=date[2], hour=time[0], minute=time[1], second=time[2])

                    if (np.ndim(data) == 1):
                        add_channel = Channel(data)
                        add_channel.sampleRate = sampleRate
                        add_channel.number= 0
                        self._channels.append(add_channel)
                    else: 
                        for i in range(np.ndim(data)): 
                            add_channel = Channel(np.transpose(data)[i] )
                            add_channel.sampleRate = sampleRate
                            add_channel.number = i
                            self.channels.append(add_channel) 
                elif self.paths.sessionFile.endswith(".m4a"):
                    #To Do.  Restucture this.
                    #tag = TinyTag.get(self.paths.sessionFile)
                    #print("done this")
                    #self._sampleRate  = tag.sampleRate
                    #data = AudioSegment.from_file(sessionpath, "m4a")
                    #print("done that")
                    #self._sampleRate = sampleRate
                    if (np.ndim(data) == 1):
                        add_channel = Channel(data = data, fs= self._sampleRate)
                        add_channel.number = 0
                        self._channels.append(add_channel)
                    else: 
                        for i in range(np.ndim(data)): 
                            #add_channel = Channel(data = np.transpose(data)[i], fs= self._sampleRate)
                            add_channel.number = i
                            self.channels.append(add_channel) 
            except BaseException as err:
                print(f"Unexpected {err}, {type(err)}")

        if (self.paths.eventFile != None):
            try: 
                with open(self.paths.eventFile ) as event_file:
                    timestamps = event_file.readlines()
                    timestamps = timestamps[2:]
                    for timestamp in timestamps:
                        eventname, eventtime = timestamp.strip().split(',')
                        eventtime = float(eventtime)
                        if eventname.startswith('_'):

                            if eventname.startswith('_ch'):
                                if any(neuron.name == eventname[1:] for neuron in self._neurons):
                                    n = self._neurons[eventname[1:]]
                                    n.timestamps.append(eventtime)                            
                                else:
                                    n = Neuron(eventname[1:])
                                    n.timestamps.append( eventtime)
                                    self._neurons.append(n)
                            else:
                                print("to do :" + eventname)
                        else:
                            if any(event.name == eventname for event in self._events):
                                self._events[eventname].timestamps.append(eventtime)
                                #print('Appending time to event:' + eventname )
                            else:
                                e = Event(eventname)
                                #print('Adding new event:' + eventname )
                                e.timestamps.append(eventtime)
                                self._events.append(e)


            except BaseException as err:
                print(f"Unexpected {err}, {type(err)}")
            except: 
                print("This event file doesn't exist in your working directory.")
            
   
    @property
    def sessionID(self):
        return self._sessionID
    @sessionID.setter
    def sessionID(self, value):
        self._sessionID = value
        
    @property
    def subject(self): #subject is returned
      return self._subject
    @subject.setter
    def subject(self, value):
        self._subject = value
    
    @property
    def datetime(self): #date and time of session is returned
      return self._datetime
    @datetime.setter
    def datetime(self, value):
        self._datetime= value    
    
    @property
    def paths(self): #returns list of channel objects
      return self._paths
      
    @property
    def channels(self): #returns list of channel objects
      return self._channels
    
    @property
    def events(self):
      return self._events

    @property
    def neurons(self):
      return self._neurons
    
    @property
    def plot(self):
        return self._plot 
    
class Sessions(UserList):
    def __init__(self, sessions = None):
        super().__init__(sessions)
        if sessions is None:
            self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, session):
        return self.data.append(session)