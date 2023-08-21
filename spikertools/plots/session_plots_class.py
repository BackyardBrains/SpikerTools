import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

class SessionPlots:
    def __init__(self, key=None):
        self._Session = None

        from spikertools import Session
        if isinstance(key, Session):
            self._Session = key

    def overview(self, session=None):
        if session is None:
            session = self._Session

        fig, f1_axes = plt.subplots(ncols=1, nrows=3 + len(session.channels), constrained_layout=True, figsize=(11, 8.5))
        idx = 0

        session_overview = f"""
        File Name: {session.paths.sessionFile}
        Date and Time : {session.datetime}
        Sample Rate: {session.channels[0].sampleRate}
        Session duration (in samples): {len(session.channels[0].time)} samples
        Session duration (in hh:mm:ss): {time.strftime('%H:%M:%S', time.gmtime(len(session.channels[0].time)/session.channels[0].sampleRate))} 
        Session ID: {session.sessionID}
        Subject: {session.subject}
        """
        plt.axes(f1_axes[idx])
        plt.text(0, 0, session_overview)
        plt.title(f"Session Overview: {session.sessionID}")
        plt.axis("off")
        idx = idx + 1

        for chan in session.channels:
            plt.axes(f1_axes[idx])
            idx = idx + 1
            plt.plot(chan.time, chan.data, color=chan.color)
            plt.xlim(0, chan.time[-1])
            plt.title(f"Channel {chan.number}: {chan.name}:  Mean: {round(chan.mean, 2)} | Standard Dev: {round(chan.std, 2)}")
            plt.axis("off")

        plt.axes(f1_axes[idx])
        self.events()

        ax = f1_axes[idx]
        ax.axis('off')
        idx = idx + 1

        plt.axes(f1_axes[idx])
        col_index = 0
        for ev in session.events:
            inter_event_interval = np.diff(ev.timestamps)
            plt.text(0, 0.75 - ((0.75 / len(session.events)) * col_index), f"       Event [{ev.name}] (n = {len(ev.timestamps)}): Mean Inter-Event Interval = {inter_event_interval[0]:.2f}s \n", color=ev.color)
            col_index += 1
        plt.axis("off")

        plt.show()

    # plot helper functions
    def events(self, session=None, timerange=None):
        if session is None:
            session =  self._Session
        if timerange is None:
            if len(session.channels) > 0:
                timerange = [0, max(session.channels[0].time)]
        row_index = 0
        #n_colors = len(self._events)
        event_plots = []
        event_labels = []
        event_colors = []
        for event in session.events:
            color = event.color
            time_markers = event.timestamps
            time_markers_interval = []
            event_label = f"Event {event}"
            event_labels.append(event_label)
            for marker in time_markers: 
                if (timerange[0]<= marker) and (timerange[1] >= marker):
                    time_markers_interval.append(marker)
            #markerlength = 10*(max_data - min_data)
            markerlength = 10
            y = [0.5*(row_index/(len(session.events)))+0.25]*len(time_markers_interval)
            event_plot = plt.scatter(time_markers_interval, y, c = color, marker = "|")
            plt.ylim((1,0))
            #event_plot = plt.eventplot(time_markers_interval, lineoffsets=offset, linelengths= markerlength, linewidths = 1, colors = color, label ='Event')
            time_axis_lim= session.channels[0].time[-1]
            plt.xlim(0,time_axis_lim)
            event_plots.append(event_plot)
            row_index = row_index + 1
        return event_labels, event_plots, event_colors

    def monte_carlo_avg(self, channel, event, timewindow):
        
        if type(channel) == int:
            channel = self._channels[channel]
        channel_data = channel.data
        onsets = event.timestamps
        num_mc_epochs = len(onsets)
        num_mc_sim = 100

        sample_size = int((timewindow[1]-timewindow[0])*self._samplerate)
        mc_avgs = np.zeros((num_mc_sim, sample_size))
        
        for sim in range(num_mc_sim):
            
            mc_event_start_idxs = []
            for i in range(num_mc_epochs):
                import random
                mc_event_start_idxs.append(random.randint(0, len(channel_data)-sample_size))
                 
            data_mc = []
            for start in mc_event_start_idxs:
                data_mc.append( channel_data[start: start+sample_size] )
                
            avg_raw_epoch_mc = np.mean(data_mc, axis=0)
            mc_avgs[sim,:] = avg_raw_epoch_mc

        avg_raw_epoch_mc = np.mean(mc_avgs,0)
        mc_std = np.std(mc_avgs,0)
        mc_avg_epoch = avg_raw_epoch_mc
        mc_plus_epoch = avg_raw_epoch_mc + (2*mc_std)
        mc_minus_epoch = avg_raw_epoch_mc - (2*mc_std)
        
        return mc_avg_epoch, mc_plus_epoch, mc_minus_epoch



