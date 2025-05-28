# tests/test_plotting.py

import unittest
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for testing
import numpy as np
from spikertools.core import Session

class TestPlotting(unittest.TestCase):

    def setUp(self):
        # Path to the rate coding experiment data for neuron tests
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.wav_file_path = os.path.join(current_dir, '../data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29.wav')
        self.events_file_path = os.path.join(current_dir, '../data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29-events.txt')

        if not os.path.exists(self.wav_file_path):
            raise FileNotFoundError(f"WAV file not found at: {self.wav_file_path}")
        if not os.path.exists(self.events_file_path):
            raise FileNotFoundError(f"Events file not found at: {self.events_file_path}")

        # Initialize the Session with the WAV and events files
        self.session = Session(self.wav_file_path, self.events_file_path)

        # Assign colors to events and neurons for consistency
        color_map = {'1': 'r', '2': 'g', '3': 'b'}
        for event in self.session.events:
            if event.name in color_map:
                event.color = color_map[event.name]

        for neuron in self.session.neurons:
            neuron.color = 'k'  # Default black

    def tearDown(self):
        pass  # No cleanup needed

    def test_plot_session(self):
        try:
            self.session.plots.plot_session(show=False)
        except Exception as e:
            self.fail(f'plot_session raised an exception: {e}')

    def test_plot_intervals_histogram_event(self):
        try:
            self.session.plots.plot_intervals_histogram(data_type='event', name='1', show=False)
        except Exception as e:
            self.fail(f'plot_intervals_histogram (event) raised an exception: {e}')

    def test_plot_intervals_histogram_neuron(self):
        try:
            self.session.plots.plot_intervals_histogram(data_type='neuron', name='_ch0_neuron0', show=False)
        except Exception as e:
            self.fail(f'plot_intervals_histogram (neuron) raised an exception: {e}')

    def test_plot_raster(self):
        try:
            self.session.plots.plot_raster(neuron_name='_ch0_neuron0', event_name='1', show=False)
        except Exception as e:
            self.fail(f'plot_raster raised an exception: {e}')

    def test_plot_peth(self):
        try:
            self.session.plots.plot_peth(neuron_name='_ch0_neuron0', event_name='1', show=False)
        except Exception as e:
            self.fail(f'plot_peth raised an exception: {e}')

    def test_plot_epochs(self):
        try:
            event = next(e for e in self.session.events if e.name == '1')
            channel = self.session.channels[0]
            self.session.plots.plot_epochs(event=event, channel=channel, show=False)
        except Exception as e:
            self.fail(f'plot_epochs raised an exception: {e}')

    def test_plot_spectrogram(self):
        # Use EEG data for spectrogram test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eeg_wav_file_path = os.path.join(current_dir, '../data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25.wav')
        eeg_events_file_path = os.path.join(current_dir, '../data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25-events.txt')

        if not os.path.exists(eeg_wav_file_path):
            raise FileNotFoundError(f"WAV file not found at: {eeg_wav_file_path}")
        if not os.path.exists(eeg_events_file_path):
            raise FileNotFoundError(f"Events file not found at: {eeg_events_file_path}")

        eeg_session = Session(eeg_wav_file_path, eeg_events_file_path)

        # Assign colors to events
        for event in eeg_session.events:
            if event.name == 'Close':
                event.color = 'purple'
            elif event.name == 'Open':
                event.color = 'orange'

        try:
            eeg_session.plots.plot_spectrogram(channel_index=0, freq_range=(0, 30), event_names=['Open', 'Close'], show=False)
        except Exception as e:
            self.fail(f'plot_spectrogram raised an exception: {e}')

    def test_plot_average_power(self):
        # Use EEG data for average power spectrum test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eeg_wav_file_path = os.path.join(current_dir, '../data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25.wav')
        eeg_events_file_path = os.path.join(current_dir, '../data/eeg/alpha_wave/BYB_Recording_2015-07-26_21.47.25-events.txt')

        if not os.path.exists(eeg_wav_file_path):
            raise FileNotFoundError(f"WAV file not found at: {eeg_wav_file_path}")
        if not os.path.exists(eeg_events_file_path):
            raise FileNotFoundError(f"Events file not found at: {eeg_events_file_path}")

        eeg_session = Session(eeg_wav_file_path, eeg_events_file_path)

        # Assign colors to events
        for event in eeg_session.events:
            if event.name == 'Close':
                event.color = 'purple'
            elif event.name == 'Open':
                event.color = 'orange'

        try:
            eeg_session.plots.plot_average_power(event_names=['Open', 'Close'], freq_range=(0, 30), epoch_window=(0, 5), channel_index=0, show=False)
        except Exception as e:
            self.fail(f'plot_average_power raised an exception: {e}')

    def test_plot_erp(self):
        # Use P300 data for ERP test
        current_dir = os.path.dirname(os.path.abspath(__file__))
        p300_wav_file_path = os.path.join(current_dir, '../data/eeg/p300/BYB_Recording_2019-06-11_13.23.58.wav')
        p300_events_file_path = os.path.join(current_dir, '../data/eeg/p300/BYB_Recording_2019-06-11_13.23.58-events.txt')

        if not os.path.exists(p300_wav_file_path):
            raise FileNotFoundError(f"WAV file not found at: {p300_wav_file_path}")
        if not os.path.exists(p300_events_file_path):
            raise FileNotFoundError(f"Events file not found at: {p300_events_file_path}")

        p300_session = Session(p300_wav_file_path, p300_events_file_path)

        # Assign names and colors to events
        for event in p300_session.events:
            if event.name.strip() == '1':
                event.name = 'standard'
                event.color = 'blue'
            elif event.name.strip() == '2':
                event.name = 'oddball'
                event.color = 'red'

        try:
            p300_session.plots.plot_erp(event_names=['standard', 'oddball'], epoch_window=(-0.2, 0.8), channel_index=0, show=False)
        except Exception as e:
            self.fail(f'plot_erp raised an exception: {e}')

if __name__ == '__main__':
    unittest.main()
