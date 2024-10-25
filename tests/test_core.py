# tests/test_core.py

import unittest
import os
import numpy as np
from spikertools.core import Session, Channel, Event, Neuron
from datetime import datetime

class TestSession(unittest.TestCase):

    def setUp(self):
        # Path to the rate coding experiment data for neuron tests
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.wav_file_path = os.path.join(current_dir, '../data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29.wav')
        self.events_file_path = os.path.join(current_dir, '../data/neurons/rate_coding/BYB_Recording_2022-01-13_13.18.29-events.txt')

        # Verify that the files exist
        if not os.path.exists(self.wav_file_path):
            raise FileNotFoundError(f"WAV file not found at: {self.wav_file_path}")
        if not os.path.exists(self.events_file_path):
            raise FileNotFoundError(f"Events file not found at: {self.events_file_path}")

        # Initialize the Session with the WAV and events files
        self.session = Session(self.wav_file_path, self.events_file_path)

    def tearDown(self):
        pass  # No cleanup needed since we're using original files

    def test_session_initialization(self):
        self.assertIsNotNone(self.session)
        self.assertGreaterEqual(len(self.session.channels), 1)
        self.assertIsNotNone(self.session.sample_rate)
        # Check datetime is correctly extracted
        expected_datetime = datetime(2022, 1, 13, 13, 18, 29)
        self.assertEqual(self.session.datetime, expected_datetime)

    def test_channels(self):
        channel = self.session.channels[0]
        self.assertIsInstance(channel, Channel)
        self.assertEqual(channel.number, 0)
        # Assuming the WAV file has a certain number of samples
        expected_length = len(self.session.data)
        self.assertEqual(len(channel.data), expected_length)

    def test_events(self):
        self.assertGreaterEqual(len(self.session.events), 3)  # Events '1', '2', '3'
        event_names = [event.name for event in self.session.events]
        self.assertIn('1', event_names)
        self.assertIn('2', event_names)
        self.assertIn('3', event_names)
        # Test event colors
        for event in self.session.events:
            self.assertIsNotNone(event.color)

    def test_neurons(self):
        self.assertGreaterEqual(len(self.session.neurons), 1)
        neuron_names = [neuron.name for neuron in self.session.neurons]
        self.assertIn('_ch0_neuron0', neuron_names)
        neuron = self.session.neurons[0]
        self.assertGreaterEqual(len(neuron.timestamps), 1)
        # Test neuron color
        self.assertIsNotNone(neuron.color)

    def test_event_intervals(self):
        event = next((e for e in self.session.events if e.name == '1'), None)
        intervals = event.inter_event_intervals()
        self.assertIsNotNone(intervals)
        self.assertGreaterEqual(len(intervals), 1)
        # Test event rate
        rate = event.event_rate()
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0)

    def test_neuron_intervals(self):
        neuron = self.session.neurons[0]
        intervals = neuron.inter_spike_intervals()
        self.assertIsNotNone(intervals)
        self.assertGreaterEqual(len(intervals), 1)
        # Test firing rate
        rate = neuron.firing_rate()
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0)

    def test_channel_filtering(self):
        channel = self.session.channels[0]
        original_data = channel.data.copy()
        channel.filter(ftype='hp', cutoff=300)  # High-pass filter at 300 Hz
        self.assertFalse(np.array_equal(original_data, channel.data))
        # Check if the filter was recorded
        self.assertEqual(len(channel.filters_applied), 1)
        self.assertEqual(channel.filters_applied[0]['type'], 'hp')

    def test_channel_normalization(self):
        channel = self.session.channels[0]
        channel.normalize(method='zscore')
        self.assertAlmostEqual(np.mean(channel.data), 0, places=5)
        self.assertAlmostEqual(np.std(channel.data), 1, places=5)

class TestChannel(unittest.TestCase):

    def setUp(self):
        # Create a dummy channel with synthetic data
        data = np.random.randn(1000)  # 1000 samples of random data
        self.channel = Channel(data, sample_rate=10000, number=0, name='Test Channel')

    def tearDown(self):
        pass  # No cleanup needed

    def test_normalization_mean(self):
        original_data = self.channel.data.copy()
        self.channel.normalize(method='mean')
        expected_data = original_data - np.mean(original_data)
        np.testing.assert_array_almost_equal(self.channel.data, expected_data)

    def test_normalization_zscore(self):
        original_data = self.channel.data.copy()
        self.channel.normalize(method='zscore')
        expected_data = (original_data - np.mean(original_data)) / np.std(original_data)
        np.testing.assert_array_almost_equal(self.channel.data, expected_data)

    def test_filter_highpass(self):
        original_data = self.channel.data.copy()
        self.channel.filter(ftype='hp', cutoff=300)
        self.assertFalse(np.array_equal(original_data, self.channel.data))
        self.assertEqual(len(self.channel.filters_applied), 1)
        self.assertEqual(self.channel.filters_applied[0]['type'], 'hp')

class TestEvent(unittest.TestCase):

    def setUp(self):
        self.event = Event('Test Event', timestamps=[0.1, 0.5, 1.0], color='blue')

    def tearDown(self):
        pass  # No cleanup needed

    def test_event_attributes(self):
        self.assertEqual(self.event.name, 'Test Event')
        self.assertEqual(len(self.event.timestamps), 3)
        self.assertEqual(self.event.color, 'blue')
        intervals = self.event.inter_event_intervals()
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0], 0.4)
        self.assertEqual(intervals[1], 0.5)
        rate = self.event.event_rate()
        self.assertEqual(rate, 3 / (1.0 - 0.1))

class TestNeuron(unittest.TestCase):

    def setUp(self):
        self.neuron = Neuron('_ch0_neuron0', timestamps=[0.2, 0.6, 1.1], color='red')

    def tearDown(self):
        pass  # No cleanup needed

    def test_neuron_attributes(self):
        self.assertEqual(self.neuron.name, '_ch0_neuron0')
        self.assertEqual(len(self.neuron.timestamps), 3)
        self.assertEqual(self.neuron.color, 'red')
        intervals = self.neuron.inter_spike_intervals()
        self.assertEqual(len(intervals), 2)
        # Use assertAlmostEqual for floating-point comparisons
        self.assertAlmostEqual(intervals[0], 0.4, places=7)
        self.assertAlmostEqual(intervals[1], 0.5, places=7)
        rate = self.neuron.firing_rate()
        self.assertEqual(rate, 3 / (1.1 - 0.2))

if __name__ == '__main__':
    unittest.main()
