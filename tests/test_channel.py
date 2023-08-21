import unittest
import numpy as np
from spikertools import Channel   # import your channel class

class TestChannel(unittest.TestCase):   # define a class for the test case

    def setUp(self):
        # create a channel object for use in all tests
        self.channel = Channel(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fs=10000, filterfreqs=[0, 2000], label="Ch1", color='b')

    def test_attributes(self):
        # test property getters
        self.assertEqual(self.channel.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(self.channel.fs, 10000)
        self.assertEqual(self.channel.originalfiltersettings, [0, 2000])
        self.assertEqual(self.channel.label, 'Ch1')
        self.assertEqual(self.channel.color, 'b')
        
        # test property setters
        self.channel.data = [5, 6, 7]
        self.channel.fs = 5000
        self.channel.originalfiltersettings = [0, 1000]
        self.channel.label = 'Ch2'
        self.channel.color = 'r'
        
        self.assertEqual(self.channel.data, [5, 6, 7])
        self.assertEqual(self.channel.fs, 5000)
        self.assertEqual(self.channel.originalfiltersettings, [0, 1000])
        self.assertEqual(self.channel.label, 'Ch2')
        self.assertEqual(self.channel.color, 'r')

    def test_filt(self):
        self.channel.filt(500, 'lp')
        # perform some kind of assertion that checks the result of the filtering

    def test_normalize(self):
        self.channel.normalize('mean')
        # perform some kind of assertion that checks the result of the normalization

# run the tests
if __name__ == "__main__":
    unittest.main()
