import unittest
import numpy as np
from spikertools import Channel   # import your channel class

class TestChannel(unittest.TestCase):   # define a class for the test case

    def setUp(self):
        # create a channel object for use in all tests
        self.channel = Channel(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.channel.sampleRate =10000
        self.channel.filters.software = [0, 2000]
        self.channel.name = "Ch1"
        self.channel.color='b'

    def test_attributes(self):
        # test property getters
        self.assertEqual(self.channel.data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(self.channel.sampleRate, 10000)
        self.assertEqual(self.channel.filters.software, [0, 2000])
        self.assertEqual(self.channel.name, 'Ch1')
        self.assertEqual(self.channel.color, 'b')
        
        # test property setters
        self.channel.data = [5, 6, 7]
        self.channel.sampleRate  = 5000
        self.channel.filters.hardware = [300, 1300]
        self.channel.filters.software = [500, 1000]
        self.channel.filters.analysis = [0, 500]
        self.channel.name = 'Ch2'
        self.channel.color = 'r'
        
        self.assertEqual(self.channel.data, [5, 6, 7])
        self.assertEqual(self.channel.sampleRate, 5000)
        self.assertEqual(self.channel.filters.hardware, [300, 1300])
        self.assertEqual(self.channel.filters.software, [500, 1000])
        self.assertEqual(self.channel.filters.analysis, [0, 500])
        self.assertEqual(self.channel.name, 'Ch2')
        self.assertEqual(self.channel.color, 'r')

    def test_filt(self):
        orig_mean = self.channel.mean
        self.channel.filter('lp', 500)
        filtered_mean = self.channel.mean
        self.assertNotEqual(orig_mean, filtered_mean)
        
    def test_normalize(self):
        orig_mean = self.channel.mean
        self.channel.normalize('mean')
        self.assertLess(self.channel.mean, orig_mean)
        
        # perform some kind of assertion that checks the result of the normalization

# run the tests
if __name__ == "__main__":
    unittest.main()
