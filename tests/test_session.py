import unittest
from spikertools import Session

class TestSesssion(unittest.TestCase):

    def setUp(self):
        self.session = Session()

    def test_channels(self):
        # Test that the channels were initialized as expected.
        self.assertEqual(self.session.channels, [])

    def test_sessionpath(self):
        self.assertEqual(self.session.paths.sessionFile, None)

    def test_eventspath(self):
        self.assertEqual(self.session.paths.eventFile, None)
      
    def test_sessionID(self):
        self.assertEqual(self.session.sessionID, None)

    def test_subject(self):
        self.assertEqual(self.session.subject, None)

    def test_datetime(self):
        self.assertEqual(self.session.datetime, None)

    def test_events(self):
        self.assertEqual(len(self.session.events), 0)

    def test_set_channels(self):
        from spikertools import Channel
        self.session.channels.append(Channel())
        self.session.channels.append(Channel())
        self.assertEqual(len(self.session.channels), 2)

    def test_set_sessionpath(self):
        self.session.paths.sessionFile = "/path/to/session"
        self.assertEqual(self.session.paths.sessionFile, "/path/to/session")
        
    def test_set_eventspath(self):
        self.session.paths.eventFile = "/path/to/events"
        self.assertEqual(self.session.paths.eventFile, "/path/to/events")

    def test_set_events(self):
        from spikertools import Event
        event = Event("Dummy")  # replace with your Events instance
        self.session.events.append( event )
        self.assertEqual(self.session.events[0], event)      

    def test_loading_file(self):
        self.session = Session('./demo/data/rate_coding_example_data/BYB_Recording_2022-01-13_13.18.29.wav')   
        
        self.session.events['1'].name = 'Soft'   
        self.session.events['Soft'].color = 'r'   
        
        self.session.events['2'].name = 'Medium'    
        self.session.events['Medium'].color = 'b'   
        
        self.session.events['3'].name = 'Hard'    
        self.session.events['Hard'].color = 'g'   
        
       
       
        
        self.session.channels[0].name = 'Cockroach Spikes'
        self.session.channels[0].filter( 'hp', 500)
        self.session.channels[0].filters.hardware = [300, 1200]
        self.session.channels[0].filters.sofware = [500, 10000]

        import copy
        lfp = copy.deepcopy(self.session.channels[0])
        self.session.channels.append( lfp )
        self.session.channels[1].number = 1
        self.session.channels[1].filter('lp', 100)
        self.session.channels[1].name = 'Cockroach LFP'
        self.session.channels[1].filters.sofware = [1, 100]

        self.session.sessionID = "Rate Coding Session"
        self.session.subject = "Cockroach C18"
        #from spikertools import SessionPlots
        #sp = SessionPlots()
        self.session.plot.overview()

        self.assertEqual(len( self.session.channels), 2)   
        self.assertEqual(len( self.session.events), 3) 
        self.assertEqual(len( self.session.neurons), 1) 
        self.assertEqual(len(self.session.neurons[0].timestamps), 379)
        
if __name__ == '__main__':
    unittest.main()