import unittest
from spikertools import Event, Events

class TestEvent(unittest.TestCase):

    def test_create_event(self):
        event = Event('test_event')
        self.assertEqual(event.name, 'test_event')
        self.assertEqual(event.number, None)
        self.assertEqual(event.timestamps, [])

    def test_create_event_from_int(self):
        event = Event(5)
        self.assertEqual(event.name, '5')
        self.assertEqual(event.number, 5)
        self.assertEqual(event.timestamps, [])

    def test_create_event_copy(self):
        original = Event('original')
        original.number = 10
        original.timestamps.extend([1, 2, 3])
        copy = Event(original)
        self.assertEqual(copy.name, 'original')
        self.assertEqual(copy.number, 10)
        self.assertEqual(copy.timestamps, [1, 2, 3])

    def test_rename_event_copy(self):
        e = Events()
        original = Event('original')
        original.number = 10
        original.timestamps.extend([1, 2, 3])
        e.append( original)
        e['original'].name = 'copy'
        self.assertEqual(e['copy'].name, 'copy')
        self.assertEqual(e['copy'].number, 10)


    def test_invalid_key(self):
        with self.assertRaises(ValueError):
            Event(2.4)

class TestEvents(unittest.TestCase):

    def test_setitem_new(self):
        events = Events()
        events['test'] = 10.5
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, 'test')
        self.assertEqual(events[0].timestamps, [10.5])

    def test_setitem_existing(self):
        events = Events()
        events['test'] = 10.5
        events['test'] = 20.5
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, 'test')
        self.assertEqual(events[0].timestamps, [10.5, 20.5])

if __name__ == '__main__':
    unittest.main()
