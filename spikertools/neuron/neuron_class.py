class Neuron:
    def __init__(self, key=None, timestamps=[], threshold_high=None, threshold_low=None):
        import re

        if isinstance(key, Neuron):
            self._name = key.name
            self._timestamps = list(key.timestamps)
            self._threshold_high = key.threshold_high  
            self._threshold_low = key.threshold_low 
            self._channelID = key._channelID
            self._neuronID = key._neuronID

        elif isinstance(key, str):
            self._name = key
            self._timestamps = timestamps
            self._threshold_high = threshold_high 
            self._threshold_low = threshold_low 
            self._channelID = int(re.search(r'ch(\d+)_neuron', self._name ).group(1))
            self._neuronID = int(re.search(r'ch\d+_neuron(\d+)', self._name ).group(1))

        else:
            raise ValueError("Key should be a string or neuron class")

    def __repr__(self): #Allows you to nicely list event names instead of <<objects>>s
        return self._name
        
    @property
    def name(self):
        return self._name if self._name is not None else ""

    @name.setter
    def name(self, key):
        if not isinstance(key, str):
            raise ValueError("Invalid name format. Please provide a string value.")
        self._name = key

    @property
    def threshold_high(self):
        return self._threshold_high

    @property
    def threshold_low(self):
        return self._threshold_low

    @property
    def timestamps(self):
        return self._timestamps



class Neurons(list):
    def __init__(self):
         self.items = []

    def append(self, item):
         self.items.append (item)

    def __repr__(self):
        return self.items

    def __str__(self):
        return self.neuronNames

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
            n = Neuron(key)
            n.timestamps.append(float(newvalue))
            self.items.append(n)

    @property
    def neuronNames(self):
        return str(self.items)
 