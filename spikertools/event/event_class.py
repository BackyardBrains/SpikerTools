class Event:
    def __init__(self, key=None):
        self._timestamps = []
        self._eventNumber = None 
        self._color = 'k'  

        if isinstance(key, Event):
            self._name = key.name
            self._eventNumber = key.number
            self._timestamps = list(key.timestamps)
        elif isinstance(key, str):
            self.name = key
            if key.isdigit():
                self.number = int(key)
        elif isinstance(key, int):
            self.number = key
            self.name = str(key)
        else:
            raise ValueError("Key should be a string or integer")

    def __repr__(self): #Allows you to nicely list event names instead of <<objects>>s
        return self._name
        
    @property
    def color(self):
        return self._color if self._color is not None else "k"
    @color.setter
    def color(self, key):
        if not isinstance(key, str):
            raise ValueError("Invalid name format. Please provide a string value.")
        self._color = key
   
    @property
    def name(self):
        return self._name if self._name is not None else ""
    @name.setter
    def name(self, key):
        if not isinstance(key, str):
            raise ValueError("Invalid name format. Please provide a string value.")
        self._name = key

    @property
    def number(self):
        return self._eventNumber
    @number.setter
    def number(self, num):
        if not isinstance(num, int):
            raise ValueError("Invalid event number format. Please provide an integer value.")
        self._eventNumber = num

    @property
    def timestamps(self):
        return self._timestamps

class Events:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

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
        found = False
        for item in self.items:
            if item.name == key:
                item.timestamps.append(float(newvalue))
                found = True
                break
        if not found:
            e = Event(key)
            e.timestamps.append(float(newvalue))
            self.items.append(e)

    @property
    def eventNames(self):
        return str(self.items)
 