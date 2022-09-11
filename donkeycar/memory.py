import logging

logger = logging.getLogger(__name__)


class Memory:
    """
    A convenience class to save key/value pairs.
    """
    def __init__(self, *args, **kw):
        self.d = {}
    
    def __setitem__(self, key, value):
        if type(key) is str:
            self.d[key] = value
        else:
            if type(key) is not tuple:
                key = tuple(key)
                value = tuple(value)
            for i, k in enumerate(key):
                self.d[k] = value[i]
        
    def __getitem__(self, key):
        if type(key) is tuple:
            return [self.d[k] for k in key]
        else:
            return self.d[key]
        
    def update(self, new_d):
        self.d.update(new_d)
        
    def put(self, keys, inputs):
        for k, i in zip(keys, inputs):
            self.d[k] = i

    def get(self, keys):
        result = [self.d.get(k) for k in keys]
        return result
    
    def keys(self):
        return self.d.keys()
    
    def values(self):
        return self.d.values()
    
    def items(self):
        return self.d.items()
        