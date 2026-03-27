import numpy as np
import timeit

class Algorithm:
    _name = None
    def __init__(self, name):
        self._name = name
        self._timing = None

    def run(self, X):
        start_time = timeit.default_timer()
        result = self._run(X)
        end_time = timeit.default_timer()
        self._timing = end_time - start_time
        
        return result
    
    def _run(self, X) :
        raise NotImplementedError("Subclasses should implement this method.")
    
    @property
    def name(self):
        return self._name
    
    @property
    def timing(self):
        return self._timing