import numpy as np

class Metric:
    _name = None
    def __init__(self, name):
        self._name = name

    def compute(self, y_pred, y_true = None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @property
    def name(self):
        return self._name
    