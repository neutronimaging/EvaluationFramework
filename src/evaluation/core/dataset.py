import numpy as np

class Dataset:
    def __init__(self, name, data, ground_truth = None):
        self._name = name
        self._data = data
        self._ground_truth = ground_truth

    @property
    def name(self):
        return self._name
    
    @property
    def data(self):
        return self._data

    @property
    def ground_truth(self):
        return self._ground_truth