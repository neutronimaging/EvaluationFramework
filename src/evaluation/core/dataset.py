import numpy as np
from ..io import readers as rd 

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
    
    def save(self, path):
        # Placeholder for actual saving logic
        # In a real implementation, you would use an appropriate library to save the data, e.g., numpy.save for arrays or PIL for images
        
        rd.save_TIFF(path, self._data)