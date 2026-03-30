import numpy as np
from .dataset import Dataset

class ImageGenerator:
    def __init__(self, name : str, dims : list[int]):
        if dims is not None and (len(dims) != 2 and len(dims) != 3):
            raise ValueError("dims must be a list of 2 or 3 integers")
        
        self._dims = dims
        self._name = name

    def generate(self, batch_size : int, noise_level : float, params : dict = {}) -> list[Dataset]:
        if params is None:
            params = {}

        return self._generate(batch_size, noise_level, params)

    def _generate(self, batch_size : int, noise_level : float, params : dict = {}) -> list[Dataset]:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self):
        return self._name