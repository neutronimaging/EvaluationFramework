import numpy as np
import scipy
import skimage.filters as flt

from ..core.algorithm import Algorithm

class identity(Algorithm):
    def __init__(self, name="identity"):
        super().__init__(name)

    def _run(self, X):
        return X
    
class gauss_filter(Algorithm):
    def __init__(self, name="gauss_filter", sigma=1.0):
        super().__init__(name)
        self.sigma = sigma

    def _run(self, X):
        return scipy.ndimage.gaussian_filter(X, sigma=self.sigma)

class median_filter(Algorithm):
    def __init__(self, name="median_filter", footprint_size=3):
        super().__init__(name)
        self._footprint_size = footprint_size

    def _run(self, X):
        return flt.median(X, footprint=np.ones((self._footprint_size, self._footprint_size)))