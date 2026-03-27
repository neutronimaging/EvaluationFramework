import numpy as np
import scipy

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