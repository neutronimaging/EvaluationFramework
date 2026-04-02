import numpy as np
from ..core.metric import Metric
from ..utils import metrics

class mad_outlier_fraction(Metric):
    def __init__(self, name="mad_outlier_fraction"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        return { self.name : metrics.mad_outlier_fraction(y_pred) }
    
class local_mad_outlier_fraction(Metric):
    def __init__(self, name="local_mad_outlier_fraction", footprint_size=7):
        super().__init__(name)
        self._footprint_size = footprint_size

    def compute(self, y_pred, y_true = None):
        return { self.name : metrics.local_mad_outlier_fraction(y_pred, footprint=np.ones((self._footprint_size, self._footprint_size))) }
    
class fitted_outlier_fraction(Metric):
    def __init__(self, name="fitted_outlier_fraction", k=3.0):
        super().__init__(name)
        self._k = k

    def compute(self, y_pred, y_true = None):
        return { self.name : metrics.fitted_outlier_fraction(y_pred, k=self._k) }  