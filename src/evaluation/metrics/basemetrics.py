import numpy as np

from ..core.metric import Metric

class mse(Metric):
    def __init__(self, name="mse"):
        super().__init__(name)

    def compute(self, y_true, y_pred = None):
        if y_pred is None:
            raise ValueError("y_pred cannot be None for mse metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return ((y_true - y_pred) ** 2).mean()