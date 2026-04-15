import numpy as np
from skimage.metrics import structural_similarity as ssim
import scipy.stats as stats

from ..core.metric import Metric


class mse(Metric):
    def __init__(self, name="mse"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for mse metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name :((y_true - y_pred) ** 2).mean() }  
    
class mae(Metric):
    def __init__(self, name="mae"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for mae metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : np.abs(y_true - y_pred).mean() } 
    
class r2(Metric):
    def __init__(self, name="r2"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_pred cannot be None for r2 metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return { self.name : r2_score }
    
class rmse(Metric):
    def __init__(self, name="rmse"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_pred cannot be None for rmse metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : np.sqrt(((y_true - y_pred) ** 2).mean()) }
    
class skewness(Metric):
    def __init__(self, name="skewness"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        y_pred = np.asarray(y_pred)
        return { self.name : stats.skew(y_pred.ravel()) }

class kurtosis(Metric):
    def __init__(self, name="kurtosis"):
        super().__init__(name)

    def compute(self, y_pred, y_true  = None):
        y_pred = np.asarray(y_pred)
        return { self.name : stats.kurtosis(y_pred.ravel()) }

class mae_percentage(Metric):
    def __init__(self, name="mae_percentage"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for mae_percentage metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : (np.abs(y_true - y_pred) / np.abs(y_true)).mean() * 100 }
    
class mse_percentage(Metric):
    def __init__(self, name="mse_percentage"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for mse_percentage metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : ((y_true - y_pred) ** 2 / np.abs(y_true)).mean() * 100 }
    
class rmse_percentage(Metric):
    def __init__(self, name="rmse_percentage"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for rmse_percentage metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : np.sqrt(((y_true - y_pred) ** 2 / np.abs(y_true)).mean()) * 100 }
    
class r2_percentage(Metric):
    def __init__(self, name="r2_percentage"):
        super().__init__(name)

    def compute(self, y_true, y_pred = None):
        if y_pred is None:
            raise ValueError("y_pred cannot be None for r2_percentage metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return { self.name : r2_score * 100 }
    
class explained_variance(Metric):
    def __init__(self, name="explained_variance"):
        super().__init__(name)

    def compute(self, y_true, y_pred = None):
        if y_pred is None:
            raise ValueError("y_pred cannot be None for explained_variance metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        variance_true = np.var(y_true)
        variance_residual = np.var(y_true - y_pred)
        explained_var = 1 - (variance_residual / variance_true) if variance_true != 0 else 0.0
        return { self.name : explained_var }
    
class explained_variance_percentage(Metric):
    def __init__(self, name="explained_variance_percentage"):
        super().__init__(name)

    def compute(self, y_true, y_pred = None):
        if y_pred is None:
            raise ValueError("y_pred cannot be None for explained_variance_percentage metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        variance_true = np.var(y_true)
        variance_residual = np.var(y_true - y_pred)
        explained_var = 1 - (variance_residual / variance_true) if variance_true != 0 else 0.0
        return { self.name : explained_var * 100 }
    
class mean_absolute_percentage_error(Metric):
    def __init__(self, name="mean_absolute_percentage_error"):
        super().__init__(name)

    def compute(self, y_true, y_pred = None):
        if y_pred is None:
            raise ValueError("y_pred cannot be None for mean_absolute_percentage_error metric.")
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return { self.name : (np.abs(y_true - y_pred) / np.abs(y_true)).mean() * 100 }

class mean(Metric):
    def __init__(self, name="mean"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        y_pred = np.asarray(y_pred)
        return { self.name : y_pred.mean() }
    
class median(Metric):
    def __init__(self, name="median"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        y_pred = np.asarray(y_pred)
        return { self.name : np.median(y_pred) }
    
class std(Metric):
    def __init__(self, name="std"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        y_pred = np.asarray(y_pred)
        return { self.name : y_pred.std() }
    
class mad(Metric):
    def __init__(self, name="mad"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        y_pred = np.asarray(y_pred)
        median = np.median(y_pred)
        mad_value = np.median(np.abs(y_pred  - median))
        return { self.name : mad_value }
    
class mSSIM(Metric):
    def __init__(self, name="mSSIM"):
        super().__init__(name)

    def compute(self, y_pred, y_true = None):
        if y_true is None:
            raise ValueError("y_true cannot be None for mSSIM metric.")

        if y_true.ndim != 2 or y_pred.ndim != 2:
            raise ValueError("y_true and y_pred must be 2D arrays for mSSIM metric.")   
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Placeholder for actual mSSIM computation

        return { self.name : ssim(y_true, y_pred, data_range=y_true.max() - y_true.min()) }