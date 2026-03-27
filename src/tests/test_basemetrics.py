import numpy as np
import pytest 
import evaluation.metrics.basemetrics as basemetrics

def test_mse():
    metric = basemetrics.mse()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert metric.compute(y_true, y_pred) == 0.0

    y_pred = [2, 3, 4]
    assert metric.compute(y_true, y_pred) == 1.0