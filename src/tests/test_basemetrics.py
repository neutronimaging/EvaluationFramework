import numpy as np
import pytest 
import evaluation.metrics.basemetrics as basemetrics

def test_mse():
    metric = basemetrics.mse()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 0.0

    y_pred = [2, 3, 4]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 1.0

def test_mae():
    metric = basemetrics.mae()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 0.0

    y_pred = [2, 3, 4]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 1.0  

def test_mse_no_pred():
    metric = basemetrics.mse()
    y_true = [1, 2, 3]
    with pytest.raises(ValueError):
        metric.compute(y_true) 

def test_mae_no_pred():
    metric = basemetrics.mae()
    y_true = [1, 2, 3]
    with pytest.raises(ValueError):
        metric.compute(y_true)  

def test_r2():
    metric = basemetrics.r2()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 1.0

    y_pred = [2, 3, 4]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == -0.5

def test_rmse():
    metric = basemetrics.rmse()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 0.0

    y_pred = [2, 3, 4]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 1.0

def test_mae_percentage():
    metric = basemetrics.mae_percentage()
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 0.0

    y_pred = [2, 3, 4]
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 61.11111111111111