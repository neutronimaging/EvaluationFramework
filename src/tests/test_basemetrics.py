import numpy as np
import pytest 
from skimage.metrics import structural_similarity as ssim
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

# def test_mae_percentage():
#     metric = basemetrics.mae_percentage()
#     y_true = [1, 2, 3]
#     y_pred = [1, 2, 3]
#     res = metric.compute(y_true, y_pred)
#     assert res[metric.name] == 0.0

#     y_pred = [2, 3, 4]
#     res = metric.compute(y_true, y_pred)
#     assert res[metric.name] == 61.11111111111111

def test_mSSIM():
    metric = basemetrics.mSSIM()
    N=100
    x,y = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, N-1, N))
    w = x+y*N
    y_true = w
    y_pred = w
    res = metric.compute(y_true, y_pred)
    assert res[metric.name] == 1.0
    
    np.random.seed(42)
    y_pred = w+np.random.normal(0, 100, size=w.shape)
    res = metric.compute(y_true, y_pred)
    assert np.round(res[metric.name],6) ==  0.946007