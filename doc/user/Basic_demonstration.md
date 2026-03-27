# Basic demonstration run
The demonstration run is set up to use
- Two 512x512 images with Gaussian noise
- The default metrics (MSE, MAE, R2, and RMSE)
- Two algorithms (Identity operation, Gaussian filter)
  
## Setup
In the repo root run
```bash
pixi install
pixi shell
```
## Run the test script
Go to the demo script folder
```bash
cd src/scripts
```
Run the demo script
```bash
python3 basic_evaluation.py
```
This should produce a table like this

| Run number |algorithm |  dataset |       timing |      mse |      mae |       r2 |     rmse |
|---|---|---|---|---|---|---|---|
|0 |     identity |  dataset1 | 3.749956e-07 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
|1 |     identity | dataset2  | 2.919987e-07 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
|2 | gauss_filter | dataset1  |2.887417e-03  | 0.759632 | 0.695294 | 0.240264 | 0.871569 |
|3 | gauss_filter | dataset2  | 2.838333e-03 | 0.759769 | 0.695865 | 0.241485 | 0.871647 |

