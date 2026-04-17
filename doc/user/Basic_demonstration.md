# Basic demonstration run
The demonstration run is set up to use
- Two 512x512 images with Gaussian noise
- The default metrics (MSE, MAE, R2, and RMSE)
- Two algorithms (Identity operation, Gaussian filter)

A note: The examples presented here are stored in the repository under [src/scripts](https://github.com/neutronimaging/EvaluationFramework/tree/main/src/scripts)

## Implement the test script
The basic evaluation example described here is a full script to be run from the terminal command line. The full script is already implemented and [can be found in the repository](https://github.com/neutronimaging/EvaluationFramework/blob/main/src/scripts/basic_evaluation.py)

An evaluation run has three steps
- Configuration
- Running 
- Reporting

### Configuration
In the configuration step you create instances of the modules you want to use during the evaluation. 
There will be modules for
- Data generation
- Processing algorithms. These are the algorithms you want to evaluate. I.e. they will process the provided test data.
- Metrics. These are the metrics you use to measure the performance of the algorithms. The output is a scalar number that will be added to the result table.

```python
# code to create lists of modules 
```
The modules will be added to the runner instance
```python
# Code to create a runner instance with the modules
```

### Run the evaluation
The configured runner will be used to execute the list of algorithms on the provided test data and provide a list of dictionaries with the results from the runs.

```python
# code to start the runner
```
### Reporting
There are different ways to present the results. The easiest is to print them in the terminal, maybe first create a dataframe that allows better formatting. 
```python
df = pd.DataFrame(results)
print(df)
```

## Run the test script
Follow the steps described in the [preparation](preparation.md) section of this documentation to provide the needed python environment for the evaluation. Open a terminal and go to the location of the script.
Go to either to the location of the script you wrote or to the demo script folder
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

## A more complicated run
The previous example only printed the results on the terminal. [Here is a more complicated run scheme](https://github.com/neutronimaging/EvaluationFramework/blob/main/src/scripts/test_eval.py) that also saves the runs as json files with time stamps.
