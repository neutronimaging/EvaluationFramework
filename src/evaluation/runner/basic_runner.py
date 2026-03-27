from ..metrics import basemetrics
from ..core.algorithm import Algorithm
from ..core.dataset import Dataset
from ..core.metric import Metric

class BasicRunner:
    def __init__(self, metrics: list[Metric] = []):
        if len(metrics) == 0:
            self._metrics = [basemetrics.mse(), basemetrics.mae(), basemetrics.r2(), basemetrics.rmse()]
        else:
            self._metrics = metrics

    def run(self, datasets: list[Dataset], algorithms: list[Algorithm]):
        run_results = []
        for algorithm in algorithms:
            for dataset in datasets:
                y_true = dataset.data

                y_pred = algorithm.run(y_true)
                results = {'algorithm': algorithm.name, 
                           'dataset':   dataset.name,
                           'timing':    algorithm.timing}
            
                for metric in self._metrics:
                    res = metric.compute(y_true, y_pred)
                    results.update(res)
                
                run_results.append(results)

        return run_results
    