import numpy as np
import pandas as pd

import sys
sys.path.append("../")
import evaluation.algorithms.basic_algorithms as alg
import evaluation.core.dataset as ds
import evaluation.metrics.basemetrics as metrics
from evaluation.runner import basic_runner as runner


if __name__ == "__main__":
    # Create some synthetic datasets

    N=512

    datasets = [ds.Dataset(name="dataset1", data=np.random.normal(size=[N,N])),
                ds.Dataset(name="dataset2", data=np.random.normal(size=[N,N]))]

    # Create some algorithms
    algorithms = [alg.identity(), alg.gauss_filter(sigma=1.0)]

    # Create a runner and run the evaluation
    my_runner = runner.BasicRunner()
    results = my_runner.run(datasets=datasets, algorithms=algorithms)

    df = pd.DataFrame(results)
    # Print results
    print(df)
