import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import evaluation.algorithms.basic_algorithms as alg
# from evaluation.core.dataset import DataSet as ds
import evaluation.metrics.basemetrics as metrics
from evaluation.runner import basic_runner as runner
import evaluation.datasets.constant_level_data as cld 
import evaluation.datasets.image_loader as img_loader  

if __name__ == "__main__":
    # Create some synthetic datasets

    N=100

    # gen = cld.ConstantLevelData(dims=[N,N])
    #     # datasets = [ds.Dataset(name="dataset1", data=np.random.normal(size=[N,N])),
    #     #             ds.Dataset(name="dataset2", data=np.random.normal(size=[N,N]))]
    # datasets = gen.generate(batch_size=5, noise_level=0.1, params={"level": 0.5, "seed": 42})

    gen = img_loader.ImageLoader(dims=[200,200])
    # gen = img_loader.ImageLoader(dims=None)
    datasets = gen.generate(batch_size=5, noise_level=0.1, params={"image_paths": ["../../data/mixture12_00001.fits"]})
    # Create some algorithms
    algorithms = [alg.identity(), alg.gauss_filter(sigma=1.0)]

    # Create a runner and run the evaluation
    my_runner = runner.BasicRunner()
    results = my_runner.run(datasets=datasets, algorithms=algorithms, save_images=True)

    df = pd.DataFrame(results)
    # Print results
    print(df)
