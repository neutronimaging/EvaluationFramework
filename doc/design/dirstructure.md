```
benchmark/
│
├── src/                         # all framework source code
│   └── benchmark/
│       ├── __init__.py
│
│       ├── core/               # framework abstractions (stable layer)
│       │   ├── dataset.py
│       │   ├── sample.py
│       │   ├── algorithm.py
│       │   ├── metric.py
│       │   ├── experiment.py
│       │   ├── result.py
│       │   └── registry.py     # plugin/registration system
│
│       ├── datasets/           # dataset implementations
│       │   ├── base.py
│       │   ├── image_folder.py
│       │   ├── hdf5_dataset.py
│       │   └── neutron_*.py    # your domain datasets
│
│       ├── algorithms/         # algorithm implementations
│       │   ├── base.py
│       │   ├── filtering/
│       │   │   ├── median.py
│       │   │   ├── gaussian.py
│       │   │   └── bilateral.py
│       │   ├── denoising/
│       │   │   ├── nlm.py
│       │   │   ├── bm3d.py
│       │   ├── segmentation/
│       │   │   ├── threshold.py
│       │   │   ├── watershed.py
│       │   └── reconstruction/
│       │       ├── fbp.py
│       │       ├── sirt.py
│       │       └── cgls.py
│
│       ├── metrics/            # evaluation metrics
│       │   ├── base.py
│       │   ├── reference/
│       │   │   ├── psnr.py
│       │   │   ├── ssim.py
│       │   │   └── rmse.py
│       │   ├── no_reference/
│       │   │   ├── entropy.py
│       │   │   ├── sharpness.py
│       │   │   └── noise.py
│       │   ├── task/
│       │   │   ├── dice.py
│       │   │   └── iou.py
│       │   └── custom/
│       │       ├── edge_fidelity.py
│       │       ├── outlier_fraction.py
│       │       └── streak_metric.py
│
│       ├── params/             # parameter space definitions
│       │   ├── grid.py
│       │   ├── sampling.py     # random / latin hypercube
│       │   └── definitions/
│       │       ├── median.yaml
│       │       ├── nlm.yaml
│       │       └── sirt.yaml
│
│       ├── execution/          # execution engine
│       │   ├── runner.py
│       │   ├── parallel.py
│       │   ├── scheduler.py
│       │   ├── caching.py
│       │   └── failure_handling.py
│
│       ├── io/                 # I/O utilities
│       │   ├── results_writer.py
│       │   ├── results_reader.py
│       │   ├── image_io.py
│       │   └── serialization.py
│
│       ├── tracking/           # optional integration (MLflow, etc.)
│       │   ├── mlflow_logger.py
│       │   └── local_logger.py
│
│       └── utils/              # small helpers
│           ├── timing.py
│           ├── hashing.py
│           └── config.py
│
├── configs/                    # experiment definitions (user-facing)
│   ├── benchmark.yaml
│   ├── datasets/
│   ├── algorithms/
│   └── studies/
│       ├── denoising_snr.yaml
│       ├── segmentation.yaml
│       └── reconstruction.yaml
│
├── data/                       # NOT version-controlled (or via DVC)
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── results/                    # generated outputs
│   ├── raw/
│   │   └── results.parquet
│   ├── aggregated/
│   └── figures/
│
├── analysis/                   # notebooks / scripts (no core logic!)
│   ├── summary.ipynb
│   ├── parameter_sensitivity.ipynb
│   └── plots.py
│
├── tests/                      # unit tests (important!)
│   ├── test_algorithms.py
│   ├── test_metrics.py
│   ├── test_runner.py
│   └── test_reproducibility.py
│
├── scripts/                    # CLI entry points
│   ├── run_benchmark.py
│   ├── analyze_results.py
│   └── export_tables.py
│
├── pyproject.toml
└── README.md
```
