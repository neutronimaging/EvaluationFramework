# Introduction 

The evaluation framework is a designed as a pipeline processor with segments for each evaluation task. 

Each evaluation segment can contain several task modules which are provided during the pipeline setup. As shown in the figure below.

![The evaluation processing pipeline.](../figures/framework_overview.svg)

The pipeline execution is implemented as a runner class which takes lists with data generators, algorithms, metrics as input. The runner loops over algorithms for all data patches and measures the output for each algorithm/data combination using the metrics provided in the metrics list. 

The [basic runners class](https://github.com/neutronimaging/EvaluationFramework/blob/main/src/evaluation/runner/basic_runner.py) is a first implementation of a pipeline runner. More detailed runners will come.

A figure should come here to show the flow of the runner.

[Return to the main user documentation page](index)