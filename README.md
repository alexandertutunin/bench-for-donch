# bench-for-donch
This is a NN-benchmark w.r.t. performance created in order to investigate how fast PyTorch-based models are trained on different devices.

## Dataset
`/dataset` folder contains train and test data as well as special `dataset.py`. Use `dataset.py` as:
>

`from dataset import dataset`
>
`data, labels = dataset.load()`
>

## Benchmarks

`/Benchmark` contains 2 Jupyter Notebooks: `benchmark_gpu.ipynb` and `benchmark_gpu-preloaded.ipynb`. The only difference between these files is `benchmark_gpu-preloaded.ipynb` performs dataset preloading into CUDA device while `benchmark_gpu.ipynb` keeps dataset on CPU and performs batch-wise sending to CUDA. Pay attention what device data are in. 
