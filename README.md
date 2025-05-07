# bench-for-donch
This is a NN-benchmark w.r.t. performance created in order to investigate how fast PyTorch-based models are trained on different devices.

`/dataset` folder contains train and test data as well as special `dataset.py`. Use `dataset.py` as:
>
`
from dataset import dataset
data, labels = dataset.load()
`
