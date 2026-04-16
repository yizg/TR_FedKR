import numpy as np
import pandas as pd
from functools import partial


def read_s_dataset(s):
    X = pd.read_csv(
        f'data/s-originals/s{s}.txt', header=None, sep='   ', engine='python', dtype=np.float32)
    y = pd.read_csv(
        f'data/s-originals/s{s}-label.pa', header=None, sep='   ', skiprows=5, engine='python', dtype=np.int32)[0]
    return X.values, y.values


def read_wine_dataset():
    data = pd.read_csv(
        'data/wine+quality/winequality-white.csv', sep=';',  dtype=np.float32)
    X = data.drop('quality', axis=1)
    y = data['quality'].astype(np.int32)
    return X.values, y.values


def read_yeast_dataset():
    data = pd.read_csv('data/yeast/yeast.data', sep="\s+",
                       header=None)
    X = data.drop([0, 9], axis=1).astype(np.float32)
    y = pd.factorize(data[9])[0].astype(np.int32)
    return X.values, y


def read_statlog_dataset():
    data = pd.read_csv(
        'data/statlog+landsat+satellite/sat.trn', sep=' ', header=None)
    y = data[36].astype(np.int32)
    X = data.drop(36, axis=1).astype(np.float32)
    return X.values, y.values


def read_letter_dataset():
    X = pd.read_csv(f'data/letter/letter.txt',
                    header=None, sep=' ', dtype=np.float32)
    y = pd.read_csv('data/letter/letter.pa', header=None, sep='   ',
                    skiprows=5, engine='python', dtype=np.int32)[0]
    return X.values, y.values


def read_mnist_dataset():
    data = pd.read_csv('data/mnist_train.csv', header=None)
    X = data.drop(0, axis=1).astype(np.float32)
    y = data[0].astype(np.int32)
    return X.values, y.values


dataset_loader_map = {
    "S1": partial(read_s_dataset, 1),
    "S2": partial(read_s_dataset, 2),
    "S3": partial(read_s_dataset, 3),
    "S4": partial(read_s_dataset, 4),
    "statlog": read_statlog_dataset,
    "letter": read_letter_dataset,
    "wine": read_wine_dataset,
    "yeast": read_yeast_dataset,
    "mnist": read_mnist_dataset,
}


def read_dataset(name):
    """Load and return (X, y) for the given dataset name."""
    if name not in dataset_loader_map:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(dataset_loader_map.keys())}")
    loader = dataset_loader_map[name]
    return loader()
