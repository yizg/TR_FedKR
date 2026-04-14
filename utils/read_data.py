import numpy as np
import pandas as pd


def read_s_dataset(s):
    X = pd.read_csv(f'data/s-originals/s{s}.txt', header=None, sep='   ')
    y = pd.read_csv(
        f'data/s-originals/s{s}-label.pa', header=None, sep='   ', skiprows=5)[0]
    return X.values, y.values
