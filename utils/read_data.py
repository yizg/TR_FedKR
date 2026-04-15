import numpy as np
import pandas as pd


def read_s_dataset(s):
    X = pd.read_csv(
        f'data/s-originals/s{s}.txt', header=None, sep='   ', engine='python')
    y = pd.read_csv(
        f'data/s-originals/s{s}-label.pa', header=None, sep='   ', skiprows=5, engine='python')[0]
    return X.values, y.values


def read_letter_dataset():
    X = pd.read_csv(f'data/letter/letter.txt', header=None, sep=' ', )
    y = pd.read_csv('data/letter/letter.pa', header=None,
                    sep='   ', skiprows=5, engine='python')[0]
    return X.values, y.values
