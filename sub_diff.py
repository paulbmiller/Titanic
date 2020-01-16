# -*- coding: utf-8 -*-
"""
Small module to get the differences between two submission CSV files.
"""
import pandas as pd


def sub_diff(filename1, filename2):
    """
    Returns the difference between two submission files.

    Parameters
    ----------
    filename1 : string
        Path to file 1.
    filename2 : string
        Path to file 2.

    Returns
    -------
    int
        Sum of differences between the bool output predictions.

    """
    submission1 = pd.read_csv(filename1)
    submission2 = pd.read_csv(filename2)

    return (submission1.Survived != submission2.Survived).sum()
