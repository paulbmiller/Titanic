# -*- coding: utf-8 -*-
"""
Small module to get the differences between two submission CSV files.
"""
import pandas as pd


def sub_diff(filename1, filename2):
    submission1 = pd.read_csv(filename1)
    submission2 = pd.read_csv(filename2)

    return (submission1.Survived != submission2.Survived).sum()
