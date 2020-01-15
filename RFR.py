# -*- coding: utf-8 -*-
"""
Use of Random Forest Regression to fill missing age values and another RFR to
predict if a survivor survived.
"""
import numpy as np
import pandas as pd
from preprocess import Preprocess
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.svm import SVR


def submit(survived_array, filename):
    # Add a penalty if the age value is missing
    survived_array -= penalty_for_missing_age * 0
    survived_array = survived_array > 0.5

    # Get the passenger ids for the test set for submission
    test_ids = pd.read_csv('test.csv').PassengerId
    survived = pd.Series(survived_array.astype(int))

    frame = {'PassengerId': test_ids, 'Survived': survived}

    submission_df = pd.DataFrame(frame)

    submission_df.to_csv(
        filename,
        index=False
    )
    print('\nPredictions submitted to ' + filename)


def get_next_sub_name(path, sub):
    sub_name = 'sub' + str(sub) + '.csv'
    if os.path.exists(path + sub_name):
        print('{} already exists'.format(sub_name))
        return get_next_sub_name(path, sub + 1)
    else:
        return sub


if __name__ == '__main__':
    # Standard run
    sub = 47
    subs = 1
    path = "F:\\Users\\SilentFart\\Documents\\PythonProjects\\Titanic\\subs\\"
    next_sub_id = get_next_sub_name(path, sub)
    for i in range(subs):
        penalty_for_missing_age, X_train, y_train, X_test = Preprocess()
        reg = RandomForestRegressor(n_estimators=1000)
        reg.fit(X_train, y_train)
        y_test = reg.predict(X_test)
        filename = path + 'sub' + str(next_sub_id + i) + '.csv'
        submit(y_test, filename)
