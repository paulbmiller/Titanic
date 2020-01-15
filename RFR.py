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
        return path + sub_name


if __name__ == '__main__':
    """
    # Grid Search
    accs = []
    epochs_list = [10]
    lr_list = [3e-3]
    mb_sizes = [32]
    optims = ['Adam', 'RMSprop']
    nets = [
            [128, 64, 32, 0, 0.2, 0.2],
            [128, 64, 32, 0, 0.1, 0.1],
            [128, 64, 32, 0, 0, 0]
            ]
    for mb_size in mb_sizes:
        train_loader, test_loader, penalty_for_missing_age, X_train,\
            y_train = Preprocess(mb_size)
        for lr in lr_list:
            for epochs in epochs_list:
                for opt in optims:
                    for net in nets:
                        net_obj = Net(in_channels, net[0], net[1], net[2],
                                      net[3], net[4], net[5]).to(device)
                        test(net_obj, epochs, lr, mb_size, accs, opt, net)
    accs.sort(key=sort_accs, reverse=True)
    """
    # Standard run
    sub = 42
    subs = 1
    path = "F:\\Users\\SilentFart\\Documents\\PythonProjects\\Titanic\\subs\\"
    for i in range(subs):
        penalty_for_missing_age, X_train, y_train, X_test = Preprocess()
        reg = RandomForestRegressor(n_estimators=10000)
        reg.fit(X_train, y_train)
        y_test = reg.predict(X_test)
        filename = get_next_sub_name(path, sub)
        submit(y_test, filename)
