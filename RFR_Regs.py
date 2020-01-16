# -*- coding: utf-8 -*-
"""
Use of Random Forest Regression to fill missing age values and another RFR to
predict if a survivor survived.
"""
import pandas as pd
from preprocessing import preprocess
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC


def submit(survived_array, filename):
    """
    Returns the prediction array to the output file with ´filename´.

    Parameters
    ----------
    survived_array : numpy.ndarray
        Prediction array of values between 0 and 1.
    filename : string
        Name of the submission file.

    Returns
    -------
    None.

    """
    # Add a penalty if the age value is missing
    # survived_array -= penalty_for_missing_age * 0.0
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
    print('Predictions submitted to ' + filename)


def get_next_sub_name(path, sub):
    """
    Checks to see if the submission ´sub´ already exists in the submission
    folder. If it already exists, it will call itself with ´sub + 1´ until it
    finds a number which hasn't already been created. In the other case, it
    will return the number ´sub´.

    Parameters
    ----------
    path : string
        Path to the file submission folder.
    sub : int
        Number of the submission we are checking.

    Returns
    -------
    int
        Number of the next submission.

    """
    sub_name = 'sub' + str(sub) + '.csv'
    if os.path.exists(path + sub_name):
        # print('{} already exists'.format(sub_name))
        return get_next_sub_name(path, sub + 1)
    else:
        return sub


if __name__ == '__main__':
    # Standard run
    sub = 56
    subs = 1
    path = "F:\\Users\\SilentFart\\Documents\\PythonProjects\\Titanic\\subs\\"
    next_sub_id = get_next_sub_name(path, sub)
    for i in range(subs):
        penalty_for_missing_age, X_train, y_train, X_test = preprocess()
        # reg = RandomForestRegressor(n_estimators=10000)
        # reg = SVR(kernel='rbf')
        reg = SVC(kernel='rbf')
        reg.fit(X_train, y_train)
        y_test = reg.predict(X_test)
        filename = path + 'sub' + str(next_sub_id + i) + '.csv'
        submit(y_test, filename)
