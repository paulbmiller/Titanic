# -*- coding: utf-8 -*-
"""
Use of Random Forest Regression to fill missing age values and Gradient
Boosting to predict if a survivor survived.
"""
import numpy as np
import pandas as pd
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
from itertools import product


def submit(n_estimators, lr, feat, depth, X_train, y_train, X_test, filename):
    """
    Creates the Gradient Boosting classifier, fits it to the entire training
    set and writes predictions to the file ´filename´.

    Parameters
    ----------
    n_estimators :
        
    lr : float
        Learning rate.
    feat : int
        Number of max features.
    depth : int
        Max depth.
    X_train : pd.DataFrame
        Training set features.
    y_train : pd.DataFrame
        Training set targets.
    X_test : pd.DataFrame
        Test set features.
    filename : string
        Name of the submission file.

    Returns
    -------
    None.

    """
    gb = GradientBoostingClassifier(n_estimators=n_estimators,
                                    learning_rate=lr,
                                    max_features=feat,
                                    max_depth=depth)
    
    gb.fit(X_train, y_train)
    
    survived_array = gb.predict(X_test)

    test_ids = pd.read_csv('test.csv').PassengerId
    survived = pd.Series(survived_array.astype(int))

    frame = {'PassengerId': test_ids, 'Survived': survived}

    submission_df = pd.DataFrame(frame)

    submission_df.to_csv(
        filename,
        index=False
    )
    
    print('\nPredictions submitted to {}'.format(filename))
    print('Gradient Boosting')
    print('n_estimators={}, lr={}, max_features={}, max_depth={}'.format(
        n_estimators, lr, feat, depth))
    


if __name__ == '__main__':
    # Standard run
    penalty_for_missing_age, X_train_val, y_train_val, X_test = preprocess()

    # Grid search
    # Overrides current results in GridSearch
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=val_size)
    
    estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    lrs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
           0.12, 0.13, 0.14, 0.15, 0.2]
    feats = [2, 3, 5, 7, 10, 15]
    depths = [2, 3, 5, 7, 10]
    
    gboost_results_fn = 'GBoostResults.csv'
    col_names = ['n_estimators', 'lr', 'max_features', 'max_depth',
                 'train acc', 'test acc']
    
    results_arr = None
    
    models = [model_params for model_params in product(estimators, lrs,
                                                       feats, depths)]
    
    for n_estimators, lr, feat, depth in tqdm(models):
        gb = GradientBoostingClassifier(n_estimators=n_estimators,
                                        learning_rate=lr,
                                        max_features=feat,
                                        max_depth=depth)
        
        gb.fit(X_train, y_train)
        
        df = pd.DataFrame()
        
        if results_arr is not None:
            results_arr = np.vstack([results_arr, [n_estimators,
                                                   lr,
                                                   feat,
                                                   depth,
                                                   gb.score(X_train, y_train),
                                                   gb.score(X_val, y_val)]])
        else:
            results_arr = [n_estimators, lr, feat, depth,
                           gb.score(X_train, y_train), gb.score(X_val, y_val)]

    df = pd.DataFrame(data=results_arr, columns=col_names)
    df = df.sort_values(['test acc', 'train acc'], ascending=False)
    df.to_csv(gboost_results_fn)
    print("\nGrid search results stored in file {}".format(
        gboost_results_fn))
    """
    
    # Create prediction submission
    path = 'subs//'
    sub_nb = 58
    
    filename = path + 'sub' + str(sub_nb) + '.csv'
    submit(350, 0.11, 5, 3, X_train_val, y_train_val, X_test, filename)
    
    
