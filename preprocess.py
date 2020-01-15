# -*- coding: utf-8 -*-
"""
Module containing the preprocessing and the function Preprocess which returns
the train and test loaders.
The function encodes the name into a title, encodes categorical data, replaces
cabin names by the number of cabins for each passenger.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# * Survival - Survival (0=No, 1=Yes)
# * Pclass - Ticket class
#            - A proxy for socio-economic status (SES)
#            - (1=1st=Upper, 2=2nd=Middle, 3=3rd=Lower)
# * Sex - Sex (Male of Female)
# * Age - Age in years
#      - Age is fractional if less than 1.
#      - If the age is estimated, is it in the form of xx.5
# * Sibsp - of siblings / spouses aboard the Titanic
#      - Sibling = brother, sister, stepbrother, stepsister
#      - Spouse = husband, wife (mistresses and fiancés were ignored)
# * Parch - of parents / children aboard the Titanic
#      - Parent = mother, father
#      - Child = daughter, son, stepdaughter, stepson
#      - Some children travelled only with a nanny, therefore parch=0 for them.
# * Ticket - Ticket number
# * Fare - Passenger fare
# * Cabin - Cabin number
# * Embarked - Port of Embarkation
#              - C = Cherbourg,
#              - Q = Queenstown,
#              - S = Southampton


def Preprocess():
    """

    Parameters
    ----------
    mb_size : int
        Minibatch size for the dataloaders.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set.

    """
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    # -------- Preprocessing --------
    # Encode categorical data for Sex
    cleanup_sex = {'Sex': {'male': 0, 'female': 1}}
    training_set.replace(cleanup_sex, inplace=True)
    test_set.replace(cleanup_sex, inplace=True)

    # Encode categorical data for Embarked and avoid dummy trap
    # Replaces the two missing values by the most common one ('S')
    training_set = pd.get_dummies(training_set, columns=['Embarked'])
    test_set = pd.get_dummies(test_set, columns=['Embarked'])
    training_set = training_set.drop('Embarked_S', axis=1)
    test_set = test_set.drop('Embarked_S', axis=1)

    # Get the people that have missing values for age in the test set and store
    # the mean difference in probability to survive solely based on missing
    # age (people with age missing are less likely to have survived by about
    # 11%). This value will be substracted at test time
    mean_na_survived = training_set.Survived[training_set.Age.isna()].mean()
    mean_notna_survived = \
        training_set.Survived[training_set.Age.notna()].mean()
    bias_na = mean_notna_survived - mean_na_survived
    nan_ages = test_set.Age.isna()
    bias_na = nan_ages * bias_na

    # Get the number of cabins and suppose NaN values mean 1 cabin
    training_set.Cabin = training_set.Cabin.fillna(1)
    training_set.loc[training_set.Cabin != 1, 'Cabin'] = \
        training_set.loc[
            training_set.Cabin != 1, 'Cabin'
            ].apply(lambda x: len(x.split(' ')))
    test_set.Cabin = test_set.Cabin.fillna(1)
    test_set.loc[test_set.Cabin != 1, 'Cabin'] = \
        test_set.loc[test_set.Cabin != 1, 'Cabin'].apply(lambda x:
                                                         len(x.split(' ')))

    # Replace the fare nan values by the mean fare of 3rd class (which is
    # sufficient because there is only one from 3rd class) and replace the 0
    # values by the mean of values of that class times the number of cabins
    mean_fare1 = training_set.Fare[
        training_set.Pclass == 1
        ][training_set.Cabin == 1].mean()
    mean_fare2 = training_set.Fare[
        training_set.Pclass == 2
        ][training_set.Cabin == 1].mean()
    mean_fare3 = training_set.Fare[
        training_set.Pclass == 3
        ][training_set.Cabin == 1].mean()
    training_set.Fare.fillna(0, inplace=True)
    test_set.Fare.fillna(0, inplace=True)
    train0 = training_set.index[training_set.loc[:, 'Fare'] == 0].tolist()
    test0 = test_set.index[test_set.loc[:, 'Fare'] == 0].tolist()
    for i in train0:
        if training_set.loc[i, 'Pclass'] == 1:
            training_set.loc[i, 'Fare'] = \
                mean_fare1 * training_set.loc[i, 'Cabin']
        elif training_set.loc[i, 'Pclass'] == 2:
            training_set.loc[i, 'Fare'] = \
                mean_fare2 * training_set.loc[i, 'Cabin']
        else:
            training_set.loc[i, 'Fare'] = \
                mean_fare3 * training_set.loc[i, 'Cabin']
    for i in test0:
        if test_set.loc[i, 'Pclass'] == 1:
            test_set.loc[i, 'Fare'] = \
                mean_fare1 * test_set.loc[i, 'Cabin']
        elif test_set.loc[i, 'Pclass'] == 2:
            test_set.loc[i, 'Fare'] = \
                mean_fare2 * test_set.loc[i, 'Cabin']
        else:
            test_set.loc[i, 'Fare'] = \
                mean_fare3 * test_set.loc[i, 'Cabin']
    training_set.loc[:, 'Fare'], bins = pd.qcut(training_set.Fare, 5,
                                                labels=False, retbins=True)
    bins[0] = 0.
    bins[5] = 1000.
    test_set.loc[:, 'Fare'] = pd.cut(test_set.Fare, bins=bins,
                                     labels=False)
    test_set.Fare = test_set.Fare.astype(int)

    training_set = pd.get_dummies(training_set, columns=['Fare'])
    test_set = pd.get_dummies(test_set, columns=['Fare'])
    training_set = training_set.drop('Fare_4', axis=1)
    test_set = test_set.drop('Fare_4', axis=1)

    # Get the training set targets
    y_train = training_set.Survived
    training_set = training_set.drop('Survived', axis=1)

    # Reduce names to titles
    training_set.Name = \
        training_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

    test_set.Name = \
        test_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

    # Replace titles appearing less than 5 times in the training set by Mr/Mrs
    for i in range(len(training_set)):
        if training_set.loc[i, 'Sex'] == 0:
            title = training_set.Name.loc[i]
            try:
                if training_set.Name.value_counts()[title] <= 5:
                    training_set.loc[i, 'Name'] = 'Mr'
            except KeyError:
                training_set.loc[i, 'Name'] = 'Mrs'
        else:
            title = training_set.Name.loc[i]
            try:
                if training_set.Name.value_counts()[title] <= 5:
                    training_set.loc[i, 'Name'] = 'Mrs'
            except KeyError:
                training_set.loc[i, 'Name'] = 'Mrs'

    set_titles = set(training_set.Name)
    for i in range(len(test_set)):
        if test_set.loc[i, 'Sex'] == 0:
            if test_set.loc[i, 'Name'] not in set_titles:
                test_set.loc[i, 'Name'] = 'Mr'
        else:
            if test_set.loc[i, 'Name'] not in set_titles:
                test_set.loc[i, 'Name'] = 'Mrs'

    training_set = pd.get_dummies(training_set, columns=['Name'])
    test_set = pd.get_dummies(test_set, columns=['Name'])
    training_set = training_set.drop('Name_Mr', axis=1)
    test_set = test_set.drop('Name_Mr', axis=1)

    # Encode Pclass into categorical features and avoid dummy variable trap
    training_set = pd.get_dummies(training_set, columns=['Pclass'])
    test_set = pd.get_dummies(test_set, columns=['Pclass'])
    training_set = training_set.drop('Pclass_3', axis=1)
    test_set = test_set.drop('Pclass_3', axis=1)

    # Drop the passenger id and ticket columns
    training_set = training_set.drop('PassengerId', axis=1)
    training_set = training_set.drop('Ticket', axis=1)
    test_set = test_set.drop('PassengerId', axis=1)
    test_set = test_set.drop('Ticket', axis=1)

    # Use Random Forest Regression to predict the missing ages of passengers
    reg = RandomForestRegressor(n_estimators=10000)
    reg.fit(training_set[
        training_set.Age.notna()
        ].loc[:, training_set.columns != 'Age'],
            training_set.Age[training_set.Age.notna()])
    training_set.loc[training_set.Age.isna(), 'Age'] = \
        reg.predict(
            training_set[
                training_set.Age.isna()
                ].loc[:, training_set.columns != 'Age']
            )
    test_set.loc[test_set.Age.isna(), 'Age'] = \
        reg.predict(test_set[
            test_set.Age.isna()
            ].loc[:, test_set.columns != 'Age'])

    # Scale Age and Fare columns
    training_set.loc[:, 'Age'], bins = pd.qcut(training_set.Age, 10,
                                               labels=False, retbins=True)
    bins[0] = 0.
    bins[10] = 150.0
    test_set.loc[:, 'Age'] = pd.cut(test_set.Age, bins=bins, labels=False)
    training_set.fillna(10, inplace=True)
    test_set.fillna(10, inplace=True)

    return bias_na, training_set, y_train, test_set
