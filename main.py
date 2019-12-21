# -*- coding: utf-8 -*-
"""
Random Forest followed by a Neural Network with NN with Dropout.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# -------- Hyperparameters --------
nb_clusters = 20

# -------- Preprocessing --------
# Encode categorical data for Sex
cleanup_sex = {'Sex': {'male': 0, 'female': 1}}
training_set.replace(cleanup_sex, inplace=True)
test_set.replace(cleanup_sex, inplace=True)

# Encode categorical data for Embarked and avoid dummy trap
training_set = pd.get_dummies(training_set, columns=['Embarked'])
test_set = pd.get_dummies(test_set, columns=['Embarked'])
training_set = training_set.drop('Embarked_S', axis=1)
test_set = test_set.drop('Embarked_S', axis=1)
y_train = training_set.Survived
training_set = training_set.drop('Survived', axis=1)

# Reduce names to titles
training_set.Name = \
    training_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

test_set.Name = \
    test_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

# Replace titles which appear less than 5 times in the training set by Mr/Mrs
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


# Get the number of cabins and suppose NaN values mean 1 cabin
training_set.Cabin = training_set.Cabin.fillna(1)
training_set.loc[training_set.Cabin != 1, 'Cabin'] = \
    training_set.loc[training_set.Cabin != 1, 'Cabin'].apply(lambda x:
                                                             len(x.split(' ')))
test_set.Cabin = test_set.Cabin.fillna(1)
test_set.loc[test_set.Cabin != 1, 'Cabin'] = \
    test_set.loc[test_set.Cabin != 1, 'Cabin'].apply(lambda x:
                                                     len(x.split(' ')))
        
# Drop the passenger id and ticket columns
training_set = training_set.drop('PassengerId', axis=1)
training_set = training_set.drop('Ticket', axis=1)
test_set = test_set.drop('PassengerId', axis=1)
test_set = test_set.drop('Ticket', axis=1)

# Use Random Forest Regression to predict the missing ages of passengers
reg = RandomForestRegressor(n_estimators=1000)
reg.fit(training_set[training_set.Age.notna()].loc[:, training_set.columns != 'Age'],
        training_set.Age[training_set.Age.notna()])
training_set.loc[training_set.Age.isna(), 'Age'] = \
    reg.predict(training_set[training_set.Age.isna()].loc[:, training_set.columns != 'Age'])
test_set.loc[test_set.Age.isna(), 'Age'] = \
    reg.predict(test_set[test_set.Age.isna()].loc[:, test_set.columns != 'Age'])

# Scale data
sc = StandardScaler()
training_set.loc[:, ['Age', 'Fare']] = sc.fit_transform(training_set.loc[:, ['Age', 'Fare']])
test_set.loc[:, ['Age', 'Fare']] = sc.transform(test_set.loc[:, ['Age', 'Fare']])

































