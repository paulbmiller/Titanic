# -*- coding: utf-8 -*-
"""
Random Forest followed by a Neural Network with NN with Dropout.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler

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

# Get the number of cabins and suppose NaN values mean 1 cabin
training_set.Cabin = training_set.Cabin.fillna(1)
training_set.loc[training_set.Cabin != 1, 'Cabin'] = \
    training_set.loc[training_set.Cabin != 1, 'Cabin'].apply(lambda x:
                                                             len(x.split(' ')))
test_set.Cabin = test_set.Cabin.fillna(1)
test_set.loc[test_set.Cabin != 1, 'Cabin'] = \
    test_set.loc[test_set.Cabin != 1, 'Cabin'].apply(lambda x:
                                                     len(x.split(' ')))

# Drop the ticket column
training_set = training_set.drop('Ticket', axis=1)
test_set = test_set.drop('Ticket', axis=1)

# Get the data used for clustering
# Note: Maybe we should include Cabin position to include the probability
# of the person being at that spot (but most of these are not specified)
train_indices_clustering = np.array([2, 4, 5, 6, 7, 9, 10, 11, 12], dtype=int)
train_clust = training_set.iloc[:, train_indices_clustering]
test_clust = test_set.iloc[:, train_indices_clustering -
                           np.ones(len(train_indices_clustering))]

# =============================================================================
# dendrogram = sch.dendrogram(sch.linkage(train_clust, method='ward'))
# plt.title('Dendrogram')
# plt.xlabel('People')
# plt.ylabel('Euclidian distances')
# plt.show()
# =============================================================================

# Scale data
sc = StandardScaler()
training_set = sc.fit_transform(training_set)
test_set = sc.transform(test_set)

# Use Random Forest Regression to replace NaN values for Age using indices
# [2, 4, 6, 7, 9, 10, 11, 12]
indices = np.array([2, 4, 6, 7, 9, 10, 11, 12], dtype=int)
X_train = training_set.iloc[:, indices]
y_train = training_set.iloc[:,]
X_test = training_set.iloc[:, indices]































