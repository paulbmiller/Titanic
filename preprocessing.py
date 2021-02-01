# -*- coding: utf-8 -*-
"""
Module containing the preprocessing and the function preprocess which returns
the train and test pandas dataframes.
The function encodes the name into a title, encodes categorical data, reduces
the data down to 16 boolean variables.
"""
import pandas as pd

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


def preprocess(title_min_occ=5):
    """
    Preprocessing phase, cleaning and completing data.

    Parameters
    ----------
    title_min_occ : int, optional
        Minimum occurences of a title for it to be stripped from the name.
        The default is 5.

    Returns
    -------
    training_set : pandas DataFrame
        Training set features.
    y_train : pandas Series
        Training set target.
    test_set : pandas DataFrame
        Test set features.

    """
    training_set = pd.read_csv('train.csv')
    test_set = pd.read_csv('test.csv')
    # -------- Preprocessing --------
    # Encode categorical data for Sex
    cleanup_sex = {'Sex': {'male': 0, 'female': 1}}
    training_set.replace(cleanup_sex, inplace=True)
    test_set.replace(cleanup_sex, inplace=True)

    """
    Encode categorical data for Embarked and avoid dummy trap.
    
    We drop the two rows which have no value for embarked, since the chi-
    squared test reveals that the port is statistically significant and
    there are no null values for embarcation in the test set.
    """
    training_set.dropna(subset=['Embarked'], inplace=True)
    training_set.reset_index(drop=True, inplace=True)
    training_set = pd.get_dummies(training_set, columns=['Embarked'])
    test_set = pd.get_dummies(test_set, columns=['Embarked'])
    training_set.drop('Embarked_S', axis=1, inplace=True)
    test_set.drop('Embarked_S', axis=1, inplace=True)


    """
    Since the Chi-Squared test reveals a significant difference for age, I
    decided to encode passengers in 4 age groups :
        - Less than 16
        - Between 16 (inclusive) and 60 (inclusive)
        - More than 60
        - Null values (since they have lower survival rate)
    To avoid the dummy variable trap, we do not include a null age column.
    """
    for ds in [training_set, test_set]:
        ds['Child'] = (ds.Age < 16).astype(int)
        ds['Adult'] = (ds.Age.between(16, 60)).astype(int)
        ds['Senior'] = (ds.Age > 60).astype(int)
    
    training_set.drop('Age', axis=1, inplace=True)
    test_set.drop('Age', axis=1, inplace=True)
    
    """
    Since the Chi-Squared test reveals a significant difference in survival
    rate when the cabin value is null, we just reduce this variable to a bool
    column.
    """
    training_set.Cabin = training_set.Cabin.isna().astype(int)
    test_set.Cabin = test_set.Cabin.isna().astype(int)

    """
    I decided to drop the fare values, because the fare is in great part tied
    to the class which is contained in the Pclass column.
    """
    training_set.drop('Fare', axis=1, inplace=True)
    test_set.drop('Fare', axis=1, inplace=True)

    # Get the training set targets
    y_train = training_set.Survived
    training_set = training_set.drop('Survived', axis=1)

    """
    For names, I decided to just strip the titles from the names and use the
    values appearing at least 5 times.
    
    It would be interesting to see if the differences between these groups are
    statistically significant (TODO).
    """
    # Reduce names to titles
    training_set.Name = \
        training_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

    test_set.Name = \
        test_set.Name.apply(lambda x: x[0: x.find('.')].split(' ')[-1])

    # Replace titles appearing less than 5 times in the training set by Mr/Mrs
    for i in range(len(training_set)):
        title = training_set.Name.iloc[i]
        if training_set.Sex.iloc[i] == 0:
            if training_set.Name.value_counts()[title] < title_min_occ:
                training_set.Name.replace(title, 'Mr', inplace=True)
        else:
            if training_set.Name.value_counts()[title] < title_min_occ:
                training_set.Name.replace(title, 'Mrs', inplace=True)

    set_titles = set(training_set.Name)
    for i in range(len(test_set)):
        title = test_set.loc[i, 'Name']
        if test_set.Sex.iloc[i] == 0:
            if title not in set_titles:
                test_set.Name.replace(title, 'Mr', inplace=True)
        else:
            if title not in set_titles:
                test_set.Name.replace(title, 'Mrs', inplace=True)

    training_set = pd.get_dummies(training_set, columns=['Name'])
    test_set = pd.get_dummies(test_set, columns=['Name'])
    
    # Avoid dummy variable trap
    training_set = training_set.drop('Name_Mr', axis=1)
    test_set = test_set.drop('Name_Mr', axis=1)

    # Encode Pclass into categorical features and avoid dummy variable trap
    training_set = pd.get_dummies(training_set, columns=['Pclass'])
    test_set = pd.get_dummies(test_set, columns=['Pclass'])
    training_set = training_set.drop('Pclass_3', axis=1)
    test_set = test_set.drop('Pclass_3', axis=1)

    # Drop the passenger id and ticket columns
    training_set.drop('PassengerId', axis=1, inplace=True)
    training_set.drop('Ticket', axis=1, inplace=True)
    test_set.drop('PassengerId', axis=1, inplace=True)
    test_set.drop('Ticket', axis=1, inplace=True)

    # Reduce number of siblings/spouses to bool
    training_set['SibSp'] = (training_set['SibSp'] > 0).astype(int)
    test_set['SibSp'] = (test_set['SibSp'] > 0).astype(int)
    
    # Reduce number of parents/children to bool
    training_set['Parch'] = (training_set['Parch'] > 0).astype(int)
    test_set['Parch'] = (test_set['Parch'] > 0).astype(int)

    return training_set, y_train, test_set
