# Titanic

## Description
This is my approach to the "Titanic: Machine Learning from Disaster" competition from Kaggle (https://www.kaggle.com/c/titanic). The goal of the competition is to predict which passengers survived the shipwreck.

## Given Data
Variable ¦ Definition ¦ Key
-------- ¦ ---------- ¦ ---
survival ¦ Survival ¦ 0 = No, 1 = Yes
pclass ¦ Ticket class ¦ 1 = 1st, 2 = 2nd, 3 = 3rd
sex ¦ Sex ¦ 
Age ¦ Age in years ¦ 
sibsp ¦ # of siblings / spouses aboard the Titanic ¦ 
parch ¦ # of parents / children aboard the Titanic ¦ 
ticket ¦ Ticket number ¦ 
fare ¦ Passenger fare ¦ 
cabin ¦ Cabin number ¦ 
embarked ¦ Port of Embarkation ¦ C = Cherbourg, Q = Queenstown, S = Southampton

Training set contains 891 values (I use 889 of them, because 2 are missing a value for embarked) and test set contains 491.

Using insights from Tableau worksheets and chi-squared tests, I reduced the given data to 16 features and then tried different models for submissions.

## DropNN
The file DropNN uses a 3-layered Neural Network with dropout to predict survival and grid search, which achieves around 77% accuracy.

## Regs
The file Regs can use different regressors/classifiers from the sklearn library, such as the RandomForestRegressor, SVR and SVM models.

## GradientBoosting
The file GradientBoosting uses the Gradient Boosting classifier from sklearn and grid search.

## Ideas for further work
- Use one main file where you can choose what to routine to run
- Implement grid search for RFR_Regs
- Clean code
- Implement different models
- Visualisation of results from grid search
- Save readable results from grid search for DropNN