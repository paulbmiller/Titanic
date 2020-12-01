# Titanic

## Description
This is my approach to the "Titanic: Machine Learning from Disaster" competition from Kaggle (https://www.kaggle.com/c/titanic). The goal of the competition is to predict which passengers survived the shipwreck.

It uses Random Forest Regression to predict the missing values for the age of some passengers.

It also extracts the number of cabins for passengers, assumes NaN values represent only one cabin, uses Qcut to bin age and fare data into categorical variables.

## RFR_DropNN
The file RFR_DropNN uses a 3-layered Neural Network with dropout to predict survival.
It achieves around 78% accuracy.

## RFR_Regs
The file RFR_Regs can use different regressors/classifiers from the sklearn library, such as the RandomForestRegressor, SVR and SVM models.
Random Forest achieves around 75.6% accuracy.
SVR achieves around 77.5% accuracy.
SVM achieves around 77.5% accuracy.
SVC achieves around 78.2% accuracy.

## RFR_GradientBoosting
The file RFR_GradientBoosting uses the Gradient Boosting classifier from sklearn and a grid search implementation.
It achieves around 75.5% accuracy.

## Further work
- Try different methods for filling in missing values
- Implement grid search for RFR_Regs
