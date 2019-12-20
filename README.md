# Titanic

This is my approach to the "Titanic: Machine Learning from Disaster" competition from Kaggle (https://www.kaggle.com/c/titanic). The goal of the competition is to predict which passengers survived the shipwreck.

It uses Random Forest Regression to predict the missing values for the age of some passengers and then uses a feed-forward Neural Network with Dropout layers.

It also extracts the number of cabins for passengers and assumes NaN values represent only one cabin.