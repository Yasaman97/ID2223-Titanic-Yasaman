# Serverless Machine Learning


Course: ID2223-Scalable Machine Learning and Deep Learning 

Contributor: <a href="https://github.com/Yasaman97">Yasaman Pazhoolideh</a>

## About

This project builds a scalable machine learning system, that predicts the probability of a passenger having survived or not survived the Titanic, using a logistic regression model.

The dataset can be found through this [link](https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv).

The columns, PassengerId, Name, Ticket, and Cabin were dropped, due to lack of a predictive power. Column Sex was modified to show 0 for male and 1 for female. Column Embarked was modified to show 0 for S, 1 for C, and 2 for Q. 

## Implementation

In order to implement the system, first, the titanic-feature-pipeline.py should be run. This creates a Feature Group on Hopsworks, using the titanic dataset. Second, titanic-training-pipeline.py, which reads the data from from the Feature Group previously created, and builds a Feature View on Hopsworks. The dataset is split into 80% and 20% for train and test sets, respectively. A logistic regression model is then used to the survival of a passenger. The model is also saved on Hopsworks. A Gardio application is used to download the previously built model, and create an interactive user interface, where users can choose and enter the values for the 7 feature values the model was trained on. The prediction of the model is then displayed to users. Third, titanic-batch-inference-pipeline.py is run, which runs daily on Modal and predicts the survival of the last randomly generated passenger. A second Gardio application is used to show the prediction for the latest generated passenger, vs. the actual label, the confusion matrix that gets updated over time, and the recent prediction history. Then, titanic-feature-pipeline-daily.py is run, creating a new randomly generated passenger that can be passed on to titanic-batch-inference-pipeline.py. This file uses Modal in order to run every day.

## Hugging Face spaces

The Titanic monitoring space can be found through this[link](https://huggingface.co/spaces/Yasaman/Titanic_Monitoring).

The Titanic Survival analytics space can be found through this[link](https://huggingface.co/spaces/Yasaman/titanic).

The Iris Flower Predictive Analytics analytics space can be found through this[link](https://huggingface.co/spaces/Yasaman/Iris).

The Iris monitoring space can be found through this[link](https://huggingface.co/spaces/Yasaman/Iris_Monitoring).