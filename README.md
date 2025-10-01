Car Price Prediction:

Predict the resale price of a car using Machine Learning based on specifications such as engine size, BHP, and age. The project includes a trained regression model and a Flask API for serving predictions.

Project Overview:

Objective: Estimate car prices from features like engine capacity, horsepower, and age.

Approach: Train a regression model using scikit-learn and expose it via Flask REST API.

Use Case: Useful for car dealerships, second-hand car platforms, and buyers for fair price estimation.


Project Structure:
flask-ml-app/
│── app.py              # main flask app
│── model.py            # ML model logic (train, predict)
│── requirements.txt    # dependencies
│── data/
│    └── train.csv
│    └── test.csv
│── saved_model/
│    └── model.pkl

Technologies Used:

Python 3.x
Pandas
Scikit-learn
Flask
