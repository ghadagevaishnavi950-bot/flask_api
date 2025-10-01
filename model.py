import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train model
def train_model(df, model_path):
    X = df.drop("price", axis=1)   # Features
    y = df["price"]                # Target
    model = LinearRegression()
    model.fit(X, y)
    pickle.dump(model, open(model_path, "wb"))


# Test model
def test_model(df, model_path):
    model = pickle.load(open(model_path, "rb"))
    X = df.drop("price", axis=1)
    y = df["price"]
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    return {"mse": mse}


# Predict single JSON input
def predict_input(data, model_path):
    model = pickle.load(open(model_path, "rb"))
    X = [[data['engine'], data['bhp'], data['age']]]
    price = model.predict(X)[0]
    return {"price": round(price, 2), "category": "premium" if price > 1000000 else "budget"}
