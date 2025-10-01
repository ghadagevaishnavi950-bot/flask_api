from flask import Flask, request, jsonify
import pandas as pd
import pickle
from model import train_model, test_model, predict_input

app = Flask(__name__)

MODEL_PATH = "saved_model/model.pkl"

# ----------- API Endpoints ------------ #

# Train endpoint
@app.route("/train", methods=["POST"])
def train():
    file = request.files['file']  # CSV file input
    df = pd.read_csv(file)
    train_model(df, MODEL_PATH)
    return jsonify({"status": "ok", "message": "Model trained successfully"})


# Test endpoint
@app.route("/tst", methods=["POST"])
def test():
    file = request.files['file']
    df = pd.read_csv(file)
    results = test_model(df, MODEL_PATH)
    return jsonify({"status": "ok", "results": results})


# Predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()   # JSON input
    prediction = predict_input(data, MODEL_PATH)
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
