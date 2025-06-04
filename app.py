from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        df = pd.DataFrame(data)
        logging.debug(f"DataFrame columns: {df.columns.tolist()}")
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        logging.info("Prediction successful")
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)