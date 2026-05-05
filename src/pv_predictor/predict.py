import joblib
import numpy as np

def load_model():
    model = joblib.load("outputs/models/model.pkl")
    scaler = joblib.load("outputs/models/scaler.pkl")
    return model, scaler


def predict(model, scaler, temperature, irradiance):
    X = np.array([[temperature, irradiance]])
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]
