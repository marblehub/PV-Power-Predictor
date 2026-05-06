import joblib
import pandas as pd


FEATURE_COLUMNS = ["temperature", "irradiance"]


def load_model():
    model = joblib.load("outputs/models/model.pkl")
    scaler = joblib.load("outputs/models/scaler.pkl")
    return model, scaler


def predict(model, scaler, temperature, irradiance):
    X = pd.DataFrame(
        [[temperature, irradiance]],
        columns=FEATURE_COLUMNS,
    )

    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    return round(float(prediction), 4)
