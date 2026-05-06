#def main() -> None:
#    print("Hello from pv-predictor!")

# src/pv_predictor/__init__.py

import argparse
from pv_predictor.predict import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="PV Power Predictor")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--irradiance", type=float, required=True)

    args = parser.parse_args()

    model, scaler = load_model()
    result = predict(model, scaler, args.temperature, args.irradiance)

    print(f"Predicted power: {round(result, 4)}")
