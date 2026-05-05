from flask import Flask, render_template, request
from pv_predictor.predict import load_model, predict

app = Flask(__name__)

model, scaler = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    temperature = None
    irradiance = None

    if request.method == "POST":
        temperature = float(request.form["temperature"])
        irradiance = float(request.form["irradiance"])

        prediction = predict(model, scaler, temperature, irradiance)

    return render_template(
        "index.html",
        prediction=prediction,
        temperature=temperature,
        irradiance=irradiance,
        model_name="MLPRegressor"
    )


if __name__ == "__main__":
    #app.run(debug=True, port=5001) # for local server
    app.run(host="0.0.0.0", port=10000) # for production server
