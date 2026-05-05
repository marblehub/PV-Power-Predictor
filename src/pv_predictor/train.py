import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    model.fit(X_scaled, y_train)
    return model, scaler, X_scaled


def train_and_log(X_train, y_train):
    with mlflow.start_run():
        model, scaler, X_scaled = train_model(X_train, y_train)

        # ADD METRIC
        y_pred = model.predict(X_scaled)
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)

        mlflow.log_metric("rmse", rmse)

        # (optional but recommended)
        mlflow.log_param("hidden_layers", (64, 32))
        mlflow.log_param("max_iter", 1000)

        mlflow.sklearn.log_model(model, name="model")

        joblib.dump(model, "outputs/models/model.pkl")
        joblib.dump(scaler, "outputs/models/scaler.pkl")

        return model, scaler
