import pandas as pd
from pv_predictor.train import train_and_log

df = pd.read_csv("data/raw/pv_data.csv")

X = df[["temperature", "irradiance"]]
y = df["power"]

train_and_log(X, y)
