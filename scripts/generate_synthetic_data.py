import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(0)

# Parameters
N = 500
temperature = np.random.uniform(15, 45, N)
irradiance = np.random.uniform(200, 1000, N)

# Power with noise
power = 0.05 * irradiance - 0.1 * temperature + np.random.normal(0, 10, N)

# Combine features
X = np.vstack((temperature, irradiance)).T
y = power.reshape(-1, 1)

# Create directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Save to CSV (temperature, irradiance, power)
data = np.column_stack((temperature, irradiance, power))
np.savetxt("data/raw/pv_data.csv", data, delimiter=",", 
           header="temperature,irradiance,power", comments="")

print("Data saved to data/raw/pv_data.csv")
