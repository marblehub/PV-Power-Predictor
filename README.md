# ☀️ PV Power Output Predictor

A Flask-based web app that uses a trained artificial neural network (ANN) to predict the power output of a photovoltaic (PV) system using temperature and irradiance as inputs.

![screenshot](static/logo.png)

## 🚀 Features

- Web-based interface (HTML + CSS)
- Toggle dark mode 🌙
- Trained using PyTorch
- Easily deployable to Render
- Mobile responsive

---

## 🔧 Technologies

- Python (Flask, PyTorch, NumPy, scikit-learn)
- HTML/CSS (Quicksand font, dark/light mode)
- Deployable via Render

---
```
## 📁 Folder Structure
<pre> pv_predictor/
├── app.py
├── model.py
├── train.py
├── pv_model.pth
├── scaler.pkl
├── requirements.txt
├── Procfile
├── static/
│ ├── style.css
│ ├── logo.png
│ └── favicon.ico
└── templates/
└── index.html
```

---

## 🧠 How It Works

1. The model takes two inputs: temperature (°C) and irradiance (W/m²)
2. Inputs are scaled using `StandardScaler`
3. A trained PyTorch ANN (`PVNet`) predicts power output in kW

---

## 🧪 Local Setup

```bash
# Clone this repo
git clone https://github.com/yourusername/pv-predictor.git
cd pv-predictor

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

or Using conda:
conda create -n myenv
conda activate myenv

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
## Happy Viewing!!!
