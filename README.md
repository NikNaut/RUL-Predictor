# RUL-Predictor: Lithium-ion Battery Remaining Useful Life Estimation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: NASA PCoE](https://img.shields.io/badge/Dataset-NASA%20PCoE-orange)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
[![Live Demo](https://img.shields.io/badge/-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rul-predictor.streamlit.app/)


Predicts how many charge-discharge cycles remain before a lithium-ion battery reaches end-of-life — using a neural network trained on NASA's battery degradation dataset.

**Relevant domains:** Electric vehicle battery management · Energy storage systems · Predictive maintenance

---

## The Problem

Lithium-ion batteries degrade with every charge-discharge cycle. Once a battery's capacity drops below **80% of its original value**, it is considered to have reached End-of-Life (EOL) — the industry standard threshold used across EV and energy storage applications.

Predicting *when* this threshold will be crossed (the Remaining Useful Life) allows systems to schedule maintenance proactively, avoiding unexpected failures and reducing replacement costs.

---

## Approach
 
Battery capacity degrades non-linearly — it fades slowly at first, then accelerates near EOL. A feedforward neural network captures this curve more accurately than linear regression.
 
**Pipeline:**
1. Load per-cycle discharge capacity data from NASA's B0005, B0006, B0007 battery datasets
2. Normalise cycle index to [0, 1] for stable training
3. Train a 2-layer neural network (32 units, ReLU) to model the capacity degradation curve
4. Project the curve forward to find the cycle where capacity crosses the 80% EOL threshold
5. Compute RUL = EOL cycle − current cycle
> **Note on deployed model scope:** The model served via `api.py` (trained by `train.py`)
> is trained on **B0005 only**. The B0005/B0006/B0007 comparison in the Results section below
> comes from `main.py`'s standalone analysis pipeline and is not reflected in the live API/Streamlit app.
 
---

## Results

### Neural Network Performance

| Battery | Cycles Observed | Initial Capacity (Ah) | MAE (Ah) | RMSE (Ah) | R² | Predicted EOL (cycle) |
|---------|----------------|------------------------|----------|-----------|-----|----------------------|
| B0005   | 168            | 1.8565                 | 0.0187   | 0.0217    | 0.9869 | 105 |
| B0006   | 168            | 2.0353                 | 0.0226   | 0.0302    | 0.9855 | 62  |
| B0007   | 168            | 1.8911                 | 0.0179   | 0.0219    | 0.9814 | 123 |

### LSTM vs NN Comparison

| Battery | NN RMSE (Ah) | LSTM RMSE (Ah) | Winner |
|---------|-------------|----------------|--------|
| B0005   | 0.0217      | 0.0216         | LSTM   |
| B0006   | 0.0302      | 0.0284         | LSTM   |
| B0007   | 0.0219      | 0.0199         | LSTM   |

LSTM outperforms the feedforward neural network on all three batteries, though the margin on B0005 is minimal — the two models are nearly tied there, with LSTM's advantage more pronounced on B0006 and B0007 where the degradation curve exhibits more pronounced non-linear behaviour.

<img width="800" height="400" alt="Figure_4" src="https://github.com/user-attachments/assets/708e4638-f84d-482d-96eb-d7b852347a82" />

---

## Experiment Tracking

All training runs are tracked with MLflow — logging parameters (battery, model type, epochs, learning rate) and metrics (MAE, RMSE, R², predicted EOL, predicted RUL) for full reproducibility across all three batteries.

<img width="1919" height="927" alt="image" src="https://github.com/user-attachments/assets/437dcb25-9a4f-4186-8fa8-8e9019b93bea" />

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Project Structure

```
RUL-Predictor/
├── data/
│   ├── batt_data5.csv      # NASA battery B0005 — discharge capacity per cycle
│   ├── batt_data6.csv      # NASA battery B0006
│   └── batt_data7.csv      # NASA battery B0007
├── models/
│   ├── rul_model.h5        # Pre-trained neural network model
│   └── max_cycle.npy       # Normalisation factor used during training
├── notebooks/
│   └── RUL_Analysis.ipynb  # Exploratory analysis with SoH plots and LSTM comparison
├── main.py                 # Core pipeline: load → train → evaluate → predict → plot
├── train.py                # Trains and saves the model for API use
├── api.py                  # FastAPI app — REST endpoint for RUL prediction
├── app.py                  # Streamlit web app for interactive RUL prediction
├── mlflow.db               # MLflow db to track training runs
├── requirements.txt        # Dependencies
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/NikNaut/RUL-Predictor.git
cd RUL-Predictor
pip install -r requirements.txt

# Run the core pipeline (training, evaluation, plots)
python main.py

# View experiment tracking dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Train and save model for API use
python train.py

# Run the FastAPI server
uvicorn api:app --reload

# Or run the Streamlit web app
streamlit run app.py
```

**Output:**
- Per-battery metrics (MAE, RMSE, R²) printed to console
- State of Health plots — actual vs predicted degradation curve with EOL threshold
- Combined SoH comparison across all three batteries

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python | Core language |
| NumPy / Pandas | Data handling |
| TensorFlow / Keras | Neural network model |
| Scikit-learn | Evaluation metrics (MAE, RMSE, R²) |
| Matplotlib | Degradation curve visualisation |
| FastAPI | REST API for model serving |
| MLflow | Experiment tracking and reproducibility |

---

## Why Neural Network over Linear Regression?

Battery capacity fade follows a non-linear curve — gradual early on, accelerating near end-of-life due to lithium plating, electrolyte decomposition, and SEI layer growth. A simple linear model systematically underestimates late-stage degradation. A small neural net (2 hidden layers, ReLU activation) fits this curve significantly better, as reflected in the lower RMSE scores.

---

## Dataset

NASA Prognostics Center of Excellence (PCoE) Battery Dataset — batteries B0005, B0006, B0007 cycled at room temperature until end-of-life. Each row in the CSV represents one charge-discharge cycle with measured discharge capacity.

[Dataset source](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
