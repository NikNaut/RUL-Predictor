import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'capacity' not in df.columns:
            raise ValueError("Missing 'capacity' column")
        return df['capacity'].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])

def train_model(cycles, capacity):
    X = cycles.reshape(-1, 1).astype(np.float32)
    y = capacity.astype(np.float32)
    X_norm = X / np.max(X)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X_norm, y, epochs=300, verbose=0)
    return model

def predict_rul(model, capacity, current_cycle_count, max_train_cycle):
    threshold = 0.8 * capacity[0]
    future_cycles = np.arange(1, current_cycle_count + 300).reshape(-1, 1).astype(np.float32)
    future_cycles_norm = future_cycles / max_train_cycle  
    predictions = model.predict(future_cycles_norm, verbose=0).flatten()
    below_threshold = np.where(predictions <= threshold)[0]
    if len(below_threshold) > 0:
        return int(future_cycles[below_threshold[0]][0])
    return 0

def plot_soh_and_prediction(cycles, capacity, model, title):
    soh = (capacity / capacity[0]) * 100
    X = cycles.reshape(-1, 1).astype(np.float32)
    X_norm = X / np.max(X)
    predicted_capacity = model.predict(X_norm, verbose=0).flatten()
    predicted_soh = (predicted_capacity / capacity[0]) * 100

    plt.figure()
    plt.plot(cycles, soh, 'x-', label='Actual SoH', color='teal')
    plt.plot(cycles, predicted_soh, '--', label='NN Trend', color='red')
    plt.xlabel("Cycle")
    plt.ylabel("State of Health (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_combined_soh(cycles_list, capacities_list, labels):
    plt.figure()
    for cycles, cap, label in zip(cycles_list, capacities_list, labels):
        soh = (cap / cap[0]) * 100
        plt.plot(cycles, soh, label=label)
    plt.xlabel("Cycle")
    plt.ylabel("State of Health (%)")
    plt.title("Combined State of Health Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    files = {
        "B0005": "batt_data5.csv",
        "B0006": "batt_data6.csv",
        "B0007": "batt_data7.csv"
    }
    capacities_all = {}
    cycles_all = {}
    models_all = {}
    rul_all = {}
    for label, path in files.items():
        capacity = load_data(path)
        if len(capacity) == 0:
            continue
        cycles = np.arange(1, len(capacity) + 1)
        model = train_model(cycles, capacity)
        rul = predict_rul(model, capacity, len(cycles), np.max(cycles))
        capacities_all[label] = capacity
        cycles_all[label] = cycles
        models_all[label] = model
        rul_all[label] = rul
        plot_soh_and_prediction(cycles, capacity, model, f"{label}: SoH & NN RUL Prediction")
    print("Predicted RUL (cycle where SoH drops to 80%):")
    for label, rul in rul_all.items():
        print(f"{label}: {rul} cycles")
    plot_combined_soh(
        [cycles_all[k] for k in capacities_all],
        [capacities_all[k] for k in capacities_all],
        list(capacities_all.keys())
    )
    plt.show()
