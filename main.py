import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath):
    """
    Load battery capacity data from a CSV file.
    Expects a 'capacity' column representing discharge capacity per cycle (in Ah).
    """
    try:
        df = pd.read_csv(filepath)
        if 'capacity' not in df.columns:
            raise ValueError("Missing 'capacity' column")
        return df['capacity'].values
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])


# ── Model Training ────────────────────────────────────────────────────────────

def train_model(cycles, capacity):
    """
    Train a small feedforward neural network to model battery capacity degradation.

    Why a neural network over linear regression?
    Battery degradation is non-linear — capacity fades slowly at first, then
    accelerates near end-of-life. A simple neural net captures this curve
    better than a straight line.

    Inputs are normalised to [0, 1] to stabilise training.

    Returns:
        model       -- trained Keras model
        max_cycle   -- normalisation factor (used consistently in prediction)
    """
    max_cycle = float(np.max(cycles))
    X = (cycles / max_cycle).reshape(-1, 1).astype(np.float32)
    y = capacity.astype(np.float32)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X, y, epochs=300, verbose=0)

    return model, max_cycle


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, cycles, capacity, max_cycle):
    """
    Compute and print MAE, RMSE, and R² on the training data.

    These metrics tell us how well the model fits the observed degradation curve.
    Lower MAE/RMSE and higher R² indicate a better fit.
    """
    X = (cycles / max_cycle).reshape(-1, 1).astype(np.float32)
    predicted = model.predict(X, verbose=0).flatten()

    mae = mean_absolute_error(capacity, predicted)
    rmse = np.sqrt(mean_squared_error(capacity, predicted))
    r2 = r2_score(capacity, predicted)

    return mae, rmse, r2, predicted


# ── RUL Prediction ────────────────────────────────────────────────────────────

def predict_rul(model, capacity, current_cycle_count, max_cycle):
    """
    Predict the cycle number at which battery capacity drops to 80% of initial.

    Why 80%? This is the industry-standard End-of-Life (EOL) threshold for
    lithium-ion batteries, widely used in EV and energy storage applications.
    Below this point, battery performance degrades rapidly and unpredictably.

    Returns:
        RUL -- number of cycles remaining from now until EOL
    """
    threshold = 0.8 * capacity[0]

    # Project 300 cycles into the future from current state
    future_cycles = np.arange(1, current_cycle_count + 300).reshape(-1, 1).astype(np.float32)
    future_norm = future_cycles / max_cycle
    predictions = model.predict(future_norm, verbose=0).flatten()

    below_threshold = np.where(predictions <= threshold)[0]

    if len(below_threshold) > 0:
        eol_cycle = int(future_cycles[below_threshold[0]][0])
        rul = max(eol_cycle - current_cycle_count, 0)
        return eol_cycle, rul

    return None, None  # EOL not reached within projection window


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_soh_and_prediction(cycles, capacity, predicted_capacity, title):
    """
    Plot actual vs predicted State of Health (SoH).

    SoH = (current capacity / initial capacity) × 100%
    A battery at 100% SoH is brand new; at 80% SoH it has reached end-of-life.
    """
    soh_actual = (capacity / capacity[0]) * 100
    soh_predicted = (predicted_capacity / capacity[0]) * 100

    plt.figure(figsize=(8, 4))
    plt.plot(cycles, soh_actual, 'x-', label='Actual SoH', color='teal')
    plt.plot(cycles, soh_predicted, '--', label='NN Trend', color='red')
    plt.axhline(y=80, color='gray', linestyle=':', label='EOL threshold (80%)')
    plt.xlabel("Cycle")
    plt.ylabel("State of Health (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_combined_soh(cycles_list, capacities_list, labels):
    """Compare degradation curves across all three battery datasets."""
    plt.figure(figsize=(8, 4))
    for cycles, cap, label in zip(cycles_list, capacities_list, labels):
        soh = (cap / cap[0]) * 100
        plt.plot(cycles, soh, label=label)
    plt.axhline(y=80, color='gray', linestyle=':', label='EOL threshold (80%)')
    plt.xlabel("Cycle")
    plt.ylabel("State of Health (%)")
    plt.title("Combined State of Health — B0005, B0006, B0007")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    files = {
        "B0005": "batt_data5.csv",
        "B0006": "batt_data6.csv",
        "B0007": "batt_data7.csv"
    }

    capacities_all = {}
    cycles_all = {}
    results = {}

    print("=" * 55)
    print("  RUL Predictor — NASA Battery Dataset")
    print("=" * 55)

    for label, path in files.items():
        capacity = load_data(path)
        if len(capacity) == 0:
            continue

        cycles = np.arange(1, len(capacity) + 1)

        # Train model
        model, max_cycle = train_model(cycles, capacity)

        # Evaluate
        mae, rmse, r2, predicted_capacity = evaluate_model(model, cycles, capacity, max_cycle)

        # Predict RUL
        eol_cycle, rul = predict_rul(model, capacity, len(cycles), max_cycle)

        # Store for combined plot
        capacities_all[label] = capacity
        cycles_all[label] = cycles

        # Print results
        print(f"\n{label}:")
        print(f"  Cycles observed : {len(cycles)}")
        print(f"  Initial capacity: {capacity[0]:.4f} Ah")
        print(f"  MAE             : {mae:.4f} Ah")
        print(f"  RMSE            : {rmse:.4f} Ah")
        print(f"  R²              : {r2:.4f}")
        if eol_cycle:
            print(f"  EOL at cycle    : {eol_cycle}")
            print(f"  RUL             : {rul} cycles remaining")
        else:
            print(f"  EOL not reached within projection window")

        plot_soh_and_prediction(cycles, capacity, predicted_capacity,
                                f"{label}: State of Health & RUL Prediction")

    # Combined plot
    plot_combined_soh(
        [cycles_all[k] for k in capacities_all],
        [capacities_all[k] for k in capacities_all],
        list(capacities_all.keys())
    )

    plt.show()