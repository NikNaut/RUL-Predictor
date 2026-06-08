import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Battery RUL Predictor", page_icon="🔋")

# ── Model ─────────────────────────────────────────────────────────────────────

def train_model(cycles, capacity):
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

def predict_rul(model, capacity, current_cycle, max_cycle):
    threshold = 0.8 * capacity[0]
    all_cycles = np.arange(1, current_cycle + 300).reshape(-1, 1).astype(np.float32)
    all_preds  = model.predict(all_cycles / max_cycle, verbose=0).flatten()
    below      = np.where(all_preds <= threshold)[0]
    if len(below) > 0:
        eol = int(below[0] + 1)
        rul = max(eol - current_cycle, 0)
        return eol, rul
    return None, None

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔋 Battery RUL Predictor")
st.markdown("Upload a CSV file with a `capacity` column to predict remaining useful life.")

st.info("CSV must have a `capacity` column with discharge capacity (Ah) per cycle. One row = one cycle.")

with st.expander("See example CSV format"):
    example = pd.DataFrame({'capacity': [1.8560, 1.8451, 1.8392, 1.8103, 1.7865]})
    st.dataframe(example)
    st.caption("Each row represents one charge-discharge cycle.")

uploaded_file = st.file_uploader("Upload battery CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'capacity' not in df.columns:
        st.error("CSV must have a 'capacity' column.")
        st.stop()

    capacity = df['capacity'].values
    cycles   = np.arange(1, len(capacity) + 1)
    soh_now  = (capacity[-1] / capacity[0]) * 100
    threshold = 0.8 * capacity[0]

    # ── Battery Overview ──────────────────────────────────────────────────────

    st.subheader("Battery Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cycles Observed", len(cycles))
    col2.metric("Initial Capacity", f"{capacity[0]:.4f} Ah")
    col3.metric("Current Capacity", f"{capacity[-1]:.4f} Ah")
    col4.metric("Current SoH", f"{soh_now:.1f}%")

    # ── Train ─────────────────────────────────────────────────────────────────

    with st.spinner("Training model..."):
        model, max_cycle = train_model(cycles, capacity)
        X = (cycles / max_cycle).reshape(-1, 1).astype(np.float32)
        predicted = model.predict(X, verbose=0).flatten()
        mae  = mean_absolute_error(capacity, predicted)
        rmse = np.sqrt(mean_squared_error(capacity, predicted))
        r2   = r2_score(capacity, predicted)

    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Ah)", f"{mae:.4f}")
    col2.metric("RMSE (Ah)", f"{rmse:.4f}")
    col3.metric("R²", f"{r2:.4f}")

    # ── RUL + EOL ─────────────────────────────────────────────────────────────

    eol, predicted_rul = predict_rul(model, capacity, len(cycles), max_cycle)

    # Actual EOL from raw capacity data
    actual_eol_idx = np.where(capacity <= threshold)[0]
    actual_eol = int(actual_eol_idx[0] + 1) if len(actual_eol_idx) > 0 else None

    st.subheader("RUL Prediction")
    if eol is not None:
        if predicted_rul == 0:
            st.error(f"Battery has already passed End-of-Life at cycle {eol}. RUL = 0 cycles.")
        else:
            st.success(f"**Remaining Useful Life: {predicted_rul} cycles**")
            st.info(f"Predicted EOL at cycle {eol} · Threshold: 80% of initial capacity ({threshold:.4f} Ah)")
    else:
        st.info("EOL not reached within projection window (+300 cycles).")

    # ── EOL Comparison ────────────────────────────────────────────────────────

    st.subheader("Predicted vs Actual EOL")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted EOL", f"Cycle {eol}" if eol else "N/A")
    col2.metric("Actual EOL", f"Cycle {actual_eol}" if actual_eol else "Not reached in data")
    if eol and actual_eol:
        error = abs(eol - actual_eol)
        pct   = (error / actual_eol) * 100
        col3.metric("EOL Error", f"{error} cycles")

    # ── Plot ──────────────────────────────────────────────────────────────────

    st.subheader("State of Health")
    soh_actual    = (capacity / capacity[0]) * 100
    soh_predicted = (predicted / capacity[0]) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles, soh_actual,    label='Actual SoH',    color='teal')
    ax.plot(cycles, soh_predicted, label='NN Prediction', color='red', linestyle='--')
    ax.axhline(y=80, color='gray', linestyle=':', linewidth=1, label=f'EOL threshold (80% = {threshold:.4f} Ah)')
    if eol:
        ax.axvline(x=eol, color='orange', linestyle='--', linewidth=1, label=f'Predicted EOL: cycle {eol}')
    if actual_eol:
        ax.axvline(x=actual_eol, color='#3fb950', linestyle='--', linewidth=1, label=f'Actual EOL: cycle {actual_eol}')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("State of Health (%)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)