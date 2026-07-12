from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# ── Load model once at startup ───────────────────────────────────────────────

model = load_model('models/rul_model.h5', compile=False)
max_cycle = float(np.load('models/max_cycle.npy')[0])

print(f"Model loaded. max_cycle = {max_cycle}")

# ── Request schema ────────────────────────────────────────────────────────────


class BatteryData(BaseModel):
    capacity: list[float]

# ── RUL logic ─────────────────────────────────────────────────────────────────


def predict_rul(capacity, current_cycle):
    threshold = 0.8 * capacity[0]
    all_cycles = np.arange(
        1, current_cycle + 300).reshape(-1, 1).astype(np.float32)
    all_preds = model.predict(all_cycles / max_cycle, verbose=0).flatten()
    below = np.where(all_preds <= threshold)[0]
    if len(below) > 0:
        eol = int(below[0] + 1)
        rul = max(eol - current_cycle, 0)
        return eol, rul
    return None, None

# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"message": "Battery RUL Predictor API", "docs": "/docs"}


@app.post("/predict")
def predict(data: BatteryData):
    if len(data.capacity) == 0:
        return {"error": "capacity list cannot be empty"}, 400
    capacity = np.array(data.capacity)
    cycles = np.arange(1, len(capacity) + 1)

    eol, rul = predict_rul(capacity, len(cycles))

    return {
        "cycles_observed": len(cycles),
        "initial_capacity_ah": round(float(capacity[0]), 4),
        "current_capacity_ah": round(float(capacity[-1]), 4),
        "current_soh_pct": round(float(capacity[-1] / capacity[0] * 100), 2),
        "predicted_eol_cycle": eol,
        "predicted_rul_cycles": rul,
        "eol_threshold_ah": round(float(0.8 * capacity[0]), 4)
    }
