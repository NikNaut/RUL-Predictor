import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
    model.fit(X, y, epochs=300, verbose=1)
    return model, max_cycle

# Load all three batteries and combine
dfs = []
for f in ['data/batt_data5.csv', 'data/batt_data6.csv', 'data/batt_data7.csv']:
    df = pd.read_csv(f)
    dfs.append(df['capacity'].values)

# Train on B0005 as primary battery
capacity  = dfs[0]
cycles    = np.arange(1, len(capacity) + 1)
model, max_cycle = train_model(cycles, capacity)

# Save model and max_cycle
model.save('models/rul_model.h5')
np.save('models/max_cycle.npy', np.array([max_cycle]))

print(f"Model saved. max_cycle = {max_cycle}")