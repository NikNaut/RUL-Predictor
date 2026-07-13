import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(
    page_title="Battery RUL Predictor",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
#MainMenu, footer, header { visibility: hidden; }

.hero { padding: 3rem 0 2rem 0; border-bottom: 1px solid #21262d; margin-bottom: 2rem; }
.hero-tag { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px; color: #58a6ff; text-transform: uppercase; margin-bottom: 0.75rem; }
.hero-title { font-size: 2.4rem; font-weight: 600; color: #e6edf3; margin: 0; line-height: 1.2; }
.hero-title span { color: #58a6ff; }
.hero-sub { font-size: 1rem; color: #8b949e; margin-top: 0.5rem; font-weight: 300; }

.section-header { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 3px; color: #58a6ff; text-transform: uppercase; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #21262d; }

.metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.25rem 1.5rem; text-align: center; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 2px; color: #8b949e; text-transform: uppercase; margin-bottom: 0.5rem; }
.metric-value { font-size: 1.8rem; font-weight: 600; color: #e6edf3; font-family: 'IBM Plex Mono', monospace; }
.metric-value.good { color: #3fb950; }
.metric-value.warn { color: #d29922; }
.metric-value.bad  { color: #f85149; }

.rul-card { background: #161b22; border: 1px solid #21262d; border-left: 3px solid #58a6ff; border-radius: 8px; padding: 1.5rem 2rem; margin: 1rem 0; }
.rul-card.dead { border-left-color: #f85149; }
.rul-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 2px; color: #8b949e; text-transform: uppercase; margin-bottom: 0.25rem; }
.rul-value { font-size: 2.5rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }
.rul-value.bad { color: #f85149; }
.rul-sub { font-size: 0.8rem; color: #8b949e; margin-top: 0.25rem; }

.eol-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.25rem 1.5rem; text-align: center; }
.eol-pred { border-top: 3px solid #f0a30a; }
.eol-actual { border-top: 3px solid #3fb950; }
.eol-error { border-top: 3px solid #8b949e; }

.info-box { background: #161b22; border: 1px solid #21262d; border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0; font-size: 0.875rem; color: #8b949e; }
.error-box { background: #161b22; border: 1px solid #21262d; border-left: 3px solid #f85149; border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0; font-size: 0.875rem; color: #f85149; }

[data-testid="stFileUploader"] { background: #161b22; border: 1px dashed #30363d; border-radius: 8px; padding: 1rem; }
[data-testid="stExpander"] { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }
[data-testid="stDataFrame"] { background: #161b22; }

.stMainBlockContainer { padding: 1rem 5rem !important; }
</style>
""", unsafe_allow_html=True)

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
    all_cycles = np.arange(
        1, current_cycle + 300).reshape(-1, 1).astype(np.float32)
    all_preds = model.predict(all_cycles / max_cycle, verbose=0).flatten()
    below = np.where(all_preds <= threshold)[0]
    if len(below) > 0:
        eol = int(below[0] + 1)
        rul = max(eol - current_cycle, 0)
        return eol, rul
    return None, None


def set_plot_style():
    plt.rcParams.update({
        'figure.facecolor': '#161b22',
        'axes.facecolor':   '#161b22',
        'axes.edgecolor':   '#30363d',
        'axes.labelcolor':  '#8b949e',
        'axes.titlecolor':  '#e6edf3',
        'xtick.color':      '#8b949e',
        'ytick.color':      '#8b949e',
        'grid.color':       '#21262d',
        'grid.linestyle':   '--',
        'grid.linewidth':   0.6,
        'text.color':       '#e6edf3',
        'legend.facecolor': '#161b22',
        'legend.edgecolor': '#30363d',
        'legend.labelcolor': '#e6edf3',
        'font.family':      'monospace',
        'font.size':        10,
    })

# ── Hero ──────────────────────────────────────────────────────────────────────


st.markdown("""
<div class="hero">
    <h1 class="hero-title">Battery <span>RUL</span> Predictor</h1>
    <div class="hero-tag">Predictive Maintenance · NASA PCoE Dataset</div>
    <p class="hero-sub">Neural network estimation of remaining useful life for lithium-ion batteries</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">01 &nbsp; Data Input</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    Upload a CSV file with a <code>capacity</code> column — discharge capacity (Ah) per cycle. One row = one cycle.
</div>
""", unsafe_allow_html=True)

with st.expander("Expected CSV format"):
    example = pd.DataFrame(
        {'capacity': [1.8565, 1.8463, 1.8353, 1.8352, 1.8346]})
    st.dataframe(example, use_container_width=True)
    st.caption(
        "Only the capacity column is required. Additional columns are ignored.")

uploaded_file = st.file_uploader(
    "", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'capacity' not in df.columns:
        st.markdown(
            '<div class="error-box">CSV must contain a <code>capacity</code> column.</div>', unsafe_allow_html=True)
        st.stop()

    capacity = df['capacity'].values
    cycles = np.arange(1, len(capacity) + 1)
    soh_now = (capacity[-1] / capacity[0]) * 100
    threshold = 0.8 * capacity[0]

    # ── Battery Overview ──────────────────────────────────────────────────────

    st.markdown('<div class="section-header">02 &nbsp; Battery Overview</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Cycles Observed</div><div class="metric-value">{len(cycles)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Initial Capacity</div><div class="metric-value">{capacity[0]:.4f} <span style="font-size:1rem;color:#8b949e">Ah</span></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Current Capacity</div><div class="metric-value">{capacity[-1]:.4f} <span style="font-size:1rem;color:#8b949e">Ah</span></div></div>', unsafe_allow_html=True)
    with c4:
        soh_class = "good" if soh_now >= 90 else "warn" if soh_now >= 80 else "bad"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Current SoH</div><div class="metric-value {soh_class}">{soh_now:.1f}<span style="font-size:1rem">%</span></div></div>', unsafe_allow_html=True)

    # ── Train ─────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">03 &nbsp; Curve Fit Quality</div>',
            unsafe_allow_html=True)
    with st.spinner("Training neural network..."):
        model, max_cycle = train_model(cycles, capacity)
        X = (cycles / max_cycle).reshape(-1, 1).astype(np.float32)
        predicted = model.predict(X, verbose=0).flatten()
        mae = mean_absolute_error(capacity, predicted)
        rmse = np.sqrt(mean_squared_error(capacity, predicted))
        r2 = r2_score(capacity, predicted)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">MAE</div><div class="metric-value">{mae:.4f} <span style="font-size:1rem;color:#8b949e">Ah</span></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.4f} <span style="font-size:1rem;color:#8b949e">Ah</span></div></div>', unsafe_allow_html=True)
    with c3:
        r2_class = "good" if r2 >= 0.95 else "warn" if r2 >= 0.85 else "bad"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">R²</div><div class="metric-value {r2_class}">{r2:.4f}</div></div>', unsafe_allow_html=True)
    st.caption("Model is trained fresh on this battery's own data — these metrics reflect fit quality, not generalization to unseen batteries.")

    # ── RUL ───────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">04 &nbsp; RUL Prediction</div>',
                unsafe_allow_html=True)
    eol, predicted_rul = predict_rul(model, capacity, len(cycles), max_cycle)

    actual_eol_idx = np.where(capacity <= threshold)[0]
    actual_eol = int(actual_eol_idx[0] +
                    1) if len(actual_eol_idx) > 0 else None

    if eol is not None:
        if predicted_rul == 0:
            st.markdown(f"""
            <div class="rul-card dead">
                <div class="rul-label">Remaining Useful Life</div>
                <div class="rul-value bad">0 <span style="font-size:1.2rem;color:#8b949e">cycles</span></div>
                <div class="rul-sub">Battery has already passed End-of-Life at cycle {eol} &nbsp;·&nbsp; EOL threshold: {threshold:.4f} Ah</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rul-card">
                <div class="rul-label">Remaining Useful Life</div>
                <div class="rul-value">{predicted_rul} <span style="font-size:1.2rem;color:#8b949e">cycles</span></div>
                <div class="rul-sub">Predicted End-of-Life at cycle {eol} &nbsp;·&nbsp; EOL threshold: {threshold:.4f} Ah</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box">EOL not reached within projection window (+300 cycles).</div>', unsafe_allow_html=True)

    # ── EOL Comparison ────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">05 &nbsp; Predicted vs Actual EOL</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="eol-card eol-pred"><div class="metric-label">Predicted EOL</div><div class="metric-value">{"Cycle " + str(eol) if eol else "N/A"}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="eol-card eol-actual"><div class="metric-label">Actual EOL</div><div class="metric-value">{"Cycle " + str(actual_eol) if actual_eol else "Not in data"}</div></div>', unsafe_allow_html=True)
    with c3:
        if eol and actual_eol:
            error = abs(eol - actual_eol)
            pct = (error / actual_eol) * 100
            err_cls = "good" if pct < 10 else "warn" if pct < 25 else "bad"
            st.markdown(
                f'<div class="eol-card eol-error"><div class="metric-label">EOL Error</div><div class="metric-value {err_cls}">{error} <span style="font-size:1rem;color:#8b949e">cycles</span></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="eol-card eol-error"><div class="metric-label">EOL Error</div><div class="metric-value" style="color:#8b949e">N/A</div></div>', unsafe_allow_html=True)

    # ── Plot ──────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">06 &nbsp; State of Health</div>',
                unsafe_allow_html=True)
    set_plot_style()

    soh_actual = (capacity / capacity[0]) * 100
    soh_predicted = (predicted / capacity[0]) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cycles, soh_actual,    color='#58a6ff',
            linewidth=1.5, label='Actual SoH')
    ax.plot(cycles, soh_predicted, color='#f85149',
            linewidth=1.5, linestyle='--', label='NN Prediction')
    ax.axhline(y=80, color='#8b949e', linestyle=':', linewidth=1,
                label=f'EOL threshold (80% = {threshold:.4f} Ah)')
    if eol:
        ax.axvline(x=eol,        color='#f0a30a', linestyle='--',
                    linewidth=1, label=f'Predicted EOL: cycle {eol}')
    if actual_eol:
        ax.axvline(x=actual_eol, color='#3fb950', linestyle='--',
                    linewidth=1, label=f'Actual EOL: cycle {actual_eol}')
    ax.set_xlabel("Cycle")
    ax.set_ylabel("State of Health (%)")
    ax.set_title("State of Health Over Cycle Life", pad=12)
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    st.pyplot(fig)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #8b949e;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔋</div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase;">
            Upload a CSV file to begin analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
