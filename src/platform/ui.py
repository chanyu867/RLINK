import streamlit as st
import numpy as np
import pandas as pd
import time
import altair as alt

# --- Import from our refactored algorithms ---
from src.utils.npy_loader import npy_loader
from src.RL_decoders.algorithms_batch import (
    AGREL_step, Banditron_step, BanditronRP_step, HRL_step, OnlineLagBuffer
)

# --- Page Config ---
st.set_page_config(page_title="Neural Decoder Real-time Monitor", layout="wide")
st.title("🧠 RL Finger Position Decoder: Real-time Monitor")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["AGREL", "HRL", "Banditron", "BanditronRP"])
window_size = st.sidebar.slider("Display Window (Samples)", 50, 2000, 300)
speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.00, 0.2, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("**Hyperparameters**")
gamma = st.sidebar.slider("Gamma (Exploration)", 0.0, 1.0, 0.1)
alpha = st.sidebar.slider("Alpha/LR (Learning Rate)", 0.0001, 0.1, 0.01, format="%.4f")
# Simplified: using alpha for beta/mu as well for UI cleanliness
error_rate = 0.0 # user can set if needed, hardcoded for demo

# --- Data Loading ---
@st.cache_data
def load_web_data():
    base_path = "/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data"
    # Adjust these paths to your exact file locations
    sbp = npy_loader(f"{base_path}/all/sbp_all.npy")
    labels = npy_loader(f"{base_path}/classes_dir/idx_position_labels_0.33_0.66_shift0.npy")
    day_info = npy_loader(f"{base_path}/all/day_number.npy")
    raw_pos = npy_loader(f"{base_path}/all/idx_position_all.npy")
    trial_info = npy_loader(f"{base_path}/all/trial_bin.npy")
    
    # Truncate for performance if needed, or load full
    limit = 50000 
    return sbp[:limit], labels[:limit], day_info[:limit], raw_pos[:limit], trial_info[:limit]

try:
    X, y, day_info, raw_pos, trial_info = load_web_data()
    
    # Day Slider Logic
    unique_days = np.unique(day_info)
    max_days = len(unique_days)
    
    if max_days > 1:
        selected_num_days = st.sidebar.slider("Number of Days to Run", 1, int(max_days), 1)
        target_days = unique_days[:selected_num_days]
        mask = np.isin(day_info, target_days)
    else:
        st.sidebar.info("Single day dataset detected.")
        mask = np.ones(len(day_info), dtype=bool)

    X_s = X[mask]
    y_s = y[mask]
    pos_s = raw_pos[mask]
    tri_s = trial_info[mask]
    
    st.sidebar.success(f"Data Loaded: {len(y_s)} samples")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# --- Visualization Helper (Altair) ---
def create_dashboard_charts(df, window_size):
    # 1. Main Decoder Chart
    base = alt.Chart(df).encode(
        x=alt.X('index:Q', 
                scale=alt.Scale(domain=[df['index'].min(), df['index'].max()]),
                axis=alt.Axis(title='Time (Samples)', tickMinStep=1))
    )

    y_scale_shared = alt.Scale(domain=[-0.1, 1.1])

    # A. Background
    bands = pd.DataFrame([
        {'start': 0.0, 'end': 0.33, 'color': '#ffcccc'}, 
        {'start': 0.33, 'end': 0.66, 'color': '#ccffcc'}, 
        {'start': 0.66, 'end': 1.0, 'color': '#ccccff'}   
    ])
    
    background = alt.Chart(bands).mark_rect(opacity=0.3).encode(
        y=alt.Y('start:Q', scale=y_scale_shared), 
        y2='end:Q', 
        color=alt.Color('color:N', scale=None)
    )

    # B. Actual Finger Position
    line_actual = base.mark_line(color='red', strokeDash=[4, 4], strokeWidth=2).encode(
        y=alt.Y('Actual:Q', 
                scale=y_scale_shared, 
                axis=alt.Axis(title='Position (0-1)'))
    )

    # C. Prediction
    line_pred = base.mark_line(color='deepskyblue', strokeWidth=3).encode(
        y=alt.Y('PredictionMapped:Q', scale=y_scale_shared)
    )

    # D. Markers
    trial_markers = base.mark_rule(color='orange', strokeWidth=2).transform_filter(
        alt.datum.TrialChange == True
    )

    # FIX: Added width='container' to force full width
    chart_main = (background + line_actual + line_pred + trial_markers).properties(
        height=300, 
        width='container',
        title="Real-time Decoding Stream"
    )

    # 2. Performance Chart
    chart_perf = alt.Chart(df).mark_area(
        line={'color':'darkgreen'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color='darkgreen', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('index:Q', axis=None), 
        y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), title='Rolling Acc'),
        tooltip=['Accuracy']
    ).properties(
        height=100,
        width='container'
    )

    return alt.vconcat(chart_main, chart_perf).configure_axis(grid=False)


# --- Helper: Model State Initialization ---
def init_model_state(model_type, input_dim, num_classes):
    n_lag = 5 
    lag_step = 1
    lagger = OnlineLagBuffer(input_dim, n_lag, lag_step)
    d_eff = input_dim * (n_lag + 1)
    
    state = {'lagger': lagger}
    
    if model_type == "Banditron":
        # W: (Classes, Input)
        state['W'] = np.zeros((num_classes, d_eff))
        
    elif model_type == "BanditronRP":
        # Random Projection: Project d_eff -> k (or similar)
        # Using k=num_classes * 4 for projection dim
        m_dim = num_classes * 4 
        state['Wrand'] = np.random.uniform(-1, 1, (m_dim, d_eff))
        state['W'] = np.zeros((num_classes, m_dim))
        
    elif model_type == "AGREL":
        hidden_nodes = 55
        # W_H: (Hidden, Input+1bias)
        state['W_H'] = np.random.uniform(-0.1, 0.1, (hidden_nodes, d_eff + 1))
        # W_O: (Classes, Hidden)
        state['W_O'] = np.random.uniform(-0.1, 0.1, (num_classes, hidden_nodes))
        
    elif model_type == "HRL":
        hidden_nodes = [37] # Example 2 hidden layers
        # List of matrices
        # L1: Input+1 -> H1
        W = []
        W.append(np.random.randn(hidden_nodes[0], d_eff + 1) * 0.1)
        # L2: H1 -> H2
        W.append(np.random.randn(hidden_nodes[1], hidden_nodes[0]) * 0.1)
        # L3: H2 -> Output
        W.append(np.random.randn(num_classes, hidden_nodes[1]) * 0.1)
        state['W'] = W
        
    return state

# --- Main Execution ---
if st.button("🚀 Start Real-time Decoding"):
    
    # 1. Setup
    input_dim = X_s.shape[1]
    num_classes = 3 # Fixed for this problem (0, 1, 2)
    
    # Initialize State
    state = init_model_state(model_choice, input_dim, num_classes)
    
    # Params
    gamma = 0.19967440499033728
    params = {
        'gamma': gamma,
        'sparsity_rate': 0.0,
        'error': error_rate,
        'alpha': 0.08929538546844354, # Used for AGREL
        'beta': 0.021962068857794316,  # Used for AGREL
        'muH': 0.010323333452520149,   # Used for HRL
        'muO': 0.06999875904411794    # Used for HRL
    }

    # Placeholders
    with st.container(height=650, border=False):
        dashboard_ph = st.empty()

    metrics_cols = st.columns(4)
    metric_acc = metrics_cols[0].empty()
    metric_class = metrics_cols[1].empty()
    metric_trial = metrics_cols[2].empty()
    
    # Buffers
    rows = []
    correct_buffer = [] # for rolling accuracy
    rolling_win = 100
    
    # 2. Loop
    for t in range(len(y_s)):
        
        # A. Run Step
        x_curr = X_s[t]
        y_curr = y_s[t]
        
        if model_choice == "Banditron":
            pred, state, _ = Banditron_step(x_curr, y_curr, state, params)
        elif model_choice == "BanditronRP":
            pred, state, _ = BanditronRP_step(x_curr, y_curr, state, params)
        elif model_choice == "AGREL":
            pred, state, _ = AGREL_step(x_curr, y_curr, state, params)
        elif model_choice == "HRL":
            pred, state, _ = HRL_step(x_curr, y_curr, state, params)
            
        # B. Map Prediction to 0-1 Scale (Center of Bands)
        # Class 0 (0-0.33) -> 0.165
        # Class 1 (0.33-0.66) -> 0.50
        # Class 2 (0.66-1.0) -> 0.835
        mapping = {0: 0.165, 1: 0.5, 2: 0.835}
        pred_mapped = mapping.get(pred, 0.5)
        
        # C. Rolling Accuracy
        is_correct = 1 if pred == y_curr else 0
        correct_buffer.append(is_correct)
        if len(correct_buffer) > rolling_win: correct_buffer.pop(0)
        curr_acc = np.mean(correct_buffer)
        
        # D. Store Data
        rows.append({
            'index': t,
            'Actual': pos_s[t],
            'PredictionMapped': pred_mapped,
            'PredictionClass': pred,
            'TrialChange': (tri_s[t] == 0),
            'Accuracy': curr_acc
        })
        
        # Keep buffer limited size for UI speed
        if len(rows) > window_size:
            rows.pop(0)
            
        # E. Update UI (throttled)
        if t % 5 == 0:
            df = pd.DataFrame(rows)
            
            # FIX: Use direct method call instead of 'with container()' to prevent scroll jumping
            dashboard_ph.altair_chart(
                create_dashboard_charts(df, window_size), 
                use_container_width=True
            )
            
            metric_acc.metric("Rolling Accuracy", f"{curr_acc:.1%}")
            metric_class.metric("Curr Class", f"{y_curr} vs {pred}")
            metric_trial.metric("Trial", f"{tri_s[t]}")
            
            time.sleep(speed)
            
    st.success("Simulation Finished.")