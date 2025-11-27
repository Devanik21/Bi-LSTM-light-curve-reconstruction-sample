import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as st_scipy
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras import initializers

# --- CONFIGURATION & SEEDS ---
st.set_page_config(page_title="BiLSTM GRB Reconstructor", layout="wide")
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- 0. CATALOG OF FAMOUS BURSTS (The "Cheat Sheet") ---
# NASA stores data by "Target ID", not Name. We map them here to ensure success.
GRB_CATALOG = {
    "GRB 130427A (The 'Monster' Burst)": "00554620",
    "GRB 060729 (Longest X-ray Afterglow)": "00221755",
    "GRB 190114C (First TeV Detection)": "00883832",
    "GRB 131030A (Classic Decay)": "00576238"
}

# --- 1. THE LIVE DATA FETCHER (NASA/UKSSDC) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to save time
def fetch_swift_data(grb_id):
    """
    Connects to the UK Swift Science Data Centre (UKSSDC) and retrieves
    the official light curve data using the Target ID.
    """
    # Construct the URL using the correct Target ID
    # Example: https://www.swift.ac.uk/xrt_curves/00554620/flux.qdp
    url = f"https://www.swift.ac.uk/xrt_curves/{grb_id}/flux.qdp"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 404:
            return None, f"Data not found for ID {grb_id}. NASA archive might be busy."
        elif response.status_code != 200:
            return None, f"Connection Error: {response.status_code}"
            
        # Parse the QDP file format (Scientific format used by astronomers)
        data_lines = []
        raw_lines = response.text.splitlines()
        
        for line in raw_lines:
            line = line.strip()
            # Skip comments and commands
            if not line or line.startswith("!") or line.startswith("READ"):
                continue
            
            # 'NO DATA' separates modes. We skip the text but keep reading.
            if "NO DATA" in line:
                continue
                
            parts = line.split()
            # We expect at least Time, Flux, FluxErr (sometimes 4, 5 or 6 cols)
            if len(parts) >= 4:
                try:
                    # Try to convert to floats to ensure it's valid data
                    float_vals = [float(p) for p in parts]
                    data_lines.append(float_vals)
                except ValueError:
                    continue
                    
        if not data_lines:
            return None, "Data found but parsing failed (Empty dataset)."

        # Convert to DataFrame
        # QDP columns are usually: Time, T_err+, T_err-, Flux, Flux_err+, Flux_err-
        # We normalize everything to: t, flux, flux_err
        df = pd.DataFrame(data_lines)
        
        # Handle different column counts dynamically
        final_cols = ['t', 'flux', 'flux_err']
        final_data = pd.DataFrame(columns=final_cols)
        
        if df.shape[1] >= 6:
            # Full 6 columns: t, t_pos, t_neg, flux, flux_pos, flux_neg
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 3]
            final_data['flux_err'] = df.iloc[:, 4] # Use positive error
        elif df.shape[1] == 5:
            # 5 columns: t, t_pos, t_neg, flux, flux_err (symmetric)
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 3]
            final_data['flux_err'] = df.iloc[:, 4]
        elif df.shape[1] == 4:
            # 4 columns: t, t_err, flux, flux_err
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 2]
            final_data['flux_err'] = df.iloc[:, 3]
            
        # Filter bad data (negative time or non-positive flux for Log scale)
        final_data = final_data[final_data['flux'] > 0]
        final_data = final_data[final_data['t'] > 10] # Skip very early trigger noise
        
        return final_data.sort_values('t'), None
        
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- 2. THE ADAPTIVE ARCHITECT (Self-Learning) ---
def build_adaptive_model(n_data_points):
    """
    Dynamically designs the Neural Network based on how much data we have.
    """
    he_init = initializers.HeNormal()
    model = Sequential()
    
    # Architecture Logic
    if n_data_points < 80:
        # LOW DATA REGIME: Lightweight model
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Lightweight)")
        model.add(Bidirectional(LSTM(32, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(32, kernel_initializer=he_init)))
        epochs = 120 
        batch_size = 2
        
    elif n_data_points < 300:
        # MEDIUM DATA REGIME: Standard architecture
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Standard)")
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, kernel_initializer=he_init)))
        epochs = 80
        batch_size = 4
        
    else:
        # HIGH DATA REGIME: Deep architecture
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Deep Learning)")
        model.add(Bidirectional(LSTM(128, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(128, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init)))
        epochs = 50
        batch_size = 8

    model.add(Dense(1, activation='linear')) 
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model, epochs, batch_size

# --- 3. HELPER FUNCTIONS ---
def create_reconstruction_points(log_ts):
    """Creates a smooth timeline to fill in the gaps."""
    min_t, max_t = log_ts.min(), log_ts.max()
    # Create a dense grid of points (2x the resolution of original data)
    return np.linspace(min_t, max_t, num=int(len(log_ts) * 3)).reshape(-1, 1)

# --- 4. MAIN APPLICATION ---
st.title("🔭 BiLSTM-GRB: Live Adaptive Reconstruction")
st.markdown("""
This tool connects directly to the **Neil Gehrels Swift Observatory** to fetch real-time X-ray data.
The **Bi-LSTM** architecture automatically adapts its depth based on the data density of the specific burst.
""")

# Sidebar
st.sidebar.header("📡 Mission Control")

# SELECTION LOGIC: Dropdown is safer than text input
selected_name = st.sidebar.selectbox("Select Target Burst", list(GRB_CATALOG.keys()))
target_id = GRB_CATALOG[selected_name]

st.sidebar.caption(f"Target ID: {target_id}") # Show the secret ID to look pro

run_btn = st.sidebar.button("Fetch & Reconstruct", type="primary")

if run_btn:
    with st.spinner(f"Contacting Swift-XRT Repository for {selected_name}..."):
        # Fetch using the ID, not the name!
        raw_data, error_msg = fetch_swift_data(target_id)
        
    if error_msg:
        st.error(f"❌ {error_msg}")
    else:
        st.success(f"✅ Connection Established! Downloaded {len(raw_data)} data points.")
        
        # DATA PREPROCESSING
        ts = raw_data['t'].values
        fluxes = raw_data['flux'].values
        flux_errs = raw_data['flux_err'].values
        
        # Log Transformation (Critical for GRBs)
        # We add a tiny epsilon to avoid log(0) if any bad data slipped through
        log_ts = np.log10(ts)
        log_fluxes = np.log10(fluxes)
        
        # Prepare Tensors
        X = log_ts.reshape(-1, 1)
        y = log_fluxes.reshape(-1, 1)
        
        # Scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
        
        # BUILD & TRAIN
        model, epochs, batch = build_adaptive_model(len(raw_data))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.6f}")

        with st.spinner("Training BiLSTM on specific burst physics..."):
            history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=batch, 
                              verbose=0, callbacks=[ProgressBarCallback()])
        
        # INFERENCE
        recon_log_t = create_reconstruction_points(log_ts)
        recon_scaled = scaler_X.transform(recon_log_t)
        recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
        
        predictions_scaled = model.predict(recon_seq, verbose=0)
        predictions_log = scaler_y.inverse_transform(predictions_scaled)
        
        # VISUALIZATION
        st.subheader(f"Temporal Reconstruction: {selected_name.split('(')[0]}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot 1: The Real Data (Background)
        ax.errorbar(log_ts, log_fluxes, yerr=flux_errs/(fluxes*np.log(10)), 
                   fmt='o', color='gray', alpha=0.3, markersize=4, 
                   label='Official Swift-XRT Data (Observed)')
        
        # Plot 2: The AI Reconstruction
        ax.plot(recon_log_t, predictions_log, color='#FF4B4B', linewidth=2.5, 
               label='BiLSTM Reconstruction (AI)')
        
        ax.set_xlabel("log(Time) [s]")
        ax.set_ylabel("log(Flux) [erg/cm²/s]")
        ax.set_title(f"Light Curve Morphology - {selected_name}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        st.pyplot(fig)
        
        # METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("Data Points Fetched", len(raw_data))
        col2.metric("Reconstruction Loss", f"{history.history['loss'][-1]:.5f}")
        col3.metric("Flux Dynamic Range", f"10^{np.ptp(log_fluxes):.1f}")
        
        # DOWNLOAD
        df_export = pd.DataFrame({
            'log_time': recon_log_t.flatten(),
            'reconstructed_log_flux': predictions_log.flatten()
        })
        csv = df_export.to_csv(index=False)
        st.download_button("Download Reconstruction Data", csv, "reconstruction.csv", "text/csv")

else:
    st.info("👈 Select a GRB from the sidebar and click Fetch to start.")
