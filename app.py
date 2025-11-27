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

# --- 1. THE LIVE DATA FETCHER (NASA/UKSSDC) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to save time
def fetch_swift_data(grb_name):
    """
    Connects to the UK Swift Science Data Centre (UKSSDC) and retrieves
    the official light curve data for a specific Gamma-Ray Burst.
    """
    # Clean up the name (e.g., "GRB 130427A" -> "GRB130427A")
    clean_name = grb_name.replace(" ", "").upper()
    if not clean_name.startswith("GRB"):
        clean_name = "GRB" + clean_name
        
    # Standard URL pattern for Swift-XRT repository
    url = f"https://www.swift.ac.uk/xrt_curves/{clean_name}/flux.qdp"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None, "GRB not found in Swift Repository. Check the name."
        elif response.status_code != 200:
            return None, f"Connection Error: {response.status_code}"
            
        # Parse the QDP file format (Scientific format used by astronomers)
        # We need to skip '!' comments and handle 'NO DATA' separators
        data_lines = []
        raw_lines = response.text.splitlines()
        
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("READ"):
                continue
            
            # 'NO DATA' separates different telescope modes (WT/PC). 
            # We treat them as one continuous dataset for reconstruction.
            if "NO DATA" in line:
                continue
                
            parts = line.split()
            # We expect at least Time, Flux, FluxErr (sometimes 5 or 6 cols)
            if len(parts) >= 4:
                try:
                    # Try to convert to floats to ensure it's data
                    float_vals = [float(p) for p in parts]
                    data_lines.append(float_vals)
                except ValueError:
                    continue
                    
        if not data_lines:
            return None, "Data found but could not be parsed."

        # Convert to DataFrame
        # Standard QDP cols: Time, T_pos_err, T_neg_err, Flux, Flux_pos_err, Flux_neg_err
        # We handle flexible column counts just in case
        cols = ['t', 't_err_pos', 't_err_neg', 'flux', 'flux_err_pos', 'flux_err_neg']
        df = pd.DataFrame(data_lines)
        
        # Mapping columns dynamically
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = cols
        elif df.shape[1] == 5: # Sometimes errors are symmetric
             df.columns = ['t', 't_err_pos', 't_err_neg', 'flux', 'flux_err']
             df['flux_err_pos'] = df['flux_err']
             df['flux_err_neg'] = df['flux_err']
        elif df.shape[1] == 4:
             df.columns = ['t', 't_err', 'flux', 'flux_err']
             df['t_err_pos'] = df['t_err']
             df['t_err_neg'] = df['t_err']
             df['flux_err_pos'] = df['flux_err']
             df['flux_err_neg'] = df['flux_err']
             
        # Filter bad data (negative time or non-positive flux for Log scale)
        df = df[df['flux'] > 0]
        df = df[df['t'] > 0]
        
        return df.sort_values('t'), None
        
    except Exception as e:
        return None, str(e)

# --- 2. THE ADAPTIVE ARCHITECT (Self-Learning) ---
def build_adaptive_model(n_data_points):
    """
    Dynamically designs the Neural Network based on how much data we have.
    This prevents overfitting on small bursts and underfitting on large ones.
    """
    he_init = initializers.HeNormal()
    model = Sequential()
    
    # Architecture Logic
    if n_data_points < 50:
        # LOW DATA REGIME: Lightweight model to prevent memorization
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Lightweight)")
        model.add(Bidirectional(LSTM(32, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(16, kernel_initializer=he_init)))
        epochs = 150 # More epochs needed for small data to converge
        batch_size = 2
        
    elif n_data_points < 200:
        # MEDIUM DATA REGIME: Standard architecture
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Standard)")
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(32, kernel_initializer=he_init)))
        epochs = 100
        batch_size = 4
        
    else:
        # HIGH DATA REGIME: Deep architecture for complex features
        st.sidebar.markdown("🧠 **AI Status:** Adaptive Mode (Deep Learning)")
        model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)))
        model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)))
        model.add(Bidirectional(LSTM(50, kernel_initializer=he_init)))
        epochs = 60
        batch_size = 8

    model.add(Dense(1, activation='linear')) # Linear is better for Log-Space reconstruction
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model, epochs, batch_size

# --- 3. HELPER FUNCTIONS ---
def create_reconstruction_points(log_ts, fraction=0.5):
    """Creates a smooth timeline to fill in the gaps."""
    min_t, max_t = log_ts.min(), log_ts.max()
    # Create a dense grid of points
    return np.linspace(min_t, max_t, num=int(len(log_ts) * 2)).reshape(-1, 1)

# --- 4. MAIN APPLICATION ---
st.title("🔭 BiLSTM-GRB: Live Adaptive Reconstruction")
st.markdown("""
This tool connects directly to the **Neil Gehrels Swift Observatory** to fetch real-time X-ray data.
The **Bi-LSTM** architecture automatically adapts its depth based on the data density of the specific burst.
""")

# Sidebar
st.sidebar.header("📡 Mission Control")
grb_input = st.sidebar.text_input("Enter GRB Designation", "GRB 130427A")
st.sidebar.info("Try: GRB 130427A, GRB 060729, or GRB 190114C")

run_btn = st.sidebar.button("Fetch & Reconstruct", type="primary")

if run_btn:
    with st.spinner(f"Contacting Swift-XRT Repository for {grb_input}..."):
        raw_data, error_msg = fetch_swift_data(grb_input)
        
    if error_msg:
        st.error(f"❌ {error_msg}")
    else:
        st.success(f"✅ Data Acquired! Found {len(raw_data)} observational points.")
        
        # DATA PREPROCESSING
        ts = raw_data['t'].values
        fluxes = raw_data['flux'].values
        flux_errs = raw_data['flux_err_pos'].values
        
        # Log Transformation (Critical for GRBs)
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
        st.subheader(f"Temporal Reconstruction: {grb_input.upper()}")
        
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
        ax.set_title(f"{grb_input.upper()} - Light Curve Morphology")
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
    st.info("👈 Enter a GRB name (e.g., GRB 130427A) and click Fetch to start.")
