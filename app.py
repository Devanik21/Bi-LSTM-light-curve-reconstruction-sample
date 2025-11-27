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
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras import initializers

# --- CONFIGURATION & SEEDS ---
st.set_page_config(page_title="BiLSTM GRB Reconstructor", layout="wide")
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- EXPANDED CATALOG OF FAMOUS BURSTS ---
GRB_CATALOG = {
    "GRB 130427A (Monster Burst - Record Energy)": "00554620",
    "GRB 080319B (Naked Eye Visible - Brightest)": "00306757",
    "GRB 090423 (Most Distant z=8.2)": "00350184",
    "GRB 060729 (Longest X-ray Afterglow)": "00221755",
    "GRB 190114C (First TeV Gamma-Ray Detection)": "00883832",
    "GRB 131030A (Classic Power-Law Decay)": "00576238",
    "GRB 110715A (Bright Long-Duration Burst)": "00457330",
    "GRB 050509B (First Localized Short Burst)": "00118749",
    "GRB 061121 (Ultra-Bright Long Burst)": "00238995",
    "GRB 050904 (High-Redshift z=6.3)": "00154112",
    "GRB 080916C (Most Luminous - Fermi Era)": "00328000",
    "GRB 090510 (Shortest Short Burst)": "00351588",
}

# --- DATA FETCHER ---
@st.cache_data(ttl=3600)
def fetch_swift_data(grb_id):
    """Fetch official Swift-XRT light curve data from UKSSDC."""
    url = f"https://www.swift.ac.uk/xrt_curves/{grb_id}/flux.qdp"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 404:
            return None, f"Data not found for ID {grb_id}. Archive may be unavailable."
        elif response.status_code != 200:
            return None, f"Connection Error: HTTP {response.status_code}"
            
        data_lines = []
        raw_lines = response.text.splitlines()
        
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("READ"):
                continue
            if "NO DATA" in line:
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                try:
                    float_vals = [float(p) for p in parts]
                    data_lines.append(float_vals)
                except ValueError:
                    continue
                    
        if not data_lines:
            return None, "Data found but parsing failed (empty dataset)."

        df = pd.DataFrame(data_lines)
        final_cols = ['t', 'flux', 'flux_err']
        final_data = pd.DataFrame(columns=final_cols)
        
        if df.shape[1] >= 6:
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 3]
            final_data['flux_err'] = df.iloc[:, 4]
        elif df.shape[1] == 5:
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 3]
            final_data['flux_err'] = df.iloc[:, 4]
        elif df.shape[1] == 4:
            final_data['t'] = df.iloc[:, 0]
            final_data['flux'] = df.iloc[:, 2]
            final_data['flux_err'] = df.iloc[:, 3]
            
        final_data = final_data[final_data['flux'] > 0]
        final_data = final_data[final_data['t'] > 10]
        
        return final_data.sort_values('t'), None
        
    except Exception as e:
        return None, f"System Error: {str(e)}"

# --- ADAPTIVE ARCHITECTURE ---
def build_adaptive_model(n_data_points, manual_arch=None, use_dropout=False):
    """Dynamically design the Neural Network based on data density."""
    he_init = initializers.HeNormal()
    model = Sequential()
    
    if manual_arch:
        arch_type, layers, epochs, batch_size = manual_arch
        st.sidebar.markdown(f"**AI Status:** Manual Mode ({arch_type})")
        
        for i, units in enumerate(layers):
            if i < len(layers) - 1:
                model.add(Bidirectional(LSTM(units, kernel_initializer=he_init, 
                                             return_sequences=True), 
                                       input_shape=(1, 1) if i == 0 else None))
            else:
                model.add(Bidirectional(LSTM(units, kernel_initializer=he_init)))
            
            if use_dropout and i < len(layers) - 1:
                model.add(Dropout(0.2))
    else:
        if n_data_points < 80:
            st.sidebar.markdown("**AI Status:** Adaptive Mode (Lightweight)")
            model.add(Bidirectional(LSTM(32, kernel_initializer=he_init, return_sequences=True), 
                                   input_shape=(1, 1)))
            model.add(Bidirectional(LSTM(32, kernel_initializer=he_init)))
            epochs = 120 
            batch_size = 2
            
        elif n_data_points < 300:
            st.sidebar.markdown("**AI Status:** Adaptive Mode (Standard)")
            model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True), 
                                   input_shape=(1, 1)))
            model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True)))
            model.add(Bidirectional(LSTM(32, kernel_initializer=he_init)))
            epochs = 80
            batch_size = 4
            
        else:
            st.sidebar.markdown("**AI Status:** Adaptive Mode (Deep Learning)")
            model.add(Bidirectional(LSTM(128, kernel_initializer=he_init, return_sequences=True), 
                                   input_shape=(1, 1)))
            model.add(Bidirectional(LSTM(128, kernel_initializer=he_init, return_sequences=True)))
            model.add(Bidirectional(LSTM(64, kernel_initializer=he_init, return_sequences=True)))
            model.add(Bidirectional(LSTM(64, kernel_initializer=he_init)))
            epochs = 50
            batch_size = 8

    model.add(Dense(1, activation='linear')) 
    
    learning_rate = st.session_state.get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model, epochs, batch_size

# --- HELPER FUNCTIONS ---
def create_reconstruction_points(log_ts, resolution_factor):
    """Create smooth timeline with adjustable resolution."""
    min_t, max_t = log_ts.min(), log_ts.max()
    num_points = int(len(log_ts) * resolution_factor)
    return np.linspace(min_t, max_t, num=num_points).reshape(-1, 1)

# --- MAIN APPLICATION ---
st.title("BiLSTM-GRB: Advanced Adaptive Reconstruction")
st.markdown("""
Connect to the **Neil Gehrels Swift Observatory** for real-time X-ray afterglow data.
The Bi-LSTM architecture automatically adapts to data density for optimal reconstruction.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Mission Control Panel")

st.sidebar.subheader("1. Target Selection")
selected_name = st.sidebar.selectbox("Select GRB Target", list(GRB_CATALOG.keys()))
target_id = GRB_CATALOG[selected_name]
st.sidebar.caption(f"Swift Target ID: {target_id}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Model Configuration")

model_mode = st.sidebar.radio("Architecture Mode", 
                              ["Automatic (Adaptive)", "Manual Configuration"])

manual_config = None
use_dropout = False
manual_epochs = None

if model_mode == "Manual Configuration":
    arch_type = st.sidebar.selectbox("Architecture Type", 
                                     ["Lightweight", "Standard", "Deep", "Custom"])
    
    if arch_type == "Lightweight":
        layers = [32, 32]
        manual_epochs = 120
        batch_size = 2
    elif arch_type == "Standard":
        layers = [64, 64, 32]
        manual_epochs = 80
        batch_size = 4
    elif arch_type == "Deep":
        layers = [128, 128, 64, 64]
        manual_epochs = 50
        batch_size = 8
    else:
        num_layers = st.sidebar.slider("Number of Layers", 2, 5, 3)
        layers = []
        for i in range(num_layers):
            units = st.sidebar.slider(f"Layer {i+1} Units", 16, 256, 64, step=16)
            layers.append(units)
        manual_epochs = st.sidebar.slider("Training Epochs", 20, 200, 80, step=10)
        batch_size = st.sidebar.slider("Batch Size", 1, 16, 4)
    
    manual_config = (arch_type, layers, manual_epochs, batch_size)
    use_dropout = st.sidebar.checkbox("Use Dropout Regularization", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Training Parameters")

learning_rate = st.sidebar.select_slider("Learning Rate", 
                                         options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                         value=0.001)
st.session_state['learning_rate'] = learning_rate

epoch_override = st.sidebar.checkbox("Override Epoch Count", value=False)
custom_epochs = None
if epoch_override:
    custom_epochs = st.sidebar.slider("Number of Epochs", 10, 300, 80, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("4. Reconstruction Settings")

resolution_factor = st.sidebar.slider("Reconstruction Resolution", 
                                      min_value=1.0, max_value=5.0, 
                                      value=3.0, step=0.5,
                                      help="Higher values = smoother curves")

time_filter_min = st.sidebar.number_input("Minimum Time (seconds)", 
                                          value=10, min_value=1, 
                                          help="Filter out early trigger noise")

st.sidebar.markdown("---")
st.sidebar.subheader("5. Visualization")

show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=False)
plot_log_scale = st.sidebar.checkbox("Use Log-Log Scale", value=True)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Fetch & Reconstruct", type="primary", use_container_width=True)

# Architecture Details Expander
with st.sidebar.expander("Architecture & Model Details"):
    st.markdown("""
    **Bidirectional LSTM Architecture:**
    - Processes temporal data in both forward and backward directions
    - Captures long-term dependencies in light curve evolution
    - He Normal initialization for stable gradient flow
    
    **Adaptive Scaling Logic:**
    - **< 80 points**: Lightweight (2 layers, 32 units)
    - **80-300 points**: Standard (3 layers, 64-32 units)
    - **> 300 points**: Deep (4 layers, 128-64 units)
    
    **Training Strategy:**
    - Optimizer: Adam with configurable learning rate
    - Loss Function: Mean Squared Error (MSE)
    - Input: Log-transformed time series (normalized)
    - Output: Log-transformed flux predictions
    
    **Preprocessing Pipeline:**
    1. Log10 transformation (handle power-law decay)
    2. MinMax scaling to [0,1] range
    3. Sequence reshaping for LSTM input
    4. Inverse transform for physical units
    
    **Data Quality Filters:**
    - Removes negative/zero flux values
    - Filters early trigger noise (< 10s default)
    - Sorts chronologically
    - Handles QDP format variations
    """)

with st.sidebar.expander("About Swift-XRT Data"):
    st.markdown("""
    **Neil Gehrels Swift Observatory:**
    - Launched: November 2004
    - Primary Mission: Rapid GRB detection
    - X-Ray Telescope (XRT): 0.2-10 keV range
    - Typical response time: < 100 seconds
    
    **Data Source:**
    - UK Swift Science Data Centre (UKSSDC)
    - Automated light curve generation
    - QDP format (Quick Data Plot)
    - Updated within hours of observation
    
    **Flux Units:**
    - erg/cm²/s (0.3-10 keV band)
    - Logarithmic scale typical for GRBs
    - Error bars include Poisson statistics
    """)

with st.sidebar.expander("GRB Classification Guide"):
    st.markdown("""
    **Long GRBs (> 2 seconds):**
    - Associated with massive star collapse
    - Strong afterglow emission
    - Examples: GRB 130427A, GRB 080319B
    
    **Short GRBs (< 2 seconds):**
    - Neutron star mergers
    - Weaker afterglows
    - Examples: GRB 050509B, GRB 090510
    
    **Ultra-Long GRBs (> 1000 seconds):**
    - Exotic progenitors (blue supergiants?)
    - Example: GRB 111209A
    
    **Key Observables:**
    - Peak flux and fluence
    - Temporal decay index
    - Spectral hardness
    - Redshift (distance)
    """)


# --- MAIN EXECUTION ---
if run_btn:
    with st.spinner(f"Contacting Swift-XRT Repository for {selected_name.split('(')[0].strip()}..."):
        raw_data, error_msg = fetch_swift_data(target_id)
        
    if error_msg:
        st.error(f"Error: {error_msg}")
    else:
        st.success(f"Connection Established! Downloaded {len(raw_data)} data points.")
        
        # DATA PREPROCESSING
        ts = raw_data['t'].values
        fluxes = raw_data['flux'].values
        flux_errs = raw_data['flux_err'].values
        
        # Apply time filter
        mask = ts >= time_filter_min
        ts = ts[mask]
        fluxes = fluxes[mask]
        flux_errs = flux_errs[mask]
        
        log_ts = np.log10(ts)
        log_fluxes = np.log10(fluxes)
        
        X = log_ts.reshape(-1, 1)
        y = log_fluxes.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
        
        # BUILD & TRAIN
        model, epochs, batch = build_adaptive_model(len(ts), manual_config, use_dropout)
        
        # Apply epoch override if set
        if custom_epochs is not None:
            epochs = custom_epochs
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.6f}")

        with st.spinner("Training BiLSTM on burst-specific physics..."):
            history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=batch, 
                              verbose=0, callbacks=[ProgressBarCallback()])
        
        progress_bar.empty()
        status_text.empty()
        
        # INFERENCE
        recon_log_t = create_reconstruction_points(log_ts, resolution_factor)
        recon_scaled = scaler_X.transform(recon_log_t)
        recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
        
        predictions_scaled = model.predict(recon_seq, verbose=0)
        predictions_log = scaler_y.inverse_transform(predictions_scaled)
        
        # Calculate residuals and statistics
        train_predictions_scaled = model.predict(X_seq, verbose=0)
        train_predictions_log = scaler_y.inverse_transform(train_predictions_scaled)
        residuals = log_fluxes - train_predictions_log.flatten()
        
        # Convert back to linear scale for some analyses
        recon_t = 10**recon_log_t.flatten()
        recon_flux = 10**predictions_log.flatten()
        
        # VISUALIZATION
        st.subheader(f"Temporal Reconstruction: {selected_name.split('(')[0].strip()}")
        
        # PLOT 1: Main Light Curve
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        
        ax1.errorbar(log_ts, log_fluxes, yerr=flux_errs/(fluxes*np.log(10)), 
                   fmt='o', color='gray', alpha=0.4, markersize=5, 
                   label='Swift-XRT Observed Data', capsize=3)
        
        ax1.plot(recon_log_t, predictions_log, color='#FF4B4B', linewidth=3, 
               label='BiLSTM Reconstruction', zorder=5)
        
        if show_confidence:
            std_residual = np.std(residuals)
            ax1.fill_between(recon_log_t.flatten(), 
                           predictions_log.flatten() - 2*std_residual,
                           predictions_log.flatten() + 2*std_residual,
                           color='#FF4B4B', alpha=0.2, label='95% Confidence Interval')
        
        ax1.set_xlabel("log(Time) [seconds]", fontsize=12, fontweight='bold')
        ax1.set_ylabel("log(Flux) [erg/cm²/s]", fontsize=12, fontweight='bold')
        ax1.set_title(f"X-ray Light Curve: {selected_name}", fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig1)
        plt.close(fig1)
        
        # METRICS DASHBOARD
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Data Points", f"{len(ts)}")
        col2.metric("Final Loss", f"{history.history['loss'][-1]:.6f}")
        col3.metric("Flux Range", f"10^{np.ptp(log_fluxes):.2f}")
        col4.metric("Time Span", f"{ts[-1]/ts[0]:.1f}x")
        
        # ADDITIONAL PLOTS
        st.markdown("### Comprehensive Analysis")
        
        # PLOT 2: Residual Analysis
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Time
        ax2a.scatter(log_ts, residuals, color='#0068C9', alpha=0.6, s=50)
        ax2a.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        ax2a.axhline(y=np.mean(residuals), color='orange', linestyle='--', linewidth=1.5, 
                    label=f'Mean: {np.mean(residuals):.4f}')
        ax2a.fill_between(log_ts, -2*np.std(residuals), 2*np.std(residuals), 
                         alpha=0.2, color='gray', label='±2σ Band')
        ax2a.set_xlabel("log(Time) [seconds]", fontweight='bold')
        ax2a.set_ylabel("Residual [log(Flux)]", fontweight='bold')
        ax2a.set_title("Residual Distribution Over Time", fontweight='bold')
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
        
        # Residual Histogram
        ax2b.hist(residuals, bins=30, color='#0068C9', alpha=0.7, edgecolor='black')
        ax2b.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2b.axvline(x=np.mean(residuals), color='orange', linestyle='--', linewidth=1.5)
        ax2b.set_xlabel("Residual Value", fontweight='bold')
        ax2b.set_ylabel("Frequency", fontweight='bold')
        ax2b.set_title(f"Residual Histogram (σ={np.std(residuals):.4f})", fontweight='bold')
        ax2b.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig2)
        plt.close(fig2)
        
        # PLOT 3: Linear Scale Comparison
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        
        ax3.errorbar(ts, fluxes, yerr=flux_errs, fmt='o', color='gray', 
                    alpha=0.4, markersize=5, label='Observed Data', capsize=3)
        ax3.plot(recon_t, recon_flux, color='#FF4B4B', linewidth=3, 
                label='BiLSTM Reconstruction')
        ax3.set_xlabel("Time [seconds]", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Flux [erg/cm²/s]", fontsize=12, fontweight='bold')
        ax3.set_title("Light Curve - Linear Scale", fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3, which='both')
        
        st.pyplot(fig3)
        plt.close(fig3)
        
        # PLOT 4: Temporal Decay Analysis
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        
        # Calculate decay indices (local slopes)
        window_size = max(3, len(recon_log_t) // 50)
        decay_indices = []
        decay_times = []
        
        for i in range(window_size, len(recon_log_t) - window_size):
            t_window = recon_log_t[i-window_size:i+window_size].flatten()
            f_window = predictions_log[i-window_size:i+window_size].flatten()
            slope, _ = np.polyfit(t_window, f_window, 1)
            decay_indices.append(slope)
            decay_times.append(recon_log_t[i])
        
        ax4.plot(decay_times, decay_indices, color='#00C853', linewidth=2.5)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.set_xlabel("log(Time) [seconds]", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Temporal Decay Index (d log F / d log t)", fontsize=12, fontweight='bold')
        ax4.set_title("Evolution of Temporal Decay Index", fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add reference lines for common decay laws
        ax4.axhline(y=-1.0, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='Shallow decay (α=-1)')
        ax4.axhline(y=-1.5, color='purple', linestyle=':', linewidth=1.5, alpha=0.5, label='Normal decay (α=-1.5)')
        ax4.legend(loc='best')
        
        st.pyplot(fig4)
        plt.close(fig4)
        
        # PLOT 5: Prediction Confidence Analysis
        fig5, ax5 = plt.subplots(figsize=(14, 6))
        
        # Calculate point-wise uncertainty based on local residual variance
        uncertainty_window = max(5, len(log_ts) // 20)
        local_uncertainties = []
        
        for i, t_pred in enumerate(recon_log_t):
            # Find nearest training points
            distances = np.abs(log_ts - t_pred)
            nearest_idx = np.argsort(distances)[:uncertainty_window]
            local_residuals = residuals[nearest_idx]
            local_uncertainties.append(np.std(local_residuals))
        
        local_uncertainties = np.array(local_uncertainties)
        
        ax5.plot(recon_log_t, predictions_log, color='#FF4B4B', linewidth=2.5, 
                label='Reconstruction', zorder=5)
        ax5.fill_between(recon_log_t.flatten(),
                        predictions_log.flatten() - 2*local_uncertainties,
                        predictions_log.flatten() + 2*local_uncertainties,
                        color='#FF4B4B', alpha=0.3, label='Local 2σ Uncertainty')
        ax5.scatter(log_ts, log_fluxes, color='gray', alpha=0.5, s=30, 
                   label='Training Data', zorder=3)
        ax5.set_xlabel("log(Time) [seconds]", fontsize=12, fontweight='bold')
        ax5.set_ylabel("log(Flux) [erg/cm²/s]", fontsize=12, fontweight='bold')
        ax5.set_title("Reconstruction with Local Uncertainty Estimation", fontsize=14, fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        st.pyplot(fig5)
        plt.close(fig5)
        
        # PLOT 6: Training Loss Evolution (Enhanced)
        fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax6a.plot(history.history['loss'], color='#0068C9', linewidth=2.5)
        ax6a.set_xlabel("Epoch", fontweight='bold')
        ax6a.set_ylabel("Loss (MSE)", fontweight='bold')
        ax6a.set_title("Training Loss Evolution", fontweight='bold')
        ax6a.set_yscale('log')
        ax6a.grid(True, alpha=0.3)
        
        # Loss improvement rate
        loss_values = np.array(history.history['loss'])
        improvement_rate = -np.diff(loss_values) / loss_values[:-1] * 100
        ax6b.plot(range(1, len(improvement_rate)+1), improvement_rate, 
                 color='#00C853', linewidth=2)
        ax6b.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax6b.set_xlabel("Epoch", fontweight='bold')
        ax6b.set_ylabel("Loss Improvement Rate [%]", fontweight='bold')
        ax6b.set_title("Learning Rate Effectiveness", fontweight='bold')
        ax6b.grid(True, alpha=0.3)
        
        st.pyplot(fig6)
        plt.close(fig6)
        
        # COMPREHENSIVE STATISTICAL ANALYSIS
        st.markdown("### Detailed Statistical Analysis")
        
        with st.expander("Model Performance Statistics", expanded=False):
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown("**Residual Statistics**")
                st.write(f"Mean Residual: {np.mean(residuals):.6f}")
                st.write(f"Std Deviation: {np.std(residuals):.6f}")
                st.write(f"Max Absolute Error: {np.max(np.abs(residuals)):.6f}")
                st.write(f"RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")
                
            with col_stat2:
                st.markdown("**Training Metrics**")
                st.write(f"Initial Loss: {history.history['loss'][0]:.6f}")
                st.write(f"Final Loss: {history.history['loss'][-1]:.6f}")
                st.write(f"Total Improvement: {(1 - history.history['loss'][-1]/history.history['loss'][0])*100:.2f}%")
                st.write(f"Epochs Trained: {len(history.history['loss'])}")
                
            with col_stat3:
                st.markdown("**Data Characteristics**")
                st.write(f"Time Range: {ts[0]:.2e} - {ts[-1]:.2e} s")
                st.write(f"Flux Range: {fluxes.min():.2e} - {fluxes.max():.2e}")
                st.write(f"Mean Flux Error: {np.mean(flux_errs):.2e}")
                st.write(f"Median SNR: {np.median(fluxes/flux_errs):.2f}")
        
        with st.expander("Temporal Decay Analysis", expanded=False):
            st.markdown("**Power-Law Decay Fitting**")
            
            # Fit power law to reconstruction: F(t) = F0 * t^(-alpha)
            log_params = np.polyfit(recon_log_t.flatten(), predictions_log.flatten(), 1)
            decay_index = log_params[0]
            normalization = 10**log_params[1]
            
            col_decay1, col_decay2 = st.columns(2)
            
            with col_decay1:
                st.write(f"**Temporal Decay Index (α):** {decay_index:.3f}")
                st.write(f"**Normalization Constant:** {normalization:.3e} erg/cm²/s")
                
                # Classify decay phase
                if decay_index > -0.5:
                    phase = "Plateau/Rising Phase"
                elif -1.5 < decay_index <= -0.5:
                    phase = "Shallow Decay Phase"
                elif -3.0 < decay_index <= -1.5:
                    phase = "Normal Decay Phase"
                else:
                    phase = "Steep Decay Phase"
                
                st.write(f"**Decay Phase Classification:** {phase}")
            
            with col_decay2:
                st.markdown("**Reference Decay Indices:**")
                st.write("- Plateau: α ≈ 0 to -0.5")
                st.write("- Shallow: α ≈ -0.5 to -1.0")
                st.write("- Normal: α ≈ -1.2 to -1.5")
                st.write("- Steep: α < -2.0")
        
        with st.expander("Physical Interpretation", expanded=False):
            st.markdown("**GRB Afterglow Physics**")
            
            # Calculate characteristic timescales
            flux_half = (fluxes[0] + fluxes[-1]) / 2
            t_half_idx = np.argmin(np.abs(fluxes - flux_half))
            t_half = ts[t_half_idx]
            
            col_phys1, col_phys2 = st.columns(2)
            
            with col_phys1:
                st.write(f"**Peak Flux:** {fluxes.max():.3e} erg/cm²/s")
                st.write(f"**Time to Half-Max:** {t_half:.2e} seconds")
                st.write(f"**Flux Decay Factor:** {fluxes.max()/fluxes.min():.1f}x")
                st.write(f"**Observational Duration:** {(ts[-1]-ts[0])/3600:.2f} hours")
            
            with col_phys2:
                st.markdown("**Energy Estimates (0.3-10 keV):**")
                # Rough fluence calculation (trapezoidal integration)
                fluence = np.trapz(recon_flux, recon_t)
                st.write(f"**Estimated Fluence:** {fluence:.3e} erg/cm²")
                st.write(f"**Mean Flux:** {np.mean(recon_flux):.3e} erg/cm²/s")
                st.write(f"**Peak/Mean Ratio:** {fluxes.max()/np.mean(fluxes):.2f}")
        
        with st.expander("Model Architecture Summary", expanded=False):
            st.markdown("**Neural Network Configuration**")
            
            # Get model summary
            model_config = []
            for i, layer in enumerate(model.layers):
                layer_type = layer.__class__.__name__
                if hasattr(layer, 'units'):
                    params = f"{layer.units} units"
                elif hasattr(layer, 'rate'):
                    params = f"rate={layer.rate}"
                else:
                    params = "N/A"
                model_config.append(f"Layer {i+1}: {layer_type} ({params})")
            
            for config_line in model_config:
                st.write(config_line)
            
            total_params = model.count_params()
            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Parameters per Data Point:** {total_params/len(ts):.2f}")
        
        with st.expander("Uncertainty Quantification", expanded=False):
            st.markdown("**Prediction Uncertainty Analysis**")
            
            col_unc1, col_unc2 = st.columns(2)
            
            with col_unc1:
                st.write(f"**Mean Local Uncertainty:** {np.mean(local_uncertainties):.6f}")
                st.write(f"**Max Local Uncertainty:** {np.max(local_uncertainties):.6f}")
                st.write(f"**Min Local Uncertainty:** {np.min(local_uncertainties):.6f}")
                
            with col_unc2:
                # Identify high uncertainty regions
                high_unc_threshold = np.mean(local_uncertainties) + 2*np.std(local_uncertainties)
                high_unc_regions = np.sum(local_uncertainties > high_unc_threshold)
                st.write(f"**High Uncertainty Points:** {high_unc_regions} / {len(local_uncertainties)}")
                st.write(f"**Uncertainty Range:** {np.ptp(local_uncertainties):.6f}")
                st.write(f"**Coefficient of Variation:** {np.std(local_uncertainties)/np.mean(local_uncertainties):.3f}")

        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export reconstruction data
            df_export = pd.DataFrame({
                'log_time_seconds': recon_log_t.flatten(),
                'time_seconds': 10**recon_log_t.flatten(),
                'reconstructed_log_flux': predictions_log.flatten(),
                'reconstructed_flux': 10**predictions_log.flatten(),
                'local_uncertainty': local_uncertainties
            })
            
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download Reconstruction (CSV)",
                data=csv,
                file_name=f"{selected_name.split('(')[0].strip().replace(' ', '_')}_reconstruction.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Export analysis summary
            summary_data = {
                'Metric': [
                    'Data Points', 'Final Loss', 'RMSE', 'Decay Index',
                    'Peak Flux', 'Time to Half-Max', 'Total Parameters',
                    'Mean Uncertainty'
                ],
                'Value': [
                    len(ts), history.history['loss'][-1], np.sqrt(np.mean(residuals**2)),
                    decay_index, fluxes.max(), t_half, model.count_params(),
                    np.mean(local_uncertainties)
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            csv_summary = df_summary.to_csv(index=False)
            
            st.download_button(
                label="Download Analysis Summary (CSV)",
                data=csv_summary,
                file_name=f"{selected_name.split('(')[0].strip().replace(' ', '_')}_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # ADDITIONAL INFO
        with st.expander("About This Reconstruction"):
            st.markdown(f"""
            **Burst Information:**
            - **Name:** {selected_name.split('(')[0].strip()}
            - **Swift Target ID:** {target_id}
            - **Classification:** {selected_name.split('(')[1].strip(')')}
            
            **Model Configuration:**
            - **Architecture:** {'Manual' if manual_config else 'Adaptive'}
            - **Training Epochs:** {epochs}
            - **Batch Size:** {batch}
            - **Learning Rate:** {learning_rate}
            - **Dropout:** {'Enabled' if use_dropout else 'Disabled'}
            
            **Data Statistics:**
            - **Observed Points:** {len(ts)}
            - **Reconstructed Points:** {len(recon_log_t)}
            - **Time Range:** {ts[0]:.1f} - {ts[-1]:.1f} seconds
            - **Flux Range:** {fluxes.min():.2e} - {fluxes.max():.2e} erg/cm²/s
            """)

else:
    st.info("Select a GRB from the sidebar and configure parameters, then click 'Fetch & Reconstruct' to begin analysis.")
    
    st.markdown("### Available GRB Catalog")
    catalog_df = pd.DataFrame([
        {"GRB Name": name.split('(')[0].strip(), 
         "Notable Feature": name.split('(')[1].strip(')'),
         "Target ID": tid}
        for name, tid in GRB_CATALOG.items()
    ])
    st.dataframe(catalog_df, use_container_width=True)
