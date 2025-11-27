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

if model_mode == "Manual Configuration":
    arch_type = st.sidebar.selectbox("Architecture Type", 
                                     ["Lightweight", "Standard", "Deep", "Custom"])
    
    if arch_type == "Lightweight":
        layers = [32, 32]
        epochs = 120
        batch_size = 2
    elif arch_type == "Standard":
        layers = [64, 64, 32]
        epochs = 80
        batch_size = 4
    elif arch_type == "Deep":
        layers = [128, 128, 64, 64]
        epochs = 50
        batch_size = 8
    else:
        num_layers = st.sidebar.slider("Number of Layers", 2, 5, 3)
        layers = []
        for i in range(num_layers):
            units = st.sidebar.slider(f"Layer {i+1} Units", 16, 256, 64, step=16)
            layers.append(units)
        epochs = st.sidebar.slider("Training Epochs", 20, 200, 80, step=10)
        batch_size = st.sidebar.slider("Batch Size", 1, 16, 4)
    
    manual_config = (arch_type, layers, epochs, batch_size)
    use_dropout = st.sidebar.checkbox("Use Dropout Regularization", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Training Parameters")

learning_rate = st.sidebar.select_slider("Learning Rate", 
                                         options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                                         value=0.001)
st.session_state['learning_rate'] = learning_rate

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
        
        # VISUALIZATION
        st.subheader(f"Temporal Reconstruction: {selected_name.split('(')[0].strip()}")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.errorbar(log_ts, log_fluxes, yerr=flux_errs/(fluxes*np.log(10)), 
                   fmt='o', color='gray', alpha=0.4, markersize=5, 
                   label='Swift-XRT Observed Data', capsize=3)
        
        ax.plot(recon_log_t, predictions_log, color='#FF4B4B', linewidth=3, 
               label='BiLSTM Reconstruction', zorder=5)
        
        if show_confidence:
            residuals = np.interp(log_ts, recon_log_t.flatten(), predictions_log.flatten()) - log_fluxes
            std_residual = np.std(residuals)
            ax.fill_between(recon_log_t.flatten(), 
                           predictions_log.flatten() - 2*std_residual,
                           predictions_log.flatten() + 2*std_residual,
                           color='#FF4B4B', alpha=0.2, label='95% Confidence Interval')
        
        ax.set_xlabel("log(Time) [seconds]", fontsize=12, fontweight='bold')
        ax.set_ylabel("log(Flux) [erg/cm²/s]", fontsize=12, fontweight='bold')
        ax.set_title(f"X-ray Light Curve: {selected_name}", fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        st.pyplot(fig)
        
        # METRICS DASHBOARD
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Data Points", f"{len(ts)}")
        col2.metric("Final Loss", f"{history.history['loss'][-1]:.6f}")
        col3.metric("Flux Range", f"10^{np.ptp(log_fluxes):.2f}")
        col4.metric("Time Span", f"{ts[-1]/ts[0]:.1f}x")
        
        # TRAINING HISTORY
        with st.expander("View Training History"):
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            ax_loss.plot(history.history['loss'], color='#0068C9', linewidth=2)
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss (MSE)")
            ax_loss.set_title("Training Loss Evolution")
            ax_loss.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
        
        # DATA EXPORT
        st.markdown("### Export Reconstruction Data")
        df_export = pd.DataFrame({
            'log_time_seconds': recon_log_t.flatten(),
            'time_seconds': 10**recon_log_t.flatten(),
            'reconstructed_log_flux': predictions_log.flatten(),
            'reconstructed_flux': 10**predictions_log.flatten()
        })
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download Reconstruction (CSV)",
            data=csv,
            file_name=f"{selected_name.split('(')[0].strip().replace(' ', '_')}_reconstruction.csv",
            mime="text/csv"
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
