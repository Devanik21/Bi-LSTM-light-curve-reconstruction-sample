import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats as st_scipy
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, 
                                     Input, Conv1D, UpSampling1D, Concatenate,
                                     MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras import initializers
from scipy.interpolate import UnivariateSpline
import os

# --- CONFIGURATION & SEEDS ---
st.set_page_config(page_title="Multi-Model GRB Reconstructor", layout="wide")
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

# --- MODEL BUILDERS ---

def build_mlp_model(X_train, y_train):
    """Multilayer Perceptron - Best overall MSE (0.0275)"""
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=seed_value,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train.ravel())
    return mlp

def build_bilstm_model(n_data_points, use_dropout=False):
    """Bidirectional LSTM - Traditional sequence model"""
    he_init = initializers.HeNormal()
    model = Sequential()
    
    if n_data_points < 80:
        units = [32, 32]
        epochs = 120
        batch_size = 2
    elif n_data_points < 300:
        units = [64, 64, 32]
        epochs = 80
        batch_size = 4
    else:
        units = [128, 128, 64, 64]
        epochs = 50
        batch_size = 8
    
    for i, unit in enumerate(units):
        if i < len(units) - 1:
            model.add(Bidirectional(LSTM(unit, kernel_initializer=he_init, 
                                         return_sequences=True), 
                                   input_shape=(1, 1) if i == 0 else None))
            if use_dropout:
                model.add(Dropout(0.2))
        else:
            model.add(Bidirectional(LSTM(unit, kernel_initializer=he_init)))
    
    model.add(Dense(1, activation='linear'))
    
    learning_rate = st.session_state.get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model, epochs, batch_size

def build_attention_unet_model(input_shape):
    """Attention U-Net - Lowest parameter uncertainties (37.9%, 38.5%, 41.4%)"""
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
    
    # Attention mechanism
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(conv1, conv1)
    attention = LayerNormalization()(attention)
    attention = Add()([conv1, attention])
    
    # Decoder
    up = UpSampling1D(size=1)(attention)
    conv2 = Conv1D(32, 3, activation='relu', padding='same')(up)
    conv2 = Conv1D(32, 3, activation='relu', padding='same')(conv2)
    
    outputs = Conv1D(1, 1, activation='linear')(conv2)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

def build_mamba_model(input_shape):
    """Bi-Mamba inspired architecture - State space model alternative"""
    inputs = Input(shape=input_shape)
    
    # Simulated Mamba-like layers using Conv1D with different dilations
    x = Conv1D(64, 3, dilation_rate=1, activation='relu', padding='same')(inputs)
    x = Conv1D(64, 3, dilation_rate=2, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, dilation_rate=4, activation='relu', padding='same')(x)
    
    # Bidirectional processing
    forward = Conv1D(32, 1, activation='relu')(x)
    backward = Conv1D(32, 1, activation='relu')(x[:, ::-1, :])
    
    merged = Concatenate()([forward, backward[:, ::-1, :]])
    outputs = Conv1D(1, 1, activation='linear')(merged)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# --- HELPER FUNCTIONS ---
def create_reconstruction_points(log_ts, resolution_factor):
    """Create smooth timeline with adjustable resolution."""
    min_t, max_t = log_ts.min(), log_ts.max()
    num_points = int(len(log_ts) * resolution_factor)
    return np.linspace(min_t, max_t, num=num_points).reshape(-1, 1)

# --- MAIN APPLICATION ---
st.title("Multi-Model GRB Light Curve Reconstructor")
st.markdown("""
Advanced reconstruction using four state-of-the-art models based on research comparing 9 methods over 521 GRBs.
Connect to **Neil Gehrels Swift Observatory** for real-time X-ray afterglow analysis.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Mission Control Panel")

st.sidebar.subheader("1. Target Selection")
selected_name = st.sidebar.selectbox("Select GRB Target", list(GRB_CATALOG.keys()))
target_id = GRB_CATALOG[selected_name]
st.sidebar.caption(f"Swift Target ID: {target_id}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Model Selection")

st.sidebar.markdown("""
**Performance Benchmarks (521 GRBs):**
- **MLP**: MSE = 0.0275 (Best)
- **Attention U-Net**: 37.9% uncertainty reduction
- **BiLSTM**: Traditional sequence model
- **Bi-Mamba**: State-space alternative
""")

model_options = {
    "MLP (Best MSE: 0.0275)": "mlp",
    "Attention U-Net (Lowest Uncertainty: 37.9%)": "attention_unet",
    "BiLSTM (Sequence Model)": "bilstm",
    "Bi-Mamba (State Space)": "mamba",
    "Compare All Models": "all"
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_model_name]

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
                                      value=3.0, step=0.5)

time_filter_min = st.sidebar.number_input("Minimum Time (seconds)", 
                                          value=10, min_value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("5. Visualization")

show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("6. Paper Implementations")
paper_model_select = st.sidebar.selectbox("Select Paper Model", 
                                          ["None", 
                                           "Model 1: Quadratic Smoothing Spline (GRB 231210B)",
                                           "Model 2: Coming Soon",
                                           "Model 3: Coming Soon"])

run_paper_btn = False
if paper_model_select == "Model 1: Quadratic Smoothing Spline (GRB 231210B)":
    dataset_url = st.sidebar.text_input("Dataset URL (GitHub Raw)", 
        value="https://raw.githubusercontent.com/username/repo/main/GRB%20Data/GRB231210B_trimmed.csv")
    run_paper_btn = st.sidebar.button("Run Paper Model 1", type="primary")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Fetch & Reconstruct", type="primary", use_container_width=True)

# Detailed expanders
with st.sidebar.expander("Model Architecture Details"):
    st.markdown("""
    **MLP (Multilayer Perceptron):**
    - 3 hidden layers: 128-64-32 neurons
    - ReLU activation, Adam optimizer
    - Early stopping with validation
    - Best overall MSE: 0.0275
    - Uncertainty reduction: 37.2%-41.2%
    
    **Attention U-Net:**
    - Multi-head self-attention mechanism
    - U-Net encoder-decoder architecture
    - Best uncertainty reduction: 37.9%-41.4%
    - MSE: 0.134
    
    **BiLSTM:**
    - Bidirectional LSTM layers
    - Adaptive depth based on data size
    - He Normal initialization
    - Captures temporal dependencies
    
    **Bi-Mamba:**
    - State-space model inspired
    - Dilated convolutions (rates: 1,2,4)
    - Bidirectional processing
    - Efficient long-range modeling
    """)

with st.sidebar.expander("Research Context"):
    st.markdown("""
    **Study Overview:**
    - 9 models compared over 521 GRBs
    - Benchmark: Willingale 2007 (W07)
    - Focus: Plateau parameter uncertainties
    
    **Key Findings:**
    - MLP achieves lowest MSE (0.0275)
    - Attention U-Net: best uncertainty reduction
    - Critical for GRB standard candles
    - Enables redshift prediction via ML
    
    **Other Models Tested:**
    - Fourier Transform
    - Gaussian Process-Random Forest
    - cGAN
    - SARIMAX-Kalman Filter
    - Kolmogorov-Arnold Networks (KAN)
    
    **MSE Range:** 0.0339 - 0.174
    """)

with st.sidebar.expander("About Swift-XRT Data"):
    st.markdown("""
    **Neil Gehrels Swift Observatory:**
    - Launched: November 2004
    - X-Ray Telescope: 0.2-10 keV
    - Response time: < 100 seconds
    
    **Data Source:**
    - UK Swift Science Data Centre
    - Flux: erg/cm²/s (0.3-10 keV)
    - QDP format with error bars
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
        
        # Reconstruction points
        recon_log_t = create_reconstruction_points(log_ts, resolution_factor)
        recon_scaled = scaler_X.transform(recon_log_t)
        
        # Dictionary to store results
        results = {}
        
        # TRAIN SELECTED MODELS
        if selected_model == "all":
            models_to_train = ["mlp", "attention_unet", "bilstm", "mamba"]
        else:
            models_to_train = [selected_model]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, model_type in enumerate(models_to_train):
            status_text.text(f"Training {model_type.upper()}...")
            
            if model_type == "mlp":
                model = build_mlp_model(X_scaled, y_scaled)
                predictions_scaled = model.predict(recon_scaled).reshape(-1, 1)
                predictions_log = scaler_y.inverse_transform(predictions_scaled)
                train_predictions = model.predict(X_scaled).reshape(-1, 1)
                train_predictions_log = scaler_y.inverse_transform(train_predictions)
                mse = np.mean((y_scaled.ravel() - model.predict(X_scaled))**2)
                
            elif model_type == "bilstm":
                X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
                model, epochs, batch = build_bilstm_model(len(ts), use_dropout)
                
                if custom_epochs:
                    epochs = custom_epochs
                
                history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=batch, 
                                  verbose=0)
                
                recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
                predictions_scaled = model.predict(recon_seq, verbose=0)
                predictions_log = scaler_y.inverse_transform(predictions_scaled)
                train_predictions_scaled = model.predict(X_seq, verbose=0)
                train_predictions_log = scaler_y.inverse_transform(train_predictions_scaled)
                mse = history.history['loss'][-1]
                
            elif model_type == "attention_unet":
                X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
                model = build_attention_unet_model((1, 1))
                
                epochs = custom_epochs if custom_epochs else 100
                history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=4, 
                                  verbose=0)
                
                recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
                predictions_scaled = model.predict(recon_seq, verbose=0)
                predictions_log = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
                train_predictions_scaled = model.predict(X_seq, verbose=0)
                train_predictions_log = scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1))
                mse = history.history['loss'][-1]
                
            elif model_type == "mamba":
                X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
                model = build_mamba_model((1, 1))
                
                epochs = custom_epochs if custom_epochs else 100
                history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=4, 
                                  verbose=0)
                
                recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
                predictions_scaled = model.predict(recon_seq, verbose=0)
                predictions_log = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
                train_predictions_scaled = model.predict(X_seq, verbose=0)
                train_predictions_log = scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1))
                mse = history.history['loss'][-1]
            
            residuals = log_fluxes - train_predictions_log.flatten()
            
            results[model_type] = {
                'predictions': predictions_log,
                'residuals': residuals,
                'mse': mse,
                'rmse': np.sqrt(np.mean(residuals**2))
            }
            
            progress_bar.progress((idx + 1) / len(models_to_train))
        
        progress_bar.empty()
        status_text.empty()
        
        # VISUALIZATION
        st.subheader(f"Light Curve Reconstruction: {selected_name.split('(')[0].strip()}")
        
        # PLOT 1: Model Comparison
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'serif'
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        fig1.patch.set_facecolor('#0E1117')
        ax1.set_facecolor('#0E1117')
        
        ax1.errorbar(log_ts, log_fluxes, yerr=flux_errs/(fluxes*np.log(10)), 
                   fmt='o', color='#999999', ecolor='#444444', alpha=0.6, markersize=5,
                   label='Swift-XRT Data', capsize=3, zorder=1)
        # Add a glow effect for data points
        ax1.scatter(log_ts, log_fluxes, color='#FFFFFF', s=15, alpha=0.7, zorder=2)
        
        # Space-themed vibrant color palette
        colors = {'mlp': '#00FFFF',          # Cyan
                  'bilstm': '#FF00FF',       # Magenta
                  'attention_unet': '#FFFF00', # Yellow
                  'mamba': '#00FF00'}         # Green
        labels = {'mlp': 'MLP (MSE: 0.0275)', 'bilstm': 'BiLSTM',
                 'attention_unet': 'Attention U-Net', 'mamba': 'Bi-Mamba'}
        
        for model_type, result in results.items():
            ax1.plot(recon_log_t, result['predictions'], 
                    color=colors[model_type], linewidth=2.5,
                    label=labels[model_type], alpha=0.9, zorder=3)
            
            if show_confidence:
                std_resid = np.std(result['residuals'])
                # 95% confidence interval (1.96 * std)
                upper_bound = result['predictions'].flatten() + 1.96 * std_resid
                lower_bound = result['predictions'].flatten() - 1.96 * std_resid
                ax1.fill_between(recon_log_t.flatten(), lower_bound, upper_bound,
                                 color=colors[model_type], alpha=0.15,
                                 label=f'{labels[model_type]} 95% CI', zorder=0)
        
        ax1.set_xlabel("log(Time) [seconds]", fontsize=12, fontweight='bold')
        ax1.set_ylabel("log(Flux) [erg/cm²/s]", fontsize=12, fontweight='bold')
        ax1.set_title(f"Multi-Model Reconstruction: {selected_name}", fontsize=14, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.7)
        ax1.grid(True, alpha=0.5, linestyle='--')
        
        st.pyplot(fig1)
        plt.close(fig1)
        
        # METRICS DASHBOARD
        st.markdown("### Model Performance Comparison")
        
        cols = st.columns(len(results))
        for idx, (model_type, result) in enumerate(results.items()):
            with cols[idx]:
                st.metric(labels[model_type], f"MSE: {result['mse']:.6f}")
                st.caption(f"RMSE: {result['rmse']:.6f}")
        
        # PLOT 2: Residual Comparison
        st.markdown("### Residual Analysis")
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0E1117')
        fig2.patch.set_facecolor('#0E1117')

        axes = axes.flatten()
        
        for idx, (model_type, result) in enumerate(results.items()):
            if idx < 4:
                axes[idx].scatter(log_ts, result['residuals'], 
                                color=colors[model_type], alpha=0.7, s=40, edgecolors='w', linewidths=0.5)
                axes[idx].axhline(y=0, color='#FF4B4B', linestyle='--', linewidth=2)
                axes[idx].axhline(y=np.mean(result['residuals']), color='orange', 
                                linestyle='--', linewidth=1.5)
                axes[idx].fill_between(log_ts, -2*np.std(result['residuals']), 
                                      2*np.std(result['residuals']), 
                                      alpha=0.2, color='#888888')
                axes[idx].set_xlabel("log(Time) [s]", fontweight='bold')
                axes[idx].set_ylabel("Residual", fontweight='bold')
                axes[idx].set_title(f"{labels[model_type]} - σ={np.std(result['residuals']):.4f}", 
                                  fontweight='bold')
                axes[idx].grid(True, alpha=0.5, linestyle='--')
                axes[idx].set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        
        # PLOT 3: MSE Comparison Bar Chart
        st.markdown("### Performance Metrics")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0E1117')
        fig3.patch.set_facecolor('#0E1117')
        
        model_names = [labels[m] for m in results.keys()]
        mse_values = [results[m]['mse'] for m in results.keys()]
        rmse_values = [results[m]['rmse'] for m in results.keys()]
        
        ax3a.bar(model_names, mse_values, color=[colors[m] for m in results.keys()])
        ax3a.set_facecolor('#1a1a1a')
        ax3a.set_ylabel("MSE", fontweight='bold')
        ax3a.set_title("Mean Squared Error Comparison", fontweight='bold')
        ax3a.tick_params(axis='x', rotation=45)
        ax3a.grid(True, alpha=0.5, axis='y', linestyle='--')
        
        ax3b.bar(model_names, rmse_values, color=[colors[m] for m in results.keys()])
        ax3b.set_facecolor('#1a1a1a')
        ax3b.set_ylabel("RMSE", fontweight='bold')
        ax3b.set_title("Root Mean Squared Error Comparison", fontweight='bold')
        ax3b.tick_params(axis='x', rotation=45)
        ax3b.grid(True, alpha=0.5, axis='y', linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        
        # Statistical Analysis
        st.markdown("### Detailed Statistical Analysis")
        
        with st.expander("Model Performance Summary", expanded=True):
            comparison_data = []
            for model_type, result in results.items():
                comparison_data.append({
                    'Model': labels[model_type],
                    'MSE': f"{result['mse']:.6f}",
                    'RMSE': f"{result['rmse']:.6f}",
                    'Mean Residual': f"{np.mean(result['residuals']):.6f}",
                    'Std Residual': f"{np.std(result['residuals']):.6f}",
                    'Max Abs Error': f"{np.max(np.abs(result['residuals'])):.6f}"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            st.markdown("""
            **Benchmark Context (521 GRBs):**
            - MLP: 37.2%-41.2% uncertainty reduction, MSE = 0.0275
            - Attention U-Net: 37.9%-41.4% uncertainty reduction, MSE = 0.134
            - Models improve plateau parameter estimation for GRB standard candles
            """)
        
        # DATA EXPORT
        st.markdown("### Export Data")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export all model predictions
            export_dict = {
                'log_time_seconds': recon_log_t.flatten(),
                'time_seconds': 10**recon_log_t.flatten(),
            }
            for model_type, result in results.items():
                export_dict[f'{model_type}_log_flux'] = result['predictions'].flatten()
                export_dict[f'{model_type}_flux'] = 10**result['predictions'].flatten()
            
            df_export = pd.DataFrame(export_dict)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download All Reconstructions (CSV)",
                data=csv,
                file_name=f"{selected_name.split('(')[0].strip().replace(' ', '_')}_multi_model.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Export comparison summary
            csv_comparison = df_comparison.to_csv(index=False)
            
            st.download_button(
                label="Download Performance Comparison (CSV)",
                data=csv_comparison,
                file_name=f"{selected_name.split('(')[0].strip().replace(' ', '_')}_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

elif run_paper_btn:
    st.subheader("Paper Model 1: Quadratic Smoothing Spline on GRB 231210B")
    
    def train_spline():
        with st.spinner("Loading data and training spline model..."):
            # 1. Load Data
            try:
                # In a real scenario, replace with the actual raw GitHub URL
                # For now, we try to read from the provided URL or fallback to a mock if it fails/is placeholder
                if "username/repo" in dataset_url:
                    st.warning("Using placeholder URL. Please update the Dataset URL in the sidebar to point to the raw CSV.")
                    # Mock data for demonstration if URL is invalid
                    # Creating synthetic GRB 231210B-like data
                    t_mock = np.logspace(1, 5, 50)
                    f_mock = 1e-10 * (t_mock**-1.5) * (1 + 0.1*np.random.randn(50))
                    trimmed_data = pd.DataFrame({'time': t_mock, 'flux': f_mock, 'flux_err': 0.1*f_mock})
                else:
                    response = requests.get(dataset_url)
                    if response.status_code == 200:
                        trimmed_data = pd.read_csv(io.StringIO(response.text))
                    else:
                        st.error(f"Failed to load data from URL: {response.status_code}")
                        return
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return

            grb_name = "GRB231210B"
            folder_path = "." 
            
            # Identify columns
            cols = trimmed_data.columns
            t_col = next((c for c in cols if 'time' in c.lower() or 't' == c.lower()), cols[0])
            f_col = next((c for c in cols if 'flux' in c.lower()), cols[1])
            err_col = next((c for c in cols if 'err' in c.lower()), None)

            t_val = trimmed_data[t_col].values
            f_val = trimmed_data[f_col].values
            
            # Filter valid
            mask = (t_val > 0) & (f_val > 0)
            t_val = t_val[mask]
            f_val = f_val[mask]
            
            train_x_denorm = np.log10(t_val)
            train_y_denorm = np.log10(f_val)
            
            # Errors
            if err_col:
                f_err = trimmed_data[err_col].values[mask]
                # Log error propagation: err_log = err / (f * ln(10))
                y_err_log = f_err / (f_val * np.log(10))
                lower_err_log = y_err_log
                upper_err_log = y_err_log
            else:
                lower_err_log = np.zeros_like(f_val)
                upper_err_log = np.zeros_like(f_val)

            # Spline Fitting
            sort_idx = np.argsort(train_x_denorm)
            x_sorted = train_x_denorm[sort_idx]
            y_sorted = train_y_denorm[sort_idx]
            w_sorted = 1.0 / (lower_err_log[sort_idx] + 1e-9)

            spline = UnivariateSpline(x_sorted, y_sorted, w=w_sorted, k=2)
            
            # Reconstruction
            test_x_denorm = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            mean_prediction_denorm = spline(test_x_denorm)
            
            # Bootstrap for CI
            n_boot = 100
            boot_preds = []
            residuals = y_sorted - spline(x_sorted)
            for _ in range(n_boot):
                y_boot = spline(x_sorted) + np.random.choice(residuals, size=len(y_sorted), replace=True)
                try:
                    sp_boot = UnivariateSpline(x_sorted, y_boot, w=w_sorted, k=2)
                    boot_preds.append(sp_boot(test_x_denorm))
                except: pass
            
            if boot_preds:
                boot_preds = np.array(boot_preds)
                lower_denorm = np.percentile(boot_preds, 2.5, axis=0)
                upper_denorm = np.percentile(boot_preds, 97.5, axis=0)
            else:
                lower_denorm = mean_prediction_denorm - 0.1
                upper_denorm = mean_prediction_denorm + 0.1

            # Synthetic Reconstruction Points
            expanded = test_x_denorm.reshape(-1, 1)
            recon_errorbar = (upper_denorm - lower_denorm) / 4.0
            log_reconstructed_flux = mean_prediction_denorm + np.random.normal(0, recon_errorbar, size=len(mean_prediction_denorm))
            recon_denorm_log = log_reconstructed_flux
            sampled_time_errs = np.abs(test_x_denorm * 0.01)
            sampled_flux_errs = recon_errorbar

            # --- PLOTTING ---
            fig = plt.figure(figsize=(10, 6))
            
            # a) Plot original data with updated y-errors
            plt.errorbar(train_x_denorm, train_y_denorm, zorder=4, yerr=[lower_err_log, upper_err_log], linestyle="")

            # b) Plot reconstructed points with synthetic error bars
            plt.errorbar(test_x_denorm, log_reconstructed_flux, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', zorder=3, label="Reconstructed Points")

            # c) Scatter original observed points on top
            plt.scatter(train_x_denorm, train_y_denorm, zorder=5, label="Observed Points")

            # d) Plot the mean prediction curve
            plt.plot(test_x_denorm, mean_prediction_denorm, label="Mean Prediction", zorder=2)

            # e) Add 95% confidence interval shading
            plt.fill_between(test_x_denorm.flatten(), lower_denorm, upper_denorm, alpha=0.5, color='orange', label="95% Confidence Region", zorder=1)

            plt.legend(loc='lower left')
            plt.xlabel('log$_{10}$(Time) (s)', fontsize=15)
            plt.ylabel('log$_{10}$(Flux) ($erg\\,cm^{-2}\\,s^{-1}$)', fontsize=15)
            plt.title(f'Quadratic Smoothing Spline on {grb_name}', fontsize=18)
            # plt.savefig(f"{folder_path}/Figures/{grb_name}.png", dpi=300) # Disabled for Streamlit
            # print("\n----- RECONSTRUCTED GRB (Figure saved) -----\n")
            # plt.show() # Disabled for Streamlit
            st.pyplot(fig)

            # 11) Build combined DataFrame
            combined_df = trimmed_data.copy(deep=True)
            new_rows = []
            for i in range(len(expanded)):
                logt_pt = expanded[i][0]
                t_lin = 10 ** logt_pt
                pos_t_lin = 10 ** (logt_pt + sampled_time_errs[i])
                neg_t_lin = 10 ** (logt_pt - sampled_time_errs[i])
                flux_lin = 10 ** recon_denorm_log[i]
                pos_f_lin = 10 ** (recon_denorm_log[i] + sampled_flux_errs[i])
                neg_f_lin = 10 ** (recon_denorm_log[i] - sampled_flux_errs[i])
                new_rows.append({
                    "t": t_lin,
                    "pos_t_err": abs(pos_t_lin - t_lin),
                    "neg_t_err": abs(t_lin - neg_t_lin),
                    "flux": flux_lin,
                    "pos_flux_err": abs(pos_f_lin - flux_lin),
                    "neg_flux_err": abs(flux_lin - neg_f_lin)
                })
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([combined_df, new_df], ignore_index=True)
            # combined_df.to_csv(f"{folder_path}/CSV_data/{grb_name}.csv", index=False) # Disabled for Streamlit
            # print(f"Saved combined CSV to: {folder_path}/CSV_data/{grb_name}.csv")
            
            st.success(f"Reconstruction Complete for {grb_name}")
            csv_buffer = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Combined CSV", data=csv_buffer, file_name=f"{grb_name}_spline.csv", mime="text/csv")

    train_spline()

else:
    st.info("Select a GRB and model from the sidebar, then click 'Fetch & Reconstruct' to begin analysis.")
    
    st.markdown("### Available GRB Catalog")
    catalog_df = pd.DataFrame([
        {"GRB Name": name.split('(')[0].strip(), 
         "Notable Feature": name.split('(')[1].strip(')'),
         "Target ID": tid}
        for name, tid in GRB_CATALOG.items()
    ])
    st.dataframe(catalog_df, use_container_width=True)
    
    st.markdown("### Model Selection Guide")
    st.markdown("""
    **Choose based on your research needs:**
    
    | Model | Best For | Key Strength |
    |-------|----------|--------------|
    | **MLP** | Overall accuracy | Lowest MSE (0.0275) |
    | **Attention U-Net** | Uncertainty reduction | 37.9% parameter uncertainty reduction |
    | **BiLSTM** | Temporal patterns | Long-term dependencies |
    | **Bi-Mamba** | Efficiency | State-space modeling |
    | **Compare All** | Comprehensive analysis | Full model comparison |
    
    **Research Context:** Based on study of 9 models over 521 GRBs compared to Willingale 2007 benchmark.
    """)
