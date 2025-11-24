import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as st_scipy
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras import initializers

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Streamlit page configuration
st.set_page_config(page_title="BiLSTM Light Curve Reconstruction", layout="wide")

st.title("🌌 BiLSTM Light Curve Reconstruction for GRBs")
st.markdown("""
This app demonstrates Bidirectional LSTM reconstruction of Gamma-Ray Burst (GRB) light curves.
It fills gaps in sparse observational data using deep learning.
""")

# Sidebar parameters
st.sidebar.header("Configuration")
grb_name = st.sidebar.text_input("GRB Name", "GRB050820A")
n_data_points = st.sidebar.slider("Number of Data Points", 20, 200, 50)
noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
epochs = st.sidebar.slider("Training Epochs", 20, 200, 50)

# Generate synthetic GRB light curve data
@st.cache_data
def generate_synthetic_grb_data(n_points=50, noise=0.1):
    """Generate synthetic GRB light curve with power-law decay"""
    np.random.seed(42)
    
    # Time in log scale (simulating GRB observations)
    log_t = np.sort(np.random.uniform(-1, 3, n_points))
    
    # Flux: power-law decay with some plateau
    log_flux = -0.8 * log_t + 2.0 + 0.3 * np.sin(log_t * 2) + np.random.normal(0, noise, n_points)
    
    # Add errors
    pos_t_err = 10**log_t * 0.05
    neg_t_err = 10**log_t * 0.05
    flux = 10**log_flux
    pos_flux_err = flux * 0.15
    neg_flux_err = flux * 0.15
    
    data = pd.DataFrame({
        't': 10**log_t,
        'pos_t_err': pos_t_err,
        'neg_t_err': neg_t_err,
        'flux': flux,
        'pos_flux_err': pos_flux_err,
        'neg_flux_err': neg_flux_err
    })
    
    return data

def create_reconstruction_points(log_ts, min_gap=0.05, fraction=0.2):
    """Create interpolation points in gaps"""
    recon_log_t = [log_ts[0]]
    total_span = log_ts[-1] - log_ts[0]
    n_points = max(20, int(fraction * len(log_ts)))
    
    for i in range(len(log_ts) - 1):
        gap_size = log_ts[i+1] - log_ts[i]
        if gap_size > min_gap:
            interval_points = max(2, int(n_points * gap_size / total_span))
            interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
            recon_log_t.extend(interval[1:])
    
    recon_log_t = np.array(recon_log_t)
    recon_t = 10**np.array(recon_log_t)
    recon_t = np.unique(recon_t)
    return np.log10(recon_t).reshape(-1, 1)

def train_bilstm_model(X_scaled, y_scaled, epochs=50):
    """Train BiLSTM model"""
    he_init = initializers.HeNormal()
    X_seq = X_scaled.reshape(X_scaled.shape[0], 1, 1)
    
    model = Sequential([
        Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True), input_shape=(1, 1)),
        Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)),
        Bidirectional(LSTM(100, kernel_initializer=he_init, return_sequences=True)),
        Bidirectional(LSTM(100, kernel_initializer=he_init)),
        Dense(1, activation='leaky_relu')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    with st.spinner('Training BiLSTM model...'):
        history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=3, verbose=0)
    
    return model, history

# Main execution
if st.sidebar.button("Generate & Reconstruct", type="primary"):
    # Generate data
    with st.spinner("Generating synthetic GRB data..."):
        trimmed_data = generate_synthetic_grb_data(n_data_points, noise_level)
    
    st.success(f"✅ Generated {len(trimmed_data)} data points")
    
    # Process data
    ts = trimmed_data["t"].to_numpy()
    fluxes = trimmed_data["flux"].to_numpy()
    log_ts = np.log10(ts)
    log_fluxes = np.log10(fluxes)
    
    # Calculate errors
    positive_fluxes_err = trimmed_data["pos_flux_err"]
    negative_fluxes_err = trimmed_data["neg_flux_err"]
    pos_fluxes = fluxes + positive_fluxes_err
    neg_fluxes = fluxes - negative_fluxes_err
    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)
    
    # Create reconstruction points
    log_recon_t = create_reconstruction_points(log_ts)
    
    # Prepare data for training
    X = np.array(log_ts).reshape(-1, 1)
    y = np.array(log_fluxes).reshape(-1, 1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Train model
    model, history = train_bilstm_model(X_scaled, y_scaled, epochs)
    
    # Make predictions
    recon_scaled = scaler_X.transform(log_recon_t)
    log_recon_seq = recon_scaled.reshape(recon_scaled.shape[0], 1, 1)
    recon_pred = model.predict(log_recon_seq, verbose=0)
    predictions = scaler_y.inverse_transform(recon_pred).flatten()
    
    # Add noise to reconstructed points
    fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
    logfluxerrs = fluxes_error / (fluxes * np.log(10))
    errparameters = st_scipy.norm.fit(logfluxerrs)
    err_dist = st_scipy.norm(loc=errparameters[0], scale=errparameters[1])
    recon_errorbar = err_dist.rvs(size=len(log_recon_t))
    
    # Generate jiggled realizations
    num_samples = 1000
    jiggled_realizations = []
    for _ in range(num_samples):
        point_specific_noise = []
        for j in range(len(predictions)):
            fitted_dist = norm(loc=predictions[j], scale=np.abs(recon_errorbar[j]))
            point_noise = fitted_dist.rvs() - predictions[j]
            point_specific_noise.append(point_noise)
        jiggled_realizations.append(predictions + np.array(point_specific_noise))
    
    jiggled_realizations = np.array(jiggled_realizations)
    mean_jiggled = np.mean(jiggled_realizations, axis=0)
    ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
    ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)
    
    point_specific_noise = mean_jiggled - predictions
    jiggled_points = predictions + point_specific_noise
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Loss")
        fig_loss, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history['loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Model Training Loss')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
    
    with col2:
        st.subheader("Statistics")
        st.metric("Original Points", len(log_ts))
        st.metric("Reconstructed Points", len(log_recon_t))
        st.metric("Final Loss", f"{history.history['loss'][-1]:.6f}")
    
    # Main reconstruction plot
    st.subheader("Light Curve Reconstruction")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observed data with error bars
    ax.errorbar(log_ts, log_fluxes, 
                yerr=[log_fluxes-neg_log_fluxes, pos_log_fluxes-log_fluxes],
                fmt='o', label='Observed Points', zorder=5, capsize=3)
    
    # Plot reconstructed points
    ax.errorbar(log_recon_t.flatten(), jiggled_points, 
                yerr=np.abs(recon_errorbar), fmt='o', 
                label='Reconstructed Points', color='yellow', 
                alpha=0.7, zorder=3, capsize=3)
    
    # Plot mean predictions
    ax.plot(log_recon_t.flatten(), predictions, 
            label='Mean Predictions', color='red', linewidth=2, zorder=2)
    
    # Plot confidence interval
    ax.fill_between(log_recon_t.flatten(), ci_95_lower, ci_95_upper,
                    color='orange', alpha=0.3, label='95% Confidence Interval', zorder=1)
    
    ax.set_xlabel('log₁₀(Time) (s)', fontsize=13)
    ax.set_ylabel('log₁₀(Flux) (erg cm⁻² s⁻¹)', fontsize=13)
    ax.set_title(f'BiLSTM Reconstruction - {grb_name}', fontsize=15, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Download reconstructed data
    st.subheader("Download Reconstructed Data")
    
    df_reconstructed = pd.DataFrame({
        't': 10**log_recon_t.flatten(),
        'log_t': log_recon_t.flatten(),
        'flux': 10**jiggled_points,
        'log_flux': jiggled_points,
        'flux_error': 10**jiggled_points * np.log(10) * np.abs(recon_errorbar),
        'prediction': 10**predictions,
        'ci_95_lower': 10**ci_95_lower,
        'ci_95_upper': 10**ci_95_upper
    })
    
    csv = df_reconstructed.to_csv(index=False)
    st.download_button(
        label="📥 Download Reconstructed Light Curve (CSV)",
        data=csv,
        file_name=f"{grb_name}_reconstructed.csv",
        mime="text/csv"
    )
    
    st.success("✅ Reconstruction complete!")

else:
    st.info("👈 Configure parameters in the sidebar and click 'Generate & Reconstruct' to start")
    
    # Show example plot
    st.markdown("### Example Output")
    st.image("https://via.placeholder.com/800x400.png?text=Example+GRB+Light+Curve+Reconstruction", 
             caption="Example of BiLSTM reconstruction filling gaps in GRB observations")

st.markdown("---")
st.markdown("""
**About**: This app uses Bidirectional LSTM neural networks to reconstruct gamma-ray burst (GRB) 
light curves by filling observational gaps. The model learns temporal patterns from observed data 
and predicts flux values at unobserved times with uncertainty quantification.
""")
