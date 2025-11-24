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

st.title("Bidirectional LSTM Network for Temporal Reconstruction of Gamma-Ray Burst Light Curves")
st.markdown("""
### Theoretical Framework

Gamma-Ray Bursts represent some of the most energetically violent phenomena in the observable universe, 
with prompt emission typically lasting from milliseconds to several minutes followed by a multi-wavelength 
afterglow phase. The observational challenge lies in the inherently sparse and irregular temporal sampling 
of these transient events due to instrumental limitations, observational cadence constraints, and the 
stochastic nature of photon arrival times at detector arrays.

This application implements a sequential deep learning architecture based on Bidirectional Long Short-Term 
Memory networks to perform temporal interpolation and extrapolation of light curve data. The methodology 
addresses the fundamental problem of reconstructing continuous flux evolution from discrete, noisy 
observational data points with heteroscedastic uncertainties.

**Methodological Approach:**

The reconstruction framework employs a multi-layer bidirectional recurrent neural network architecture 
that learns temporal dependencies in both forward and backward directions through the time series. This 
bidirectional processing enables the model to capture context from both past and future observations 
when inferring flux values at unobserved epochs. The LSTM cells incorporate gating mechanisms (input, 
forget, and output gates) that regulate information flow, allowing the network to learn long-range 
temporal dependencies characteristic of GRB light curve morphology including steep decay phases, 
plateau regions, and potential jet break features.

The training procedure utilizes logarithmically scaled time and flux values to accommodate the 
multi-order-of-magnitude dynamic range typical of GRB observations. Uncertainty quantification is 
achieved through Monte Carlo sampling from learned error distributions, generating an ensemble of 
realizations that approximate the posterior predictive distribution of flux values at interpolated 
time points.
""")

# Sidebar parameters
st.sidebar.header("Model Configuration")
grb_name = st.sidebar.text_input("GRB Designation", "GRB050820A")
n_data_points = st.sidebar.slider("Observational Sample Size", 20, 200, 50)
noise_level = st.sidebar.slider("Photometric Uncertainty Level", 0.01, 0.5, 0.1)
epochs = st.sidebar.slider("Training Iterations", 20, 200, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Network Architecture:**
- 4 Bidirectional LSTM layers
- 100 hidden units per direction
- He Normal weight initialization
- Leaky ReLU output activation
- Adam optimization algorithm
- Mean Squared Error loss function
""")

# Generate synthetic GRB light curve data
@st.cache_data
def generate_synthetic_grb_data(n_points=50, noise=0.1):
    """
    Generate synthetic GRB light curve following empirical temporal decay models.
    
    Implements a composite model incorporating:
    - Power-law decay consistent with synchrotron radiation cooling
    - Sinusoidal modulation representing potential re-brightening episodes
    - Gaussian noise component modeling Poisson photon statistics
    
    Parameters represent log-space values to handle multi-decade temporal and flux ranges
    characteristic of GRB afterglow observations.
    """
    np.random.seed(42)
    
    # Temporal sampling in logarithmic space
    log_t = np.sort(np.random.uniform(-1, 3, n_points))
    
    # Flux evolution model: broken power-law with stochastic component
    log_flux = -0.8 * log_t + 2.0 + 0.3 * np.sin(log_t * 2) + np.random.normal(0, noise, n_points)
    
    # Error propagation from linear to logarithmic space
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
    """
    Adaptive temporal grid generation for interpolation points.
    
    Implements a gap-based allocation strategy where interpolation point density
    is proportional to the temporal span of observational gaps. This ensures
    adequate sampling resolution in sparsely observed regions while avoiding
    redundant interpolation in densely sampled intervals.
    """
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
    """
    Construct and train bidirectional LSTM architecture.
    
    Network topology consists of four stacked bidirectional LSTM layers with 100 units
    per directional pathway, totaling 200 effective hidden units per layer. He Normal
    initialization provides appropriate variance scaling for networks with ReLU-family
    activations. The bidirectional architecture enables the model to learn temporal
    patterns by processing sequences in both forward (past to future) and backward
    (future to past) temporal directions, capturing contextual dependencies that
    unidirectional architectures would miss.
    
    The final dense layer with leaky ReLU activation allows for non-saturating gradients
    and can represent negative flux values in log space corresponding to very low flux
    states in linear space.
    """
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
    
    with st.spinner('Training bidirectional LSTM network architecture...'):
        history = model.fit(X_seq, y_scaled, epochs=epochs, batch_size=3, verbose=0)
    
    return model, history

# Main execution
if st.sidebar.button("Execute Reconstruction Pipeline", type="primary"):
    # Generate data
    with st.spinner("Generating synthetic observational dataset..."):
        trimmed_data = generate_synthetic_grb_data(n_data_points, noise_level)
    
    st.success(f"Dataset generation complete: {len(trimmed_data)} observational epochs")
    
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
    
    # Add noise to reconstructed points through error distribution modeling
    st.info("Performing Monte Carlo uncertainty propagation...")
    
    fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
    logfluxerrs = fluxes_error / (fluxes * np.log(10))
    errparameters = st_scipy.norm.fit(logfluxerrs)
    err_dist = st_scipy.norm(loc=errparameters[0], scale=errparameters[1])
    recon_errorbar = err_dist.rvs(size=len(log_recon_t))
    
    # Generate ensemble of jiggled realizations for uncertainty quantification
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
        st.subheader("Convergence Diagnostics")
        fig_loss, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history['loss'], linewidth=2, color='#2E86AB')
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Mean Squared Error Loss', fontsize=12)
        ax.set_title('Model Convergence Profile', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
    
    with col2:
        st.subheader("Reconstruction Statistics")
        st.metric("Original Observational Points", len(log_ts))
        st.metric("Interpolated Temporal Epochs", len(log_recon_t))
        st.metric("Final Training Loss (MSE)", f"{history.history['loss'][-1]:.6f}")
        st.metric("Reconstruction Density Enhancement", f"{len(log_recon_t)/len(log_ts):.2f}x")
    
    # Main reconstruction plot
    st.subheader("Temporal Flux Evolution Reconstruction")
    st.markdown("""
    The visualization below presents the reconstructed light curve with observational data overlaid.
    Reconstructed flux values are shown with propagated uncertainties derived from the empirical
    error distribution of the training dataset. The 95% confidence interval represents the
    epistemic uncertainty in the model predictions based on Monte Carlo sampling.
    """)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observed data with asymmetric error bars
    ax.errorbar(log_ts, log_fluxes, 
                yerr=[log_fluxes-neg_log_fluxes, pos_log_fluxes-log_fluxes],
                fmt='o', label='Observed Data Points', zorder=5, capsize=3,
                color='#1B4965', markersize=6, linewidth=1.5)
    
    # Plot reconstructed points with uncertainties
    ax.errorbar(log_recon_t.flatten(), jiggled_points, 
                yerr=np.abs(recon_errorbar), fmt='o', 
                label='Reconstructed Flux Estimates', color='#FFB703', 
                alpha=0.7, zorder=3, capsize=3, markersize=5)
    
    # Plot mean model predictions
    ax.plot(log_recon_t.flatten(), predictions, 
            label='Network Mean Prediction', color='#D62828', linewidth=2.5, zorder=2)
    
    # Plot posterior predictive interval
    ax.fill_between(log_recon_t.flatten(), ci_95_lower, ci_95_upper,
                    color='#F77F00', alpha=0.25, label='95% Credible Interval', zorder=1)
    
    ax.set_xlabel('log₁₀(Time) [seconds since trigger]', fontsize=13)
    ax.set_ylabel('log₁₀(Flux) [erg cm⁻² s⁻¹]', fontsize=13)
    ax.set_title(f'Bidirectional LSTM Temporal Reconstruction: {grb_name}', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    st.pyplot(fig)
    
    # Download reconstructed data
    st.subheader("Data Export")
    
    st.markdown("""
    The reconstructed dataset includes temporal coordinates, flux values with associated 
    uncertainties, and statistical confidence bounds derived from the ensemble of Monte Carlo 
    realizations. This data can be used for subsequent temporal analysis, spectral fitting 
    procedures, or comparison with theoretical afterglow models.
    """)
    
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
        label="Download Reconstructed Light Curve Dataset (CSV)",
        data=csv,
        file_name=f"{grb_name}_reconstructed.csv",
        mime="text/csv"
    )
    
    st.success("Reconstruction pipeline execution completed successfully")

else:
    st.info("Configure model parameters in the sidebar panel and execute the reconstruction pipeline to begin analysis")
    
    # Show methodological details
    st.markdown("### Algorithmic Implementation Details")
    
    st.markdown("""
    **Data Preprocessing:**
    
    All temporal and flux measurements undergo logarithmic transformation to normalize the 
    multi-order-of-magnitude dynamic range characteristic of GRB observations. This transformation 
    ensures numerical stability during gradient-based optimization and allows the network to learn 
    relationships across vastly different time scales and flux levels.
    
    **Network Training:**
    
    The model is trained using the Adam optimizer with adaptive learning rate adjustment. The loss 
    function is mean squared error computed in the logarithmically transformed space, which 
    effectively weights relative errors rather than absolute deviations. This is physically motivated 
    as photometric uncertainties scale approximately with flux magnitude.
    
    **Uncertainty Quantification:**
    
    Error propagation follows a two-stage approach:
    
    1. Empirical error distribution is extracted from the observed dataset by fitting a Gaussian 
    model to the logarithmic flux uncertainties.
    
    2. Monte Carlo sampling generates an ensemble of 1000 realizations, each representing a 
    plausible instantiation of the light curve given the model predictions and uncertainty estimates.
    
    The 95% credible interval is derived from the 2.5th and 97.5th percentiles of this ensemble 
    distribution, providing a non-parametric estimate of the posterior predictive uncertainty.
    
    **Temporal Grid Construction:**
    
    Interpolation points are allocated adaptively based on gap size in the observational timeline. 
    Larger temporal gaps receive proportionally more interpolation points, ensuring adequate 
    resolution for capturing potential temporal structure while avoiding computational redundancy 
    in densely sampled regions.
    """)

st.markdown("---")
st.markdown("""
**Implementation Notes:**

This application represents a simplified demonstration framework. Production-level implementations 
would incorporate additional considerations including:

- Cross-validation procedures for hyperparameter optimization
- Ensemble modeling approaches to reduce prediction variance
- Physical constraint enforcement (e.g., flux positivity, causality)
- Multi-wavelength data fusion for improved temporal constraints
- Automated model selection based on light curve morphology classification

**Computational Requirements:**

Network training scales approximately as O(n × m × e) where n represents the number of 
observational data points, m denotes the number of interpolation targets, and e represents 
the number of training epochs. For typical GRB light curves with 50-200 data points, 
convergence is typically achieved within 50-100 epochs on standard CPU architectures.
""")
