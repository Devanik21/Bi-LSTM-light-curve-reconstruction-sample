import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st_scipy
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, ReLU, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# --- PAGE CONFIG ---
st.set_page_config(page_title="GRB Reconstruction - Attention U-Net", layout="wide")

# --- SEED SETTING FOR REPRODUCIBILITY ---
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- 1. MODEL ARCHITECTURE (Attention U-Net) ---
def AttentionBlock1D(x, g, inter_channels):
    theta_x = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(x)
    phi_g = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(g)
    f = ReLU()(theta_x + phi_g)
    psi_f = Conv1D(1, kernel_size=1, strides=1, padding="same", activation="sigmoid")(f)
    return x * psi_f

def UNetWithAttention1D(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)

    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)

    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    pool3 = MaxPooling1D(pool_size=2, padding='same')(conv3)

    # Bottleneck
    bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(bottleneck)

    # Decoder
    upconv3 = UpSampling1D(size=2)(bottleneck)
    # Resize to match conv3 shape if necessary (handle odd dimensions)
    upconv3 = upconv3[:, :conv3.shape[1], :] 
    attention3 = AttentionBlock1D(conv3, upconv3, inter_channels=64)
    concat3 = concatenate([upconv3, attention3], axis=-1)
    conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat3)
    conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec3)

    upconv2 = UpSampling1D(size=2)(conv_dec3)
    upconv2 = upconv2[:, :conv2.shape[1], :] 
    attention2 = AttentionBlock1D(conv2, upconv2, inter_channels=32)
    concat2 = concatenate([upconv2, attention2], axis=-1)
    conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat2)
    conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec2)

    upconv1 = UpSampling1D(size=2)(conv_dec2)
    upconv1 = upconv1[:, :conv1.shape[1], :] 
    attention1 = AttentionBlock1D(conv1, upconv1, inter_channels=16)
    concat1 = concatenate([upconv1, attention1], axis=-1)
    conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(concat1)
    conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(conv_dec1)

    outputs = Conv1D(1, kernel_size=1, activation=None)(conv_dec1)
    outputs = Flatten()(outputs)
    outputs = Dense(input_shape[-1], activation="linear")(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# --- 2. MAIN APP UI ---
st.title("GRB 231210B Reconstruction")
st.markdown("### Attention U-Net Model Deployment")
st.markdown("Reconstructing light curve with uncertainty estimation using Attention U-Net.")

# --- 3. DATA LOADING ---
GRB_Name = "GRB231210B"
try:
    # Try loading the specific file
    header_names = ['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
    # You can change the path below if running locally
    trimmed_data = pd.read_csv("GRB231210B_trimmed.csv", skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
    trimmed_data = trimmed_data.sort_values(by="t").reset_index(drop=True)
    st.success(f"Loaded {GRB_Name} data successfully!")
except FileNotFoundError:
    st.error("File 'GRB231210B_trimmed.csv' not found. Please ensure the CSV is in the same directory.")
    st.stop()

# --- 4. PREPROCESSING ---
ts = trimmed_data["t"].to_numpy()
fluxes = trimmed_data["flux"].to_numpy()
log_ts = np.log10(ts)
log_fluxes = np.log10(fluxes)

positive_fluxes_err = trimmed_data["pos_flux_err"]
negative_fluxes_err = trimmed_data["neg_flux_err"]

# Calculate error bars for plotting (original data)
fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
lower_err_log = log_fluxes - np.log10(fluxes - fluxes_error)
upper_err_log = np.log10(fluxes + fluxes_error) - log_fluxes

# --- 5. GAP HANDLING & RECONSTRUCTION POINTS ---
# Logic from Attention_unet.py
min_gap = 0.05
recon_log_t = [log_ts[0]]
total_span = log_ts[-1] - log_ts[0]

# Determine point density fraction
if len(ts) > 500:
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4

n_points = max(20, int(fraction * len(ts)))

for i in range(len(ts) - 1):
    gap_size = log_ts[i+1] - log_ts[i]
    if gap_size > min_gap:
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])

recon_log_t = np.array(recon_log_t)
# Ensure original points are included for training context, then unique sort
# (Note: For pure reconstruction we often just use the linspace, but keeping your logic)
# To keep strictly to the script logic provided:
recon_t = 10**np.array(recon_log_t)
recon_t = np.unique(recon_t)
log_recon_t = np.log10(recon_t).reshape(-1, 1)

# --- 6. TRAINING ---
if st.button("Train and Reconstruct"):
    with st.spinner("Training Attention U-Net... (This may take a moment)"):
        
        # Prepare Data
        X_train = log_ts.reshape(-1, 1, 1)
        y_train = log_fluxes.reshape(-1, 1, 1)
        
        # Build Model
        model = UNetWithAttention1D(input_shape=(1, 1))
        
        # Train
        model.fit(X_train, y_train, epochs=1000, verbose=0, batch_size=64)
        
        # Predict
        x_test = log_recon_t.reshape(-1, 1, 1)
        recon_fluxes_up = model.predict(x_test).flatten()
        
        # --- 7. NOISE & UNCERTAINTY GENERATION ---
        # Logic from Attention_unet.py
        
        # Time error analysis
        positive_ts_err = trimmed_data["pos_t_err"]
        negative_ts_err = trimmed_data["neg_t_err"]
        ts_error = (positive_ts_err - negative_ts_err)/2
        log_ts_error = ts_error/(ts*np.log(10))
        
        errparameters_time = st_scipy.norm.fit(log_ts_error)
        err_dist_time = st_scipy.norm(loc=errparameters_time[0], scale=errparameters_time[1])
        
        # Flux error analysis
        logfluxerrs = fluxes_error / (fluxes * np.log(10))
        errparameters_flux = st_scipy.norm.fit(logfluxerrs)
        err_dist_flux = st_scipy.norm(loc=errparameters_flux[0], scale=errparameters_flux[1])
        
        # Generate errors
        recon_errorbar = err_dist_flux.rvs(size=len(recon_fluxes_up))
        recon_errorbar = np.where(recon_errorbar < 0, 0, recon_errorbar)
        
        # Generate point-specific noise
        point_specific_noise = np.array([
            st_scipy.norm(loc=pred, scale=err).rvs() - pred
            for pred, err in zip(recon_fluxes_up, recon_errorbar)
        ])
        
        # Jiggled points (Reconstructed Points)
        jiggled_points = recon_fluxes_up + point_specific_noise
        
        # Downsampling logic (if needed, otherwise 1:1)
        # Your script had downsampling based on len(log_recon_t), but here x_test IS log_recon_t
        # So factor should be 1, but we keep the variables for consistency with plotting code
        factor = 1 
        recon_fluxes_up_downsampled = recon_fluxes_up[::factor]
        jiggled_points_downsampled = jiggled_points[::factor]
        recon_errorbar_downsampled = recon_errorbar[::factor]
        log_recon_t_flat = log_recon_t.flatten()

        # Monte Carlo for Confidence Intervals
        num_samples = 1000
        random_samples = np.array([
            st_scipy.norm(loc=0, scale=err).rvs(num_samples)
            for err in recon_errorbar_downsampled
        ]).T
        jiggled_realizations = recon_fluxes_up_downsampled + random_samples
        ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
        ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)

        # --- 8. PLOTTING (EXACT REQUESTED FORMAT) ---
        st.subheader("Reconstruction Results")
        
        # Create figure using matplotlib
        fig, plt_ax = plt.subplots(figsize=(10, 6))
        
        # Mapping variables to your requested snippet names
        train_x_denorm = log_ts
        train_y_denorm = log_fluxes
        test_x_denorm = log_recon_t_flat
        log_reconstructed_flux = jiggled_points_downsampled
        mean_prediction_denorm = recon_fluxes_up_downsampled
        lower_denorm = ci_95_lower
        upper_denorm = ci_95_upper
        
        # a) Plot original data with updated y-errors
        plt_ax.errorbar(
            train_x_denorm,  # Log(time)
            train_y_denorm,  # Log(flux)
            zorder=4,  # Plot on top (higher z-order)
            yerr=[lower_err_log, upper_err_log],  # Error bars for log(flux)
            linestyle="",  # No connecting lines
            ecolor='gray', # Added ecolor for visibility against dark bg if needed
            label='Observed Data' # Explicit label for legend if not covered by scatter
        )

        # b) Plot reconstructed points with synthetic error bars
        plt_ax.errorbar(
            test_x_denorm,  # Reconstructed log(time)
            log_reconstructed_flux,  # Noisy reconstructed log(flux)
            linestyle='none',  # No connecting lines
            yerr=np.abs(recon_errorbar_downsampled),  # Synthetic error bars
            marker='o',  # Circle markers
            capsize=5,  # Error bar cap length
            color='yellow',  # Yellow color for reconstructed points
            zorder=3,  # Slightly lower z-order
            label="Reconstructed Points"  # Legend label
        )

        # c) Scatter original observed points on top
        plt_ax.scatter(
            train_x_denorm,  # Log(time)
            train_y_denorm,  # Log(flux)
            zorder=5,  # Highest z-order to ensure visibility
            label="Observed Points",  # Legend label
            color='blue' # Explicit color to match standard scientific plots
        )

        # d) Plot the mean prediction curve
        plt_ax.plot(
            test_x_denorm,  # Log(time)
            mean_prediction_denorm,  # Mean predicted log(flux)
            label="Mean Prediction",  # Legend label
            zorder=2,  # Lower z-order
            color='red', # Standard red for mean line
            linewidth=2
        )

        # e) Add 95% confidence interval shading
        plt_ax.fill_between(
            test_x_denorm.flatten(),  # Log(time)
            lower_denorm,  # Lower CI bound
            upper_denorm,  # Upper CI bound
            alpha=0.5,  # Transparency
            color='orange',  # Orange color for CI
            label="95% Confidence Region",  # Legend label
            zorder=1  # Lowest z-order
        )

        plt_ax.legend(loc='lower left')  # Place legend in lower-left corner
        plt_ax.set_xlabel('log$_{10}$(Time) (s)', fontsize=15)  # X-axis label with LaTeX formatting
        plt_ax.set_ylabel('log$_{10}$(Flux) ($erg\\,cm^{-2}\\,s^{-1}$)', fontsize=15)  # Y-axis label with LaTeX
        plt_ax.set_title(f'Attention U-Net on {GRB_Name}', fontsize=18)  # Plot title
        
        # Display in Streamlit
        st.pyplot(fig)
        
        # --- 9. SAVE & EXPORT ---
        # Build combined DataFrame
        # For the export, we just export the reconstruction points as requested
        
        export_data = []
        for k in range(len(test_x_denorm)):
            # Convert back to linear for saving (as per your script logic)
            time_linear = 10**test_x_denorm[k]
            flux_linear = 10**log_reconstructed_flux[k]
            flux_err_linear = flux_linear * np.log(10) * recon_errorbar_downsampled[k]
            
            # Approximating pos/neg time error from the generated distribution
            # In your script, you used the random sample directly, here we use mean
            # using the error bar value for both pos/neg t err as approximation for CSV
            t_err_linear = 10**recon_errorbar_downsampled[k] 
            
            export_data.append({
                "t": time_linear,
                "pos_t_err": t_err_linear, 
                "neg_t_err": t_err_linear,
                "flux": flux_linear,
                "pos_flux_err": flux_err_linear,
                "neg_flux_err": flux_err_linear
            })
            
        export_df = pd.DataFrame(export_data)
        
        # Add to original dataframe (concat)
        final_df = pd.concat([trimmed_data, export_df], ignore_index=True)
        
        csv = final_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Combined CSV",
            data=csv,
            file_name=f"{GRB_Name}_reconstructed.csv",
            mime="text/csv",
        )
        
        st.success("Reconstruction Complete!")
