import streamlit as st
from PIL import Image
import pickle as pkl
from imgtodata import id as image_to_data
import time 
import documentation as doc
from input import entry
from streamlit_mermaid import st_mermaid as st_md
from documentation import references, captiom_1, captiom_2, abstract, prop_1, prop_2, dev, tech, system, test
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Detection App", page_icon=":telescope:", layout="centered")


# Load the pre-trained model 
model=pkl.load(open('exoplanet_model.pkl', 'rb'))


st.sidebar.title("Exoplanet Detection App")
mode=('Documentations','Deployment','Model Simulation')
choose=st.sidebar.selectbox('Select waht you want to do:',mode,)
st.sidebar.caption(f'you have selected to see the {choose}')

                
                
if choose=='Documentations':
    st.title('Exoplanet Detection Using Light Curves',width="content")
    st.divider()
    st.subheader('Abstract',divider='grey')
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image('assets/1637006343899.png')
    st.write(abstract)
    st.divider()
    
    
    st.subheader('Problem Statement',divider='grey')
    st.write(captiom_1)
    st.caption(captiom_2)
    st.divider()
    
    
    st.subheader('Proposed System',divider='grey')
    st.write(prop_1)
    st.caption(prop_2)
    st.divider()
    
    
    st.subheader('Model Architecture',divider='grey')
    mindmap = """
                mindmap
                root((Model))
                    (Data Loading)
                     (Data Preparation)
                     (Data Visualization)
                     (Feature Engineering)
                    (Core Engine)
                     (Data Pipeline)
                     (Random Forest Classifier)
                     (Model Evaluation)
                    (Visualization Layer)
                     (LightKurves)
                     (Streamlit)
                     (Interactive UI)
                """
    st_md(mindmap)
    st.divider()
    
    
    st.subheader('Tech Stack',divider='grey')
    st.write(tech)
    st.divider()
    
    
    st.subheader('System Requirements',divider='grey')
    st.write(system)
    st.warning(test)
    st.divider()
    
    
    st.subheader('Deployment Principle',divider='grey') 
    st.write(dev)
    st.divider()
    
    
    st.subheader('References',divider='grey') 
    st.caption(references)
    st.divider()

    




elif choose == 'Deployment':
        st.title('Practical Demonstration')
        st.divider()
        met=st.selectbox("Select whether you want to upload an image or amnually enter data:",['','Upload Image','Manually Enter the data'])
        st.divider()
        if met=='':
            pass
        if met=='Upload Image':
            mage = st.file_uploader("Upload a light curve image", type=["png", "jpg", "jpeg"])
            st.divider()
            # Initialize session state for prediction
            if "pred_result" not in st.session_state:
                st.session_state["pred_result"] = None
                st.session_state["pred_error"] = None

            def pred(image_name):
                data = image_to_data(image_name)
                st.write(data)
                if data is None:
                    st.session_state["pred_result"] = None
                    st.session_state["pred_error"] = "The uploaded image is not recognized. Please upload a valid light curve image."
                    return

                prediction = model.predict([data])

                st.session_state["pred_error"] = None
                st.session_state["pred_result"] = prediction

            if mage is not None:
                show = st.checkbox("Show Uploaded Image")

                if show:
                    img = Image.open(mage)
                    img = img.resize((256, 256))
                    st.image(img, caption="Uploaded Image", width=300)
                
                if st.button("Predict"):
                    with st.spinner('Analyzing the light curve...'):
                        time.sleep(2)  
                    pred(mage.name)
                    

                # Display whatever is stored in state, regardless of whether the button was pressed this run
                if st.session_state["pred_error"]:
                    st.error(st.session_state["pred_error"])
                elif st.session_state["pred_result"] is not None:
                    if st.session_state["pred_result"] == 0:
                        st.error("Not a planet")
                        st.write_stream(doc.no_planet)
                    elif st.session_state["pred_result"] == 1:
                        st.warning("This could be an exoplanet candidate. Further analysis required.")
                        st.write_stream(doc.can_planet)
                    else:
                        st.success("This is an exoplanet")
                        st.write_stream(doc.planet)
                        
            else:
                st.info("Please upload a light curve image to begin.")
        elif met=='Manually Enter the data':
            st.write("Please enter the following features extracted from the light curve:")
            st.divider()
            
            st.subheader("1. Basic Flux Statistics",divider='rainbow')

            mean_flux = st.number_input("1. Mean flux")
            median_flux = st.number_input("2. Median flux")
            std_flux = 4.71e-06
            skew_flux = -4.71e-06
            kurt_flux = st.number_input("5. Flux kurtosis")
            snr = st.number_input("6. Signal-to-noise ratio (SNR)")
            rms_residuals = st.number_input("7. RMS of residuals (after detrending)")

            
            st.subheader("2. Transit-Shape Features",divider='rainbow')

            transit_depth = st.number_input("8. Transit depth")
            transit_duration = st.number_input("9. Transit duration")
            ingress_duration = st.number_input("10. Transit ingress duration")
            egress_duration = st.number_input("11. Transit egress duration")
            t14 = st.number_input("12. Full-transit duration (T14)")
            t23 = st.number_input("13. Flat-bottom duration (T23)")
            asymmetry = st.number_input("14. Transit asymmetry (ingress/egress ratio)")
            transit_slope = st.number_input("15. Transit slope (flux change per time)")

            
            st.subheader("3. Period & Harmonic Features",divider='rainbow')

            orbital_period = st.number_input("16. Detected orbital period")
            secondary_periodicity = st.number_input("17. Secondary periodicity (e.g. harmonic index)")
            phase_folded_var = st.number_input("18. Phase-folded flux variance")
            odd_even_depth_diff = st.number_input("19. Odd–even transit depth difference")
            transit_consistency = st.number_input("20. Transit consistency score across cycles")

           
            st.subheader("4. Shape Model Fit Features",divider='rainbow')

            impact_parameter = st.number_input("21. Impact parameter (b)")
            rp_over_rstar = st.number_input("22. Planet-to-star radius ratio (Rp/R★)")
            chi_square = st.number_input("23. Model fit chi-square error")
            bic = st.number_input("24. Bayesian Information Criterion (BIC)")
            aic = st.number_input("25. Akaike Information Criterion (AIC)")

           
            st.subheader("5. Outlier & Noise Behavior",divider='rainbow')

            num_positive_outliers = st.number_input("26. Number of positive outliers")
            num_negative_outliers = st.number_input("27. Number of negative outliers")
            flicker_metric = st.number_input("28. Flicker noise metric")
            cdpp = st.number_input("29. CDPP (Combined Differential Photometric Precision)")

            
            st.subheader("6. Frequency-Domain Features",divider='rainbow')

            ps_peak_amp = st.number_input("30. Power spectrum peak amplitude")
            ps_entropy = st.number_input("31. Power spectral entropy")

            
            st.subheader("7. Extra Physical / Uncertainty Features",divider='rainbow')

            transit_depth_unc = st.number_input("32. Transit depth uncertainty")
            period_unc = st.number_input("33. Period uncertainty")
            stellar_radius = st.number_input("34. Stellar radius (scaled or binned)")
            stellar_teff = st.number_input("35. Stellar effective temperature (Teff, scaled/bin)")
            limb_darkening = st.number_input("36. Limb-darkening coefficient (binned/indexed)")
            extra=-0.0184
            
            #features
            features = [
            mean_flux,
            median_flux,
            std_flux,
            skew_flux,
            kurt_flux,
            snr,
            rms_residuals,
            transit_depth,
            transit_duration,
            ingress_duration,
            egress_duration,
            t14,
            t23,
            asymmetry,
            transit_slope,
            orbital_period,
            secondary_periodicity,
            phase_folded_var,
            odd_even_depth_diff,
            transit_consistency,
            impact_parameter,
            rp_over_rstar,
            chi_square,
            bic,
            aic,
            num_positive_outliers,
            num_negative_outliers,
            flicker_metric,
            cdpp,
            ps_peak_amp,
            ps_entropy,
            transit_depth_unc,
            period_unc,
            stellar_radius,
            stellar_teff,
            limb_darkening,
            extra
            ]
            
            #Prediction
            if st.button("Predict"):
                with st.spinner('Analyzing the light curve...'):
                        time.sleep(2)  
                prediction = model.predict([features])
                if prediction == 0:
                    st.error("Not a planet")
                    st.write_stream(doc.no_planet)
                elif prediction == 1:
                    st.warning("This could be an exoplanet candidate. Further analysis required.")
                    st.write_stream(doc.can_planet)
                else:
                    st.success("This is an exoplanet")
                    st.write_stream(doc.planet)
            


elif choose=='Model Simulation':
    st.title("Exoplanet Transit & Light Curve Theory – Interactive Demo")
    st.divider()
    st.markdown(
        """
    This app shows:
    - Basic transit depth: ΔF/F ≈ (Rp / R★)²  
    - A simple transit light curve: F(t) = F₀ · (1 − δ(t))  
    - A **toy** Box Least Squares (BLS-like) period search on the synthetic data  
    """
    )

    st.sidebar.divider()
    st.sidebar.header("Physical parameters")

    R_star = st.sidebar.number_input("Stellar radius R★ (in Solar radii)", 0.1, 10.0, 1.0, 0.1)
    R_p = st.sidebar.number_input("Planet radius Rp (in Earth radii)", 0.1, 20.0, 1.0, 0.1)

    # Convert Rp in Earth radii to R_star units (Solar radii) for ratio
    R_earth_in_Rsun = 1.0 / 109.1  
    Rp_over_Rstar = (R_p * R_earth_in_Rsun) / R_star

    F0 = st.sidebar.number_input("Baseline flux F₀", 0.1, 10.0, 1.0, 0.1)

    st.sidebar.header("Orbit / Transit parameters")
    P = st.sidebar.number_input("Orbital period P (days)", 0.5, 100.0, 10.0, 0.5)
    transit_duration = st.sidebar.number_input("Transit duration (fraction of period)", 0.01, 0.5, 0.05, 0.01)
    n_periods = st.sidebar.number_input("Number of periods to simulate", 1, 50, 5, 1)

    st.sidebar.header("Noise & Sampling")
    total_points = st.sidebar.number_input("Number of samples", 100, 20000, 2000, 100)
    noise_std = st.sidebar.number_input("Gaussian noise σ", 0.0, 0.1, 0.002, 0.001)


    depth = Rp_over_Rstar ** 2  
    
    st.divider()
    
    st.subheader("1. Basic Transit Depth Formula",divider='grey')

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Depth formula:**")
        st.latex(r"\frac{\Delta F}{F} \approx \left(\frac{R_p}{R_\star}\right)^2")
        st.write(f"Numerical value (ΔF/F): **{depth:.6f}**")
        st.write(f"Percentage drop in flux: **{depth * 100:.4f} %**")

    with col2:
        st.write("**Inputs interpreted as:**")
        st.write(f"- Stellar radius R★ = {R_star:.2f} R⊙")
        st.write(f"- Planet radius Rp = {R_p:.2f} R⊕")
        st.write(f"- Radius ratio Rp/R★ ≈ {Rp_over_Rstar:.6f}")

    st.divider()
    st.subheader("2. Simple Transit Light Curve Model",divider='grey')

    # Time array in days
    t_total = P * n_periods
    time = np.linspace(0, t_total, int(total_points))

    # Phase in [0, 1)
    phase = (time % P) / P

    # We'll put the transit centered at phase = 0.5
    center = 0.5
    half_width = transit_duration / 2.0
    in_transit = np.abs(phase - center) < half_width

    # Box model: flux drop = depth during transit, 0 outside
    flux = F0 * (1.0 - depth * in_transit.astype(float))

    # Add Gaussian noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, noise_std, size=time.shape)
    flux_noisy = flux + noise

    st.markdown(
        """
    We use a **box-shaped transit model**:

    - Transit is centered at phase 0.5  
    - In-transit region has a constant depth ΔF/F  
    - Out of transit: flux = F₀  
    - Then we add Gaussian noise  
    """
    )

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.scatter(time, flux_noisy, s=4, alpha=0.7, label="Noisy flux")
    ax1.plot(time, flux, linewidth=2, label="True model (no noise)")
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Flux")
    ax1.set_title("Simulated Light Curve")
    ax1.legend()
    st.pyplot(fig1)


    st.markdown("**Phase-folded light curve (folded on the true period)**")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(phase, flux_noisy, s=4, alpha=0.7)
    ax2.set_xlabel("Phase (t mod P / P)")
    ax2.set_ylabel("Flux")
    ax2.set_title("Phase-Folded Light Curve")
    st.pyplot(fig2)

    st.divider()
    st.subheader("3. Toy Box Least Squares (BLS-like) Period Search",divider='grey')

    st.markdown(
        """
    We now try to **recover the period** by scanning a range of trial periods and fitting a simple
    box model at each trial. For each trial period:

    1. Fold the time series on that period  
    2. Define a box of the same duration fraction  
    3. Compute a sum of squared residuals (χ²) between data and the best-fit box model  
    4. Invert χ² to get a "power" (smaller χ² → higher power)  

    This is a simplified demonstration of the **Box Least Squares** idea.
    """
    )

    # Define trial periods around the true P
    n_trials = 200
    P_min = P * 0.5
    P_max = P * 1.5
    trial_periods = np.linspace(P_min, P_max, n_trials)

    powers = []

    for Pt in trial_periods:
        # Fold data
        ph = (time % Pt) / Pt

        # Define in-transit region (same duration fraction)
        c = 0.5
        hw = transit_duration / 2.0
        mask_in = np.abs(ph - c) < hw

        # Simple model: out-of-transit flux = constant, in-transit flux = constant drop
        F_out = np.median(flux_noisy[~mask_in]) if np.any(~mask_in) else np.median(flux_noisy)
        F_in = np.median(flux_noisy[mask_in]) if np.any(mask_in) else np.median(flux_noisy)

        model_trial = np.where(mask_in, F_in, F_out)

        # χ² = sum (data - model)^2
        chi2 = np.sum((flux_noisy - model_trial) ** 2)

        # Convert to "power": lower χ² -> higher power
        power = 1.0 / chi2 if chi2 > 0 else 0
        powers.append(power)

    powers = np.array(powers)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(trial_periods, powers, lw=1.5)
    ax3.axvline(P, color="red", linestyle="--", label="True period")
    ax3.set_xlabel("Trial period [days]")
    ax3.set_ylabel("BLS-like power (1 / χ²)")
    ax3.set_title("Toy BLS Periodogram")
    ax3.legend()
    st.pyplot(fig3)

    best_idx = np.argmax(powers)
    best_P = trial_periods[best_idx]

    st.write(f"Best period detected by toy BLS: **{best_P:.4f} days** (true: {P:.4f} days)")

    st.divider()
    st.markdown("### What this app is showing in code form")

    st.write("**1. Transit depth formula**")
    st.latex(r"\frac{\Delta F}{F} \approx \left(\frac{R_p}{R_\star}\right)^2")

    st.write("**2. Light curve model**")
    st.latex(r"F(t) = F_0 \, (1 - \delta(t))")
    st.latex(r"\delta(t) = \frac{\Delta F}{F} \text{ during transit, and } 0 \text{ outside}")

    st.write("**3. BLS concept (simplified)**")

    st.write("Fold light curve, fit a box model, then measure fit quality:")

    st.latex(r"\chi^2 = \sum (F_{\text{data}} - F_{\text{model}})^2")

    st.write("Convert χ² into a detection score (lower error → higher power):")

    st.latex(r"Power = \frac{1}{\chi^2}")

    st.markdown(
        """
    This is a simplified implementation of how exoplanets are detected from light curves.
    Next step is connecting the **36 engineered features** to this simulator and generating training-ready datasets.
    """
    )
    st.divider()