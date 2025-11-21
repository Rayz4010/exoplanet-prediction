import streamlit as st
def entry(): 
    # ---------- 1. Basic Flux Statistics ----------
    st.header("1. Basic Flux Statistics")

    mean_flux = st.number_input("1. Mean flux")
    median_flux = st.number_input("2. Median flux")
    std_flux = st.number_input("3. Standard deviation of flux")
    skew_flux = st.number_input("4. Flux skewness")
    kurt_flux = st.number_input("5. Flux kurtosis")
    snr = st.number_input("6. Signal-to-noise ratio (SNR)")
    rms_residuals = st.number_input("7. RMS of residuals (after detrending)")

    # ---------- 2. Transit-Shape Features ----------
    st.header("2. Transit-Shape Features")

    transit_depth = st.number_input("8. Transit depth")
    transit_duration = st.number_input("9. Transit duration")
    ingress_duration = st.number_input("10. Transit ingress duration")
    egress_duration = st.number_input("11. Transit egress duration")
    t14 = st.number_input("12. Full-transit duration (T14)")
    t23 = st.number_input("13. Flat-bottom duration (T23)")
    asymmetry = st.number_input("14. Transit asymmetry (ingress/egress ratio)")
    transit_slope = st.number_input("15. Transit slope (flux change per time)")

    # ---------- 3. Period & Harmonic Features ----------
    st.header("3. Period & Harmonic Features")

    orbital_period = st.number_input("16. Detected orbital period")
    secondary_periodicity = st.number_input("17. Secondary periodicity (e.g. harmonic index)")
    phase_folded_var = st.number_input("18. Phase-folded flux variance")
    odd_even_depth_diff = st.number_input("19. Odd–even transit depth difference")
    transit_consistency = st.number_input("20. Transit consistency score across cycles")

    # ---------- 4. Shape Model Fit Features ----------
    st.header("4. Shape Model Fit Features")

    impact_parameter = st.number_input("21. Impact parameter (b)")
    rp_over_rstar = st.number_input("22. Planet-to-star radius ratio (Rp/R★)")
    chi_square = st.number_input("23. Model fit chi-square error")
    bic = st.number_input("24. Bayesian Information Criterion (BIC)")
    aic = st.number_input("25. Akaike Information Criterion (AIC)")

    # ---------- 5. Outlier & Noise Behavior ----------
    st.header("5. Outlier & Noise Behavior")

    num_positive_outliers = st.number_input("26. Number of positive outliers")
    num_negative_outliers = st.number_input("27. Number of negative outliers")
    flicker_metric = st.number_input("28. Flicker noise metric")
    cdpp = st.number_input("29. CDPP (Combined Differential Photometric Precision)")

    # ---------- 6. Frequency-Domain Features ----------
    st.header("6. Frequency-Domain Features")

    ps_peak_amp = st.number_input("30. Power spectrum peak amplitude")
    ps_entropy = st.number_input("31. Power spectral entropy")

    # ---------- 7. Extra Physical / Uncertainty Features ----------
    st.header("7. Extra Physical / Uncertainty Features")

    transit_depth_unc = st.number_input("32. Transit depth uncertainty")
    period_unc = st.number_input("33. Period uncertainty")
    stellar_radius = st.number_input("34. Stellar radius (scaled or binned)")
    stellar_teff = st.number_input("35. Stellar effective temperature (Teff, scaled/bin)")
    limb_darkening = st.number_input("36. Limb-darkening coefficient (binned/indexed)")
    
    
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
    limb_darkening
    ]
    return features