import streamlit as st
from PIL import Image
import pickle as pkl
from imgtodata import id as image_to_data
import time 
import documentation as doc
from input import entry

st.set_page_config(page_title="Exoplanet Detection App", page_icon=":telescope:", layout="centered")


# Load the pre-trained model 
model=pkl.load(open('exoplanet_model.pkl', 'rb'))

st.title('Exoplanet Detection Using Light Curves')
st.sidebar.title("Exoplanet Detection App")
mode=('Documentations','Deployment','Model Summary')
choose=st.sidebar.selectbox('Select waht you want to do:',mode,)
st.sidebar.caption(f'you have selected to see the {choose}')

                
                
if choose=='Documentations':
    pass
elif choose == 'Deployment':
        met=st.selectbox("Select whether you want to upload an image or amnually enter data:",['','Upload Image','Manually Enter the data'])
        if met=='':
            pass
        if met=='Upload Image':
            st.write(f"so you have chose {met} option")
            mage = st.file_uploader("Upload a light curve image", type=["png", "jpg", "jpeg"])

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

            
            st.subheader("1. Basic Flux Statistics",divider='rainbow')

            mean_flux = st.number_input("1. Mean flux")
            median_flux = st.number_input("2. Median flux")
            std_flux = st.number_input("3. Standard deviation of flux")
            skew_flux = st.number_input("4. Flux skewness")
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
            


else:
    pass