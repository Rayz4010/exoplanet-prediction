import streamlit as st
from PIL import Image
import pickle as pkl
from imgtodata import id as image_to_data
import time 

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
    mage = st.file_uploader("Upload a light curve image", type=["png", "jpg", "jpeg"])

    # Initialize session state for prediction
    if "pred_result" not in st.session_state:
        st.session_state["pred_result"] = None
        st.session_state["pred_error"] = None

    def pred(image_name):
        data = image_to_data(image_name)
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
            st.spinner("Predicting...")
            time.sleep(3)
            pred(mage.name)

        # Display whatever is stored in state, regardless of whether the button was pressed this run
        if st.session_state["pred_error"]:
            st.error(st.session_state["pred_error"])
        elif st.session_state["pred_result"] is not None:
            if st.session_state["pred_result"] == 0:
                st.error("Not a planet")
            elif st.session_state["pred_result"] == 1:
                st.warning("This could be an exoplanet candidate. Further analysis required.")
            else:
                st.success("This is an exoplanet")
    else:
        st.info("Please upload a light curve image to begin.")


else:
    pass