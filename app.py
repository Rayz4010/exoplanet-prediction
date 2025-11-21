import streamlit as st
from PIL import Image
import pickle as pkl
from imgtodata import id as image_to_data
import time 


# Load the pre-trained model 
model=pkl.load(open('exoplanet_model.pkl', 'rb'))

st.title('Exoplanet Detection Using Light Curves')
st.sidebar.title("Exoplanet Detection App")



mage = st.file_uploader("Upload a light curve image", type=["png", "jpg", "jpeg"])
def pred(mage):
    if mage:
        #st.write(name)
        data = image_to_data(name)
        #st.write(data)
        if data is not None:
            prediction = model.predict([data])
            if prediction == 0:
                st.error("False Positive")
            elif prediction == 1:
                st.warning("Exoplanet Detected")
            else:
                st.success("Exoplanet Confirmed")
        else:
            st.error("The uploaded image is not recognized. Please upload a valid light curve image.")


if st.button('Show Image Preview'):
    if mage is not None:
        global name
        img = Image.open(mage)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        name=mage.name
        pred(mage)
    else:
        st.warning("Please upload an image first.")

