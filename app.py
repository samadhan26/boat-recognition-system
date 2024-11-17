import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = "boat_model.h5"
model = load_model(MODEL_PATH)

# Categories corresponding to the output classes
categories = ['boats', 'buoy', 'cruise ship', 'ferry boat', 'freight boat', 
              'gondola', 'inflatable boat', 'kayak', 'paper boat', 'sailboat']

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to match model's input size
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# App layout and title
st.set_page_config(page_title="Boat Recognition System", layout="wide", initial_sidebar_state="expanded")
st.title("🚤 Boat Recognition System")
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    h1 {
        color: #2e69c8;
        font-size: 2.5em;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for uploading the image
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.write("Accepted formats: JPG, JPEG, PNG")

# Main section
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("**Processing...**")
    
    with col2:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Get predictions
        predictions = model.predict(preprocessed_image)
        predicted_class = categories[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display results
        st.success("### Recognition Results")
        st.markdown(f"**Predicted Category:** 🛶 `{predicted_class}`")
        st.markdown(f"**Confidence:** 🌟 `{confidence:.2f}%`")
        st.balloons()

# Footer
st.markdown("""
<hr>
<p style="text-align: center;">
Developed with ❤️ using <strong>Streamlit</strong>.
</p>
""", unsafe_allow_html=True)

