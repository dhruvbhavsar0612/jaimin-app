import streamlit as st
from inference import predict
from PIL import image

st.title("Sweetcorn Worm Classification")
st.write("Upload an image of a sweetcorn worm to classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    class_label, confidence = predict(image)

    # Display the predictions
    st.write("Prediction:")
    st.write(f"Class: {class_label}")
    st.write(f"Confidence: {confidence:.2f}%")