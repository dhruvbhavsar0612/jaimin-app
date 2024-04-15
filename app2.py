import streamlit as st
from inference import predict  # Assuming the inference function is defined elsewhere
from PIL import Image
import tensorflow as tf
import plotly.express as px

model = tf.keras.models.load_model('./model/model.keras')

st.set_page_config(page_title="Sweetcorn Worm Classification", layout="wide")  # Set wider layout

st.write("""
<style>
.css-1d6feon {  /* Adjust selector as needed based on Streamlit version */
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

.main {
  flex: 1;
  padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.title("Sweetcorn Worm Classification")
    st.write("By Jaimin Pancholy, Dhruvin Patel and Harsh Patel ")
    st.title("Instructions")
    st.write(" Upload an image of a sweetcorn worm to classify it.")
    # Upload image with clear instructions
    uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, or PNG format)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)

        # Display the uploaded image in a dedicated column
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.subheader("Prediction Results:")
            # Make predictions
            class_label, confidence = predict(model, image)
            st.write(f"**Class:** {class_label}")

        # Display classification results with clear formatting
        with col2:
            # Create interactive pie chart with Plotly
            fig = px.pie(
                values=[confidence, 100 - confidence],
                names=["Predicted Class", "Uncertainty"],
                title="Confidence Level",
                hole=0.8,  # Adjust hole size for aesthetics
                labels=dict(value="%")  # Format labels as percentages
            )
            fig.update_traces(textposition='outside', textinfo='percent')  # Display values inside pie slices

            st.plotly_chart(fig)  # Display the interactive plotly chart

