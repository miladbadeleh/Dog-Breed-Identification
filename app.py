# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image


# app.py (continued)
@st.cache_resource # Cache the model loading so it only happens once
def load_model_and_labels():
    """Loads the pre-trained model and class labels."""
    model = tf.keras.models.load_model('dog_breed_mobilenetv2.h5')
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_model_and_labels()



# app.py (continued)
def preprocess_image(uploaded_file, target_size=(224, 224)):
    """
    Preprocesses the uploaded image to match the model's expected input.
    """
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch of size 1
    img_array = preprocess_input(img_array) # Use the same preprocessing as MobileNet
    return img, img_array



# app.py (continued)
def predict_breed(img_array, top_k=3):
    """
    Makes a prediction and returns the top K breeds and their probabilities.
    """
    predictions = model.predict(img_array, verbose=0)[0]
    # Get the indices of the top K predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_breeds = [class_names[i] for i in top_k_indices]
    top_k_probs = [predictions[i] for i in top_k_indices]
    return top_k_breeds, top_k_probs




# app.py (continued)
def main():
    st.set_page_config(page_title="Dog Breed Identifier", page_icon="🐕")
    st.title("🐕 Dog Breed Identification")
    st.markdown("Upload an image of a dog, and this AI model will predict its breed!")

    uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        raw_img, processed_img = preprocess_image(uploaded_file)

        # Display the image
        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_img, caption="Uploaded Image", use_column_width=True)

        # Make prediction on button click
        if st.button('Identify Breed'):
            with st.spinner('Analyzing...'):
                breeds, probabilities = predict_breed(processed_img, top_k=3)

            with col2:
                st.subheader("Prediction Results")
                for breed, prob in zip(breeds, probabilities):
                    # Format the breed name (often comes as folder_name)
                    formatted_breed = breed.split('-')[-1].replace('_', ' ').title()
                    st.metric(label=formatted_breed, value=f"{prob:.2%}")

            # Create a bar chart for the top predictions
            fig, ax = plt.subplots()
            # Format all breed names for the chart
            formatted_breeds = [b.split('-')[-1].replace('_', ' ').title() for b in breeds]
            ax.barh(formatted_breeds, probabilities, color='skyblue')
            ax.set_xlabel('Confidence')
            ax.set_title('Top Predictions')
            plt.gca().invert_yaxis() # Highest probability at the top
            st.pyplot(fig)

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses a deep learning model built with **Transfer Learning**.

        - **Model:** MobileNetV2 pre-trained on ImageNet, fine-tuned on the Stanford Dogs Dataset.
        - **Capability:** Classifies images across 120 different dog breeds.
        - **Disclaimer:** This is a fun demo. Accuracy may vary with image quality!
        """)

if __name__ == "__main__":
    main()
