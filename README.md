# Dog Breed Identification

A deep learning project that classifies dog breeds from images using Transfer Learning with a Convolutional Neural Network (CNN).

## 🚀 Features

*   **Deep Learning:** Utilizes Transfer Learning with a pre-trained MobileNetV2 model for high accuracy.
*   **Data Augmentation:** Incorporates real-time image augmentation during training to improve model generalization.
*   **Interactive Web App:** Built with Streamlit for a user-friendly interface to upload images and get predictions.
*   **Top-K Predictions:** Returns the top 3 most likely breeds with confidence scores, providing more insight than a single guess.

## 🛠️ Tech Stack

*   **Framework:** TensorFlow / Keras
*   **Pre-trained Model:** MobileNetV2
*   **Data:** Stanford Dogs Dataset (120 breeds, ~20k images)
*   **Web App:** Streamlit
*   **Image Processing:** OpenCV, PIL

## 📦 Installation & Usage

### 1. Training the Model
The Jupyter Notebook `model_training.ipynb` contains the full code for:
- Downloading and preparing the dataset.
- Building the Transfer Learning model.
- Training and fine-tuning the model.
- Saving the final model and class labels.

### 2. Running the Web App
1.  Ensure you have a trained model file (`dog_breed_mobilenetv2.h5`) and label file (`class_names.pkl`).
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## 🎯 How It Works

The project follows a standard deep learning workflow:

1.  **Data Acquisition:** Uses the Stanford Dogs Dataset.
2.  **Preprocessing:** Images are resized, normalized, and augmented (rotation, flips, etc.) to create a robust training set.
3.  **Transfer Learning:** A MobileNetV2 model, pre-trained on the massive ImageNet dataset, is used as a feature extractor. Its final layers are replaced with new layers tailored for the 120-class dog breed problem.
4.  **Training:** The new head of the model is trained, followed by optional fine-tuning of the pre-trained base.
5.  **Inference:** New images are preprocessed identically and passed through the model to get a prediction.

## 🔮 Future Improvements

*   **Deployment:** Deploy the app on Streamlit Community Cloud or Hugging Face Spaces.
*   **Model Choices:** Experiment with different base models like EfficientNet or ConvNeXt for higher accuracy.
*   **Explainability:** Integrate Grad-CAM to highlight which parts of the image the model used for its prediction.
*   **Real-time:** Use the device's camera for real-time breed identification.
