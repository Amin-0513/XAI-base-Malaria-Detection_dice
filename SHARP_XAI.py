import os
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class SHAP_Explainer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.masker_blur = shap.maskers.Image("blur(128,128)", shape=(128, 128, 3))
        self.explainer_blur = shap.Explainer(self.predict_fn, self.masker_blur)

    def preprocess_image(self, image_path):
        """Preprocess an image to match the model's input shape."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        img = image.load_img(image_path, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img[0]

    def predict_fn(self, images):
        """Model prediction function for SHAP."""
        processed_images = np.array([preprocess_input(img.astype('float32')) for img in images])
        return self.model.predict(processed_images)

    def explain(self, image_path):
        """Generate and visualize SHAP explanations for a given image."""
        img = self.preprocess_image(image_path)
        shap_values = self.explainer_blur(np.array([img]))  # Ensure the correct input format
        return shap_values, img

if __name__ == "__main__":
    model_path = r'E:\AUD PROJECT\malaria_cnn_model_final_lates.h5'
    image_path = r'E:\AUD PROJECT\SAM.png'
    
    shap_explainer = SHAP_Explainer(model_path)
    shap_values, img = shap_explainer.explain(image_path)
    
    # Visualize SHAP explanation
    shap.image_plot(shap_values, np.array([img]))  # Pass the image along with SHAP values
    plt.show()
