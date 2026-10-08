import lime
import lime.lime_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.segmentation import mark_boundaries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class LIME:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.explainer = lime.lime_image.LimeImageExplainer()

    def predict_fn(self, images):
        processed_images = np.array([preprocess_input(img.astype('float32')) for img in images])
        return self.model.predict(processed_images)

    def LIME_QUERY(self, img_path, save_path="lime_explanation.png"):
        img = Image.open(img_path).resize((128, 128))  # Keep original color format
        img = np.array(img)

        # Ensure grayscale images are converted to 3 channels (RGB)
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)

        # Apply preprocessing (but don't change color representation)
        img_preprocessed = preprocess_input(img.astype(np.float32))

        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            img_preprocessed, 
            self.predict_fn, 
            top_labels=5, 
            hide_color=0, 
            num_samples=1000
        )

        # Get explanation mask for the most probable label
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )

        # Ensure proper normalization for visualization (avoid Matplotlib clipping)
        explanation_img = mark_boundaries(temp / 2 + 0.5, mask)
        explanation_img = np.clip(explanation_img, 0, 1)  # Fix for imsave()

        # Save the image
        plt.imsave(save_path, explanation_img)
        print(f"LIME explanation saved as {save_path}")

        return explanation_img,save_path

if __name__ == "__main__":
    model_path = r'E:\AUD PROJECT\malaria_cnn_model_final_lates.h5'
    img_path = r'E:\AUD PROJECT\SAM.png'
    save_path = r'E:\AUD PROJECT\lime_explanation.png'
    
    lime_explainer = LIME(model_path)
    result = lime_explainer.LIME_QUERY(img_path, save_path)

    # Display the image
    plt.imshow(result)
    plt.axis('off')
    plt.show()
