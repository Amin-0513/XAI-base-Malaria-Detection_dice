from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

class ImageClassifier:
    def __init__(self, model_path, class_labels):
        self.model = load_model(model_path)  # Load the trained model
        self.class_labels = class_labels

    def preprocess_image(self, img_path, target_size=(128, 128)):
        """
        Preprocess the image for prediction.
        """
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)  # Convert to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array

    def predict_image(self, img_path):
        """
        Predict the class of the image.
        """
        img_array = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]
        return predicted_class

    def display_image_with_prediction(self, img_path):
        """
        Display the image with the predicted class.
        """
        predicted_class = self.predict_image(img_path)
        return predicted_class
       # plt.imshow(image.load_img(img_path))
        #plt.title(f"Predicted: {predicted_class}")

       # plt.axis("off")
        #plt.show()

if __name__ == "__main__":
    model_path = r"E:\AUD PROJECT\malaria_cnn_model_final_lates.h5"
    class_labels = ["Parasitized", "Uninfected"]  # Replace with actual class names
    classifier = ImageClassifier(model_path, class_labels)
    image_path = r"E:\AUD PROJECT\C189P150ThinF_IMG_20151203_141809_cell_95.png"
    result=classifier.display_image_with_prediction(image_path)
    print(result)