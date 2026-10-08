import streamlit as st
import os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import shap
from SHARP_XAI import SHAP_Explainer
from Explainable_ai import LIME
from prediction import ImageClassifier


def save_shap_plot(shap_values, save_path="shap_output.png"):
    """Save SHAP explanation as an image."""
    fig, ax = plt.subplots()
    shap.image_plot(shap_values, show=False)  # Ensure it doesn't try to render in Streamlit
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_lime_plot(lime_exp, save_path="lime_output.png"):
    """Save LIME explanation as an image."""
    lime_exp.save_to_file(save_path)  # Ensure LIME explanation is saved correctly
    return save_path


def main():
    model_path = r"./models/malaria_cnn_model_final_lates.h5"
    class_labels = ["Parasitized", "Uninfected"]

    st.title("AI-Powered Image Classification with XAI")

    uploaded_file = st.file_uploader("Upload an image for X-AI", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        option = st.selectbox("Choose the explanation method", ["SHRP_XAI", "EXPLAIN_XAI"])

        if st.button("Predict"):
            with st.spinner("Processing... Please wait"):
                try:
                    # Save uploaded file temporarily
                    temp_dir = "temp_images"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load model and classify image
                    classifier = ImageClassifier(model_path, class_labels)
                    result = classifier.display_image_with_prediction(temp_file_path)

                    # Explanation methods
                    if option == "SHRP_XAI":
                        shap_explainer = SHAP_Explainer(model_path)
                        shap_values ,S = shap_explainer.explain(temp_file_path)  # Expecting SHAP values

                        if isinstance(shap_values, shap.Explanation):
                            shap_image_path = save_shap_plot(shap_values)
                            st.write(f"Predicted class: {result}")
                            st.image(shap_image_path, caption="SHARP X-AI", use_container_width=True)
                        else:
                            st.error("SHAP explanation failed. Please check your SHAP implementation.")

                    elif option == "EXPLAIN_XAI":
                        lime_explainer = LIME(model_path)
                        lime_exp,image2 = lime_explainer.LIME_QUERY(temp_file_path)  # Expecting a LIME object
                        image = lime_exp - lime_exp.min()
                        image = (image / (image.max() + 1e-10)) * 255
                        #img_normalized = (lime_exp - lime_exp.min()) / (lime_exp.max() - lime_exp.min() + 1e-10)
                        #lime_image_path = save_lime_plot(img_normalized)
                        st.write(f"Predicted class: {result}")
                        st.image(image2, caption="LIME X-AI", use_container_width=True)


                        #if isinstance(lime_exp, np.ndarray):
                        #    lime_image_path = save_lime_plot(lime_exp)
                        #    st.write(f"Predicted class: {result}")
                        #    st.image(lime_image_path, caption="LIME X-AI", use_column_width=True)
                        #else:
                        #    st.error("LIME explanation failed. Please check your LIME implementation.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
