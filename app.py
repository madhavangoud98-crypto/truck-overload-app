import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime
import io
from tf_explain.core.grad_cam import GradCAM
import matplotlib.pyplot as plt

# Load your trained model (make sure 'truck_classifier_model.h5' is in this folder)
model = tf.keras.models.load_model('truck_classifier_model.h5')

def load_and_preprocess_image(img):
    image = Image.open(img).convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

def get_gradcam(img_array, model, layer_name=None):
    if layer_name is None:
        # Find last conv layer in model
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    explainer = GradCAM()
    grid = explainer.explain((img_array, None), model, class_index=0, layer_name=layer_name)
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_heatmap.png')
    plt.close()

st.title("Overloaded Vehicle Detection AI")

if 'history' not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload a truck image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Truck Image', use_column_width=True)

    processed_image = load_and_preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)[0][0]

    confidence_pct = prediction * 100
    label = "🚨 Overloaded Truck" if prediction > 0.5 else "✅ Normal Truck"
    st.markdown(f"### Prediction: {label}")
    st.write(f"Confidence: {confidence_pct:.2f}%")

    # Image file details
    file_details = {
        "Filename": uploaded_file.name,
        "File Type": uploaded_file.type,
        "File Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
        "Upload Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.write("### Image Details:")
    st.json(file_details)

    st.write("### Preprocessed Image Shape:", processed_image.shape)

    # Confidence explanation
    if confidence_pct > 80:
        st.info("Model is highly confident about this prediction.")
    elif confidence_pct > 50:
        st.warning("Model is somewhat confident, please verify.")
    else:
        st.error("Model is uncertain about this prediction.")

    # Add to prediction history
    st.session_state.history.append({
        "Filename": uploaded_file.name,
        "Prediction": label,
        "Confidence (%)": round(confidence_pct, 2),
        "Upload Time": file_details['Upload Time']
    })

    st.write("### Prediction History")
    st.table(st.session_state.history)

    # Prepare downloadable report
    report_text = f"""
Prediction Report
=================
Filename: {uploaded_file.name}
Prediction: {label}
Confidence: {confidence_pct:.2f}%
Upload Time: {file_details['Upload Time']}
"""
    st.download_button(
        label="Download Report",
        data=report_text,
        file_name="prediction_report.txt",
        mime="text/plain"
    )

    # Generate and show Grad-CAM heatmap
    get_gradcam(processed_image, model)
    st.image('gradcam_heatmap.png', caption="Model Explanation: Grad-CAM Heatmap")
