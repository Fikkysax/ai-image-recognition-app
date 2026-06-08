import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
model = tf.keras.applications.MobileNetV2(weights="imagenet")
def recognize_image(image):
    image = Image.fromarray(image).resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions,
        top=5
    )[0]
    results = {}
    for prediction in decoded_predictions:
        label = prediction[1].replace("_", " ")
        confidence = float(prediction[2])
        results[label] = confidence
    return results
app = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="numpy", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="AI Predictions"),
    title="Free AI Image Recognition App",
    description="Upload an image and the AI will predict the main object it sees."
)
app.launch()