import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
model = tf.keras.applications.MobileNetV2(weights="imagenet")
def recognize_image(image):
    if image is None:
        return {}
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
        label = prediction[1].replace("_", " ").title()
        confidence = float(prediction[2])
        results[label] = confidence
    return results
app = gr.Interface(
    fn=recognize_image,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="AI Image Recognition App",
    description="""
Upload an image and this app will predict the top 5 possible objects it sees.
This app uses a free TensorFlow MobileNetV2 model, so it works without paid API credits.
""",
    article="""
### About this project
This app uses a free TensorFlow MobileNetV2 model to predict the top 5 possible objects in an uploaded image.
Note: This free version recognizes general objects, not exact product models.
"""
)


app.launch()
