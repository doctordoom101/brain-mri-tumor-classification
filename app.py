import gradio as gr
import numpy as np
import tensorflow as tf

# Load model dari folder
model = tf.keras.models.load_model("saved_model")

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_brain_tumor(img):
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return {class_labels[i]: float(pred[i]) for i in range(len(class_labels))}

iface = gr.Interface(
    fn=predict_brain_tumor,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor MRI Classification",
    description="Upload MRI brain scan to classify tumor type."
)

iface.launch()
