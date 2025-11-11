import gradio as gr
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

# Load model SavedModel as inference layer
model = TFSMLayer("saved_model", call_endpoint="serving_default")

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_brain_tumor(img):
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Model call via TFSMLayer
    output = model(img)

    # Output biasanya dict → ambil tensor
    if isinstance(output, dict):
        output = list(output.values())[0]

    pred = output[0].numpy()

    return {class_labels[i]: float(pred[i]) for i in range(len(class_labels))}

iface = gr.Interface(
    fn=predict_brain_tumor,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor MRI Classification",
)

iface.launch()
