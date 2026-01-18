import gradio as gr
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import json
import base64
from datetime import datetime

# load private key
with open("keys/private_key.pem", "rb") as f:
    private_key = serialization.load_pem_private_key(
        f.read(),
        password=None
    )

def sign_result(result_dict):
    payload = {
        "result": result_dict,
        "timestamp": datetime.utcnow().isoformat()
    }

    message = json.dumps(payload, sort_keys=True).encode()

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    payload["signature"] = base64.b64encode(signature).decode()
    return payload

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

    probs = output.numpy()[0].tolist()
    pred_dict = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}

    signed_output = sign_result(pred_dict)
    
    return pred_dict, signed_output

iface = gr.Interface(
    fn=predict_brain_tumor,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction Result"),
        gr.JSON(label="Digitally Signed Metadata (Audit Trail)")
    ],
    title="Brain Tumor MRI Classification",
    description="Hasil prediksi ditandatangani secara digital menggunakan RSA-PSS untuk menjamin integritas data."
)

iface.launch()
