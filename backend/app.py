
from flask import Flask, render_template, request
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import tensorflow as tf
import base64



app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
class_names = tf.keras.applications.mobilenet_v2.decode_predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.files['image'].read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    # Preprocess the image
    img = Image.open(request.files['image'])
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    top_predictions = class_names(predictions, top=3)[0]

    results = []
    for _, name, prob in top_predictions:
        results.append((name, round(float(prob) * 100, 2)))

    return render_template('result.html', image=encoded_image, results=results)

if __name__ == '__main__':
    app.run(debug=True)