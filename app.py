from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model TensorFlow
MODEL_PATH = './model/model_fruits_classification.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Model file not found or failed to load: {MODEL_PATH}. Error: {str(e)}")

# Fungsi untuk memuat dan memproses gambar
def process_image(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(200, 200))  # Sesuaikan ukuran dengan model Anda
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisasi jika model Anda memerlukan input yang dinormalisasi
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image at path: {image_path}. Error: {str(e)}")

# Halaman utama
@app.route('/')
def index():
    return "Welcome to Freshify Backend!"

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah file diunggah
    if 'foto' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    imagefile = request.files['foto']
    if imagefile.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file dengan nama yang aman
    filename = secure_filename(imagefile.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        imagefile.save(image_path)
    except Exception as e:
        return jsonify({'error': f"Failed to save file. Error: {str(e)}"}), 500

    # Proses gambar dan prediksi
    try:
        img_array = process_image(image_path)
        prediction = model.predict(img_array)
    except Exception as e:
        return jsonify({'error': f"Failed to process image for prediction. Error: {str(e)}"}), 500

    # Logika klasifikasi berdasarkan output model
    class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
    predicted_class = class_names[np.argmax(prediction)]

    # Response JSON
    return jsonify({
        'prediction': predicted_class,
        'image_path': f'static/uploads/{filename}'  # Path relatif untuk gambar
    })

# Halaman about
@app.route('/about')
def about():
    return "This is the Freshify Backend API!"

# Route debugging
@app.route('/debug', methods=['GET', 'POST'])
def debug():
    return jsonify({
        "message": "Debug route working!",
        "method": request.method
    })

if __name__ == "__main__":
    app.run(debug=True)
