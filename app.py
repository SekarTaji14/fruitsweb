from flask import Flask, request, jsonify
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
model = tf.keras.models.load_model(MODEL_PATH)

# Fungsi untuk memproses gambar
def process_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(200, 200))  # Sesuaikan dengan ukuran model Anda
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi jika diperlukan
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah file diunggah
    if 'foto' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['foto']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file dengan nama aman
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Prediksi gambar
    try:
        img_array = process_image(file_path)
        prediction = model.predict(img_array)
        class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']
        predicted_class = class_names[np.argmax(prediction)]
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

    # Respons dalam format JSON
    return jsonify({
        'prediction': predicted_class,
        'image_path': f'static/uploads/{filename}'  # Path relatif ke gambar yang diunggah
    })

if __name__ == "__main__":
    app.run(debug=True)
