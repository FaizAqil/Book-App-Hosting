from flask import Flask, request, jsonify
from google.cloud import firestore, storage
import tensorflow as tf
import requests
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Inisialisasi Firestore dan Google Cloud Storage
db = firestore.Client()
storage_client = storage.Client()

# URL publik model .h5 di Google Cloud Storage
model_url = os.getenv('MODEL_URL', 'https://storage.googleapis.com/storage-ml-similliar/model-book/book_recommendation_model.h5')

# Unduh model dari URL publik
model_path = 'book_recommendation_model.h5'
try:
    response = requests.get(model_url)
    response.raise_for_status()  # Memicu kesalahan jika status bukan 200
    with open(model_path, 'wb') as f:
        f.write(response.content)
except requests.RequestException as e:
    print(f"Error downloading model: {e}")
    exit(1)

# Muat model dari file .h5
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def is_image_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "success fetching api"
        },
        "data": None
    }), 200

@app.route('/upload', methods=['POST'])
def upload():
    user_data = request.json
    file = request.files.get('file')

    # Validasi input
    if not user_data or not user_data.get('user_id') or not user_data.get('title') or not user_data.get('review'):
        return jsonify({"error": "user_id, title, and review are required"}), 400

    # Periksa apakah file ada dan merupakan gambar
    if file and is_image_file(file.filename):
        filename = secure_filename(file.filename)

        # Upload file ke Google Cloud Storage
        bucket_name = os.getenv('BUCKET_NAME', 'online-book-borrowing-cloudrun')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)

        # Pastikan file dapat dibaca
        file.seek(0)
        blob.upload_from_string(file.read(), content_type=file.content_type)

        # Simpan informasi buku ke Firestore
        book_data = {
            'user_id': user_data.get('user_id'),
            'title': user_data.get('title'),
            'review': user_data.get('review'),
            'file_url': blob.public_url
        }
        db.collection('books').add(book_data)

        return jsonify({"message": "Book uploaded successfully", "file_url": blob.public_url}), 201
    else:
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

@app.route('/get_buku', methods=['GET'])
def get_buku():
    books_ref = db.collection('books')
    books = books_ref.stream()

    book_list = []
    for book in books:
        book_data = book.to_dict()
        book_data['id'] = book.id
        book_list.append(book_data)

    return jsonify({"books": book_list})

@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    book_id = request.json.get('book_id')
    if not book_id:
        return jsonify({"error": "book_id is required"}), 400

    # Mendapatkan rekomendasi buku mirip
    try:
        similar_books = get_similar_books(book_id)
        return jsonify({"recommendations": similar_books})
    except Exception as e:
        return jsonify({"error": f"Failed to generate recommendations: {str(e)}"}), 500

def get_book_data(book_id):
    book_ref = db.collection('books').document(book_id)
    book = book_ref.get()
    if not book.exists:
        raise ValueError("Book not found")
    return book.to_dict()

def prepare_input_for_model(book_data):
    features = ['feature1', 'feature2', 'feature3']  # Ganti dengan nama fitur yang sesuai
    input_data = [book_data.get(feature, 0) for feature in features]
    return np.array(input_data).reshape(1, -1)

def get_similar_books(book_id):
    book_data = get_book_data(book_id)
    input_data = prepare_input_for_model(book_data)

    # Gunakan model untuk mendapatkan rekomendasi buku mirip
    predictions = model.predict(input_data)
    return predictions.flatten().tolist()

@app.route('/rating', methods=['POST'])
def rating():
    rating_data = request.json
    book_id = rating_data.get('book_id')
    rating_value = rating_data.get('rating')

    if not book_id or rating_value is None:
        return jsonify({"error": "book_id and rating are required"}), 400

    # Simpan rating ke Firestore
    book_ref = db.collection('books').document(book_id)
    book = book_ref.get()
    if not book.exists:
        return jsonify({"error": "Book not found"}), 404

    book_ref.update({
        'ratings': firestore.ArrayUnion([rating_value])
    })

    return jsonify({"message": "Rating submitted successfully"}), 201

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
