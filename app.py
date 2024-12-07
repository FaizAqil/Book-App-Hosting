from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from io import StringIO

app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceaccountkey.json'
BUCKET_NAME = 'storage-ml-similliar'
CSV_FILE_PATH = 'dataset-book/dataset_book.csv'

def initialize_firestore():
    try:
        db = firestore.Client()
        print("Firestore initialized successfully.")
        return db
    except Exception as e:
        print(f"Error initializing Firestore: {e}")
        return None

db = initialize_firestore()

if db is None:
    print("Firestore could not be initialized. Exiting the application.")
    exit(1)  

model = tf.keras.models.load_model('book_recommendation_model.h5')

def download_blob(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    data = blob.download_as_text()
    return data

csv_data = download_blob(BUCKET_NAME, CSV_FILE_PATH)
df = pd.read_csv(StringIO(csv_data))

# Fungsi untuk menyimpan gambar ke Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "success fetching api"
        },
        "data": None
    }), 200

# Endpoint untuk upload buku dan rating
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = f"./{file.filename}"
    file.save(file_path)

    upload_blob(BUCKET_NAME, file_path, file.filename)

    book_info = {
        'name': request.form['name'],
        'id': request.form['id'],
        'author': request.form['author'],
        'rating': float(request.form['rating'].replace(',', '.')),  
        'user': request.form['user']  
    }

    try:
        db.collection('books').document(book_info['id']).set(book_info)
    except Exception as e:
        return jsonify({'error': f'Failed to save to Firestore: {e}'}), 500

    global df  
    new_entry = {
        'user': book_info['user'],
        'book': book_info['id'],
        'review/score': book_info['rating']
    }
    df = df.append(new_entry, ignore_index=True)

    book_features = df[df['title'] == book_info['name']].iloc[:, 1:].values 
    predicted_rating = model.predict(book_features)

    updated_rating = (book_info['rating'] + predicted_rating[0]) / 2

    df.loc[df['title'] == book_info['name'], 'average_rating'] = updated_rating

    os.remove(file_path)

    return jsonify({'message': 'File uploaded successfully', 'book_info': book_info, 'updated_rating': updated_rating}), 200

@app.route('/get_buku', methods=['GET'])
def get_buku():
    books_ref = db.collection('books')
    books = books_ref.stream()

    book_list = []
    for book in books:
        book_list.append(book.to_dict())

    return jsonify({'message': 'Daftar buku', 'data': book_list}), 200

@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    data = request.json
    book_title = data['book_title']

    book_features = df[df['title'] == book_title].iloc[:, 1:].values 

    predictions = model.predict(book_features)

    recommended_indices = np.argsort(predictions[0])[-5:]  
    recommended_books = df.iloc[recommended_indices]

    return jsonify(recommended_books.to_dict(orient='records')), 200

@app.route('/rating', methods=['POST'])
def rating():
    data = request.json

    user = data['user']
    book = data['book']

    review_score = df[(df['user'] == user) & (df['book'] == book)]['review/score']

    if review_score.empty:
        return jsonify({'error': 'No review found for this user and book'}), 404

    min_rating = df['review/score'].min()
    max_rating = df['review/score'].max()

    normalized_score = (review_score.values[0] - min_rating) / (max_rating - min_rating)

    return jsonify({'normalized_score': normalized_score}), 200

if __name__ == '__main__':
    app.run(debug=True)
