# app.py
from flask import Flask, request, jsonify
import pickle
import re
import string
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer once at the start
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess the input review text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    return text

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request
    review_text = data.get('review_text')  # Extract the review_text

    if not review_text:
        return jsonify({'error': 'No review_text provided'}), 400

    # Preprocess the review text
    processed_text = preprocess_text(review_text)

    # Transform the text using the vectorizer
    text_vectorized = vectorizer.transform([processed_text])

    # Predict the sentiment using the trained model
    sentiment = model.predict(text_vectorized)

    # Return the predicted sentiment as a JSON response
    return jsonify({'sentiment_prediction': 'positive' if sentiment[0] == 1 else 'negative'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
