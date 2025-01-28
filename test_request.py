import requests

url = "http://127.0.0.1:5000/predict"
data = {"review_text": "The movie was absolutely amazing!"}  # Test input

response = requests.post(url, json=data)

print("Response:", response.json())
