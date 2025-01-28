Sentiment Analysis Project

Project Setup

1. Install Dependencies
To get started, you need to install the necessary dependencies. You can do this by running the following command:

pip install -r requirements.txt

This will install all the required Python libraries for the project, including Flask, Scikit-learn, SQLite, etc.

2. Set Up the Database
The database (imdb_reviews.db) is used to store movie reviews and their sentiment labels (positive or negative). The database is set up automatically when you run the data_setup.ipynb notebook.

python data_setup.ipynb

This will download the IMDB dataset, process it, and create the necessary table (imdb_reviews) in the SQLite database.

Data Acquisition

The dataset used in this project is the IMDB movie reviews dataset, which contains labeled movie reviews (positive or negative).
The dataset is loaded using the datasets library from Hugging Face.

from datasets import load_dataset
dataset = load_dataset("imdb")

Run Instructions

1. Running the Training Script
Run the script as follows:

python train_model.ipynb

This script will load the IMDB dataset, preprocess the text, train a machine learning model (Logistic Regression in this case), and save the trained model(model.pkl) and vectorizer (vectorizer.pkl).

2. Starting the Flask Server
To deploy the trained model via a simple REST API, run the app.py script to start the Flask server.

python app.py

This will start a Flask server that listens on http://127.0.0.1:5000/.

3. Testing the Endpoint
Once the Flask server is running, you can test the API endpoint using a POST request,run the test_request.py script:

python test_request.py 

This will send a review to the /predict endpoint and receive the sentiment prediction (either positive or negative).

Model Info

Model Approach: The sentiment analysis model is based on Logistic Regression and uses TF-IDF features extracted from the cleaned review text. The training data consists of movie reviews labeled as positive or negative.

Key Results: The model is evaluated on a test set, and performance metrics like accuracy and F1-score are computed. For example:
Accuracy: 85.2%
F1-Score: 0.85

Additional Assets: EDA (Exploratory Data Analysis)

In the EDA.ipynb file, I perform an exploratory data analysis on the IMDB movie reviews dataset. Below is an overview of the steps followed in the notebook:

1. Data Loading and Cleaning
I load the data from the SQLite database (imdb_reviews.db) and clean the review text by:
Removing HTML tags.
Removing special characters and digits.
Converting the text to lowercase.
Removing extra spaces.

2. Sentiment Distribution
I visualize the distribution of sentiments (positive vs negative) in the dataset using a bar plot.

3. Average Review Length by Sentiment
I calculate the average length of reviews for each sentiment category (positive and negative) and visualize the comparison with a bar plot.

4. Word Cloud for Reviews
I generate word clouds for both positive and negative reviews to visualize the most frequent words in each category.

