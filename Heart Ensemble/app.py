import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, redirect, url_for
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models and scaler
model = pickle.load(open('voting_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the POST request
        data = {key: float(request.form[key]) for key in request.form}

        # Ensure the data contains all required fields (columns)
        required_fields = ['cp_0', 'cp_1', 'cp_2', 'cp_3', 'trestbps', 'chol', 
                           'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 
                           'thalach', 'exang_0', 'exang_1']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Convert the data into a DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess the input data (scaling it using the loaded scaler)
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the loaded ensemble model
        prediction = model.predict(scaled_data)

        # Render the prediction result on result.html
        return render_template('result.html', prediction=int(prediction[0]))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)