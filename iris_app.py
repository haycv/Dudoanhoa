# iris_app.py
import numpy as np
from flask import Flask, request, render_template
import joblib

# Initialize the Flask App
app = Flask(__name__)

# Load the model, reversefactor, and scaler
model, definitions, scaler = joblib.load(open('reafforestation_model.pkl', 'rb'))

# Default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for predicting flower species
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input values from the form
    sepal_length = float(request.form['sepal.length'])
    sepal_width = float(request.form['sepal.width'])
    petal_length = float(request.form['petal.length'])
    petal_width = float(request.form['petal.width'])

    # Creating a NumPy array with the input values
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Feature scaling using the loaded scaler
    input_features_scaled = scaler.transform(input_features)

    # Making the prediction using the loaded model
    prediction = model.predict(input_features_scaled)

    # Reverse factorize to get the actual flower species
    flower_species = definitions[prediction[0]]

    return render_template('index.html', prediction_text=f'Hoa là: {flower_species}')

# Starting the Flask Server
if __name__ == "__main__":
    app.run(debug=True)
