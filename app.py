from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import pandas as pd
import numpy as np
from flasgger import Swagger
from flasgger.utils import swag_from

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)
# Load the trained model, scaler, and columns
with open('logistic_regression_model.pkl', 'rb') as file:
    lr_clf = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    s_sc = pickle.load(file)

with open('columns.pkl', 'rb') as file:
    training_columns = pickle.load(file)

col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_val = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

@app.route("/")
@app.route("/home")
@swag_from('swagger_home.yaml', methods=['GET'])
def home():
    return "<h1>This is Home Page</h1>"

@app.route("/predict", methods=['POST'])
@swag_from('swagger_predict.yaml', methods=['POST'])
def predict():
    msg = request.json
    age = int(msg['age'])
    sex = int(msg['sex'])
    cp = int(msg['cp'])
    trestbps = int(msg['trestbps'])
    chol = int(msg['chol'])
    fbs = int(msg['fbs'])
    restecg = int(msg['restecg'])
    thalach = int(msg['thalach'])
    exang = int(msg['exang'])
    oldpeak = float(msg['oldpeak'])  # Ensure oldpeak is a float
    slope = int(msg['slope'])
    ca = int(msg['ca'])
    thal = int(msg['thal'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Apply one-hot encoding to the input data for categorical variables (same as during training)
    input_data = pd.get_dummies(input_data, columns=categorical_val)

    # Add any missing columns and reindex to match the training data
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[training_columns]

    # Scale the numerical columns using the fitted StandardScaler
    input_data[col_to_scale] = s_sc.transform(input_data[col_to_scale])

    # Make the prediction
    prediction = lr_clf.predict(input_data)
    probability = lr_clf.predict_proba(input_data)

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': probability[0].tolist()
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)
