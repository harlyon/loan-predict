from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flasgger import Swagger
import logging

# Initialize Flask app
app = Flask(__name__)
Swagger(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = 'loan_approval_model.pkl'
SCALER_PATH = 'scaler.pkl'
EXPECTED_FEATURE_ORDER = ['Age', 'Income', 'Employment_Length', 'Public_Records', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose']
REQUIRED_FIELDS = EXPECTED_FEATURE_ORDER

# Load the model and scaler
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    logger.info("Model and scaler loaded successfully.")
except FileNotFoundError:
    logger.error(f"Error: Model or scaler file not found. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the correct directory.")
    exit()
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    exit()

# Helper function to validate input data
def validate_input(data: dict) -> list:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f'Missing field: {field}')
        else:
            try:
                float(data[field])  # Ensure all fields are numeric
            except (ValueError, TypeError):
                errors.append(f'Invalid type for field: {field}. Expected numeric value.')
    return errors

# Root route
@app.route('/')
def home():
    return "Welcome to the Loan Approval API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Loan Approval
    ---
    tags:
      - Predictions
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Age:
              type: number
              example: 30
            Income:
              type: number
              example: 50000
            Employment_Length:
              type: number
              example: 5
            Public_Records:
              type: number
              example: 0
            Loan_Amount:
              type: number
              example: 10000
            Loan_Term:
              type: number
              example: 36
            Loan_Purpose:
              type: number
              example: 1
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            Loan_Status:
              type: integer
              example: 1
            message:
              type: string
              example: "Loan will likely be approved"
      400:
        description: Invalid input
      500:
        description: Internal server error
    """
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400

        # Validate input data
        errors = validate_input(data)
        if errors:
            return jsonify({'error': 'Invalid input', 'details': errors}), 400

        # Convert data to DataFrame and reorder columns
        input_df = pd.DataFrame([data])[EXPECTED_FEATURE_ORDER]

        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Prepare response
        loan_status = "Loan will likely be approved" if prediction[0] == 1 else "Loan will unlikely be approved"
        output = {'Loan_Status': int(prediction[0]), 'message': loan_status}

        return jsonify(output), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)