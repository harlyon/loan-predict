import unittest
import json
from app import app

class TestLoanApprovalAPI(unittest.TestCase):
    def setUp(self):
        # Set up the Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_valid_input(self):
        # Test with valid input data
        input_data = {
            "Age": 30,
            "Income": 50000,
            "Employment_Length": 5,
            "Public_Records": 0,
            "Loan_Amount": 10000,
            "Loan_Term": 36,
            "Loan_Purpose": 1
        }
        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Loan_Status', response.json)
        self.assertIn('message', response.json)

    def test_predict_missing_field(self):
        # Test with missing field
        input_data = {
            "Age": 30,
            "Income": 50000,
            "Employment_Length": 5,
            "Public_Records": 0,
            "Loan_Amount": 10000,
            "Loan_Term": 36
            # Missing "Loan_Purpose"
        }
        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertIn('Missing field: Loan_Purpose', response.json['details'])

    def test_predict_invalid_data_type(self):
        # Test with invalid data type (e.g., string instead of number)
        input_data = {
            "Age": "thirty",  # Invalid type
            "Income": 50000,
            "Employment_Length": 5,
            "Public_Records": 0,
            "Loan_Amount": 10000,
            "Loan_Term": 36,
            "Loan_Purpose": 1
        }
        response = self.app.post('/predict', data=json.dumps(input_data), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertIn('Invalid type for field: Age. Expected numeric value.', response.json['details'])

    def test_predict_no_input_data(self):
        # Test with no input data
        response = self.app.post('/predict', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(response.json['error'], 'No input data provided.')

if __name__ == '__main__':
    unittest.main()