import unittest
from fastapi.testclient import TestClient
from fastapi import status
from app import app, predictor
from unittest.mock import patch, MagicMock

class TestLoanApprovalAPI(unittest.TestCase):
    def setUp(self):
        # Set up the FastAPI test client
        self.client = TestClient(app)
        
        # Sample prediction result
        self.sample_prediction = {
            "loan_status": 1,
            "probability": 0.95,
            "message": "Loan approved"
        }

    @patch('app.predictor')
    def test_predict_valid_input(self, mock_predictor):
        # Mock the predictor's predict_single method
        mock_predictor.predict_single.return_value = self.sample_prediction
        
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
        
        response = self.client.post('/predict', json=input_data)
        self.assertEqual(response.status_code, 200)
        
        # Check response structure
        data = response.json()
        self.assertIn('loan_status', data)
        self.assertIn('probability', data)
        self.assertIn('message', data)
        
        # Check types
        self.assertIsInstance(data['loan_status'], int)
        self.assertIsInstance(data['probability'], float)
        self.assertIsInstance(data['message'], str)

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
        
        response = self.client.post('/predict', json=input_data)
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        # Check error details
        error_data = response.json()
        self.assertEqual(error_data['status'], 'error')
        self.assertEqual(error_data['message'], 'Validation error')
        self.assertIn('detail', error_data)
        errors = error_data['detail']
        self.assertTrue(any(error.get('field') == 'Loan_Purpose' for error in errors))

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
        response = self.client.post('/predict', json=input_data)
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        # Check error details
        error_data = response.json()
        self.assertEqual(error_data['status'], 'error')
        self.assertEqual(error_data['message'], 'Validation error')
        self.assertIn('detail', error_data)
        errors = error_data['detail']
        self.assertTrue(any(error.get('field') == 'Age' for error in errors))

    def test_predict_no_input_data(self):
        # Test with no input data
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        # Check error details
        error_data = response.json()
        self.assertEqual(error_data['status'], 'error')
        self.assertEqual(error_data['message'], 'Validation error')
        self.assertIn('detail', error_data)
        errors = error_data['detail']
        required_fields = ['Age', 'Income', 'Employment_Length', 'Public_Records', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose']
        for field in required_fields:
            self.assertTrue(any(error.get('field') == field for error in errors))

    @patch('app.predictor')
    def test_model_info(self, mock_predictor):
        # Mock the predictor's get_model_info method with string keys for loan_purpose_mapping
        mock_model_info = {
            'model_type': 'XGBClassifier',
            'features': ['Age', 'Income', 'Employment_Length', 'Public_Records', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose'],
            'scaled_features': ['Income', 'Loan_Amount', 'Loan_Term'],
            'loan_purpose_mapping': {
                '0': 'medical',
                '1': 'debt_consolidation',
                '2': 'car',
                '3': 'home_improvement'
            },
            'status': 'loaded',
            'timestamp': '2023-01-01T00:00:00+00:00'
        }
        mock_predictor.get_model_info.return_value = mock_model_info
        
        # Test model info endpoint
        response = self.client.get('/model-info')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        response_data = response.json()
        self.assertEqual(response_data['status'], 'success')
        self.assertIn('data', response_data)
        
        data = response_data['data']
        self.assertEqual(data['model_type'], 'XGBClassifier')
        self.assertEqual(data['status'], 'loaded')
        
        # Verify required fields are present
        required_fields = ['model_type', 'features', 'scaled_features', 'loan_purpose_mapping', 'status', 'timestamp']
        for field in required_fields:
            self.assertIn(field, data)
        
        # Verify loan purpose mapping has expected values (now using string keys)
        expected_loan_purposes = {
            "0": "medical",
            "1": "debt_consolidation",
            "2": "car",
            "3": "home_improvement"
        }
        self.assertEqual(data['loan_purpose_mapping'], expected_loan_purposes)

    @patch('app.predictor')
    def test_health_check(self, mock_predictor):
        # Test health check endpoint
        response = self.client.get('/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('model_loaded', data)
        
        # Test that timestamp is in ISO format with timezone
        from datetime import datetime
        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            self.fail("Timestamp is not in valid ISO format")

class TestBatchPrediction(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.valid_application = {
            "Age": 30,
            "Income": 50000,
            "Employment_Length": 5,
            "Public_Records": 0,
            "Loan_Amount": 10000,
            "Loan_Term": 36,
            "Loan_Purpose": 1
        }

    @patch('app.predictor')
    def test_batch_predict_success(self, mock_predictor):
        # Mock the predictor's batch prediction
        mock_predictor.predict_batch.return_value = [
            {"application_id": 0, "loan_status": 1, "probability": 0.92, "message": "Loan approved"},
            {"application_id": 1, "loan_status": 0, "probability": 0.45, "message": "Loan not approved"}
        ]
        
        input_data = [self.valid_application, {**self.valid_application, "Loan_Amount": 20000}]
        response = self.client.post('/batch-predict', json=input_data)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('predictions', data)
        self.assertEqual(len(data['predictions']), 2)
        self.assertEqual(data['predictions'][0]['loan_status'], 1)
        self.assertEqual(data['predictions'][1]['loan_status'], 0)

    @patch('app.predictor')
    def test_batch_predict_invalid_input(self, mock_predictor):
        # Test with invalid input (missing required field)
        invalid_application = {**self.valid_application}
        invalid_application.pop('Loan_Purpose')
        
        response = self.client.post('/batch-predict', json=[invalid_application])
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        # Check error details
        error_data = response.json()
        self.assertEqual(error_data['status'], 'error')
        self.assertEqual(error_data['message'], 'Validation error')
        self.assertIn('detail', error_data)
        errors = error_data['detail']
        self.assertTrue(any('Loan_Purpose' in str(error.get('field', '')) for error in errors))

if __name__ == '__main__':
    unittest.main()
    # To run the tests from the command line:
    # python -m pytest test_app.py -v