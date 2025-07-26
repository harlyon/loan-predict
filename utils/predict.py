import joblib
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import pytz

# Configure logging
logger = logging.getLogger(__name__)

class LoanPredictor:
    """
    A class to handle loan prediction using a trained model and scaler.
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize the LoanPredictor with model and scaler paths.
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the scaler file
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._load_model_and_scaler()
    
    def _load_model_and_scaler(self) -> None:
        """
        Load the model and scaler from the specified paths.
        """
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Successfully loaded the model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        try:
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Successfully loaded the scaler")
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            raise
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Preprocess the input data for prediction.
        
        Args:
            input_data: Dictionary containing loan application data
            
        Returns:
            pd.DataFrame: Processed DataFrame ready for prediction
        """
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        required_columns = [
            'Age', 'Income', 'Employment_Length', 
            'Public_Records', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose'
        ]
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Get all features that were used during training
        all_features = [
            'Age', 'Income', 'Employment_Length', 
            'Public_Records', 'Loan_Amount', 'Loan_Term', 'Loan_Purpose'
        ]
        
        # Scale all features that were used during training
        try:
            # Make a copy of the dataframe with all features in the correct order
            df_scaled = df[all_features].copy()
            
            # Scale the features (the scaler was fit on all features during training)
            scaled_values = self.scaler.transform(df_scaled)
            
            # Update the dataframe with scaled values
            df[all_features] = scaled_values
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise ValueError(f"Error in feature scaling: {str(e)}")
        
        return df
    
    def predict_single(self, application_data: Dict) -> Dict:
        """
        Predict loan approval for a single application.
        
        Args:
            application_data: Dictionary containing loan application data
            
        Returns:
            Dict: Prediction result with status, probability, and message
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not loaded")
        
        try:
            # Preprocess input
            df = self.preprocess_input(application_data)
            
            # Make prediction
            prediction = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            
            # Get probability of the predicted class
            probability = float(np.max(probabilities))
            
            # Prepare result
            status = int(prediction[0])
            message = "Loan approved" if status == 1 else "Loan not approved"
            
            return {
                "loan_status": status,
                "probability": probability,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, applications: List[Dict]) -> List[Dict]:
        """
        Predict loan approval for multiple applications.
        
        Args:
            applications: List of dictionaries containing loan application data
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for i, app in enumerate(applications):
            try:
                result = self.predict_single(app)
                result["application_id"] = i
                results.append(result)
            except Exception as e:
                results.append({
                    "application_id": i,
                    "error": f"Error processing application: {str(e)}"
                })
                logger.error(f"Error processing application {i}: {str(e)}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict: Model information including features, mappings, and status.
        """
        model_loaded = self.model is not None and self.scaler is not None
        
        # Create a dictionary with string keys to ensure consistent JSON serialization
        loan_purpose_mapping = {
            "0": "medical",
            "1": "debt_consolidation",
            "2": "car",
            "3": "home_improvement"
        }
        
        return {
            "model_type": type(self.model).__name__ if self.model else None,
            "features": ["Age", "Income", "Employment_Length", "Public_Records", 
                        "Loan_Amount", "Loan_Term", "Loan_Purpose"],
            "scaled_features": ["Income", "Loan_Amount", "Loan_Term"],
            "loan_purpose_mapping": loan_purpose_mapping,
            "status": "loaded" if model_loaded else "not loaded",
            "timestamp": datetime.now(pytz.utc).isoformat()
        }
