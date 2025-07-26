from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Union
import os
from datetime import datetime, timezone
import logging
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the LoanPredictor utility
from utils.predict import LoanPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "loan_approval_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

# Initialize LoanPredictor
try:
    predictor = LoanPredictor(MODEL_PATH, SCALER_PATH)
    logger.info("Successfully initialized LoanPredictor")
except Exception as e:
    logger.error(f"Failed to initialize LoanPredictor: {e}")
    predictor = None

# Request/Response Models
class LoanApplication(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 30,
                "Income": 50000,
                "Employment_Length": 5,
                "Public_Records": 0,
                "Loan_Amount": 10000,
                "Loan_Term": 36,
                "Loan_Purpose": 1
            }
        }
    )
    
    Age: int = Field(..., gt=0, le=100, description="Age of the applicant")
    Income: float = Field(..., gt=0, description="Annual income of the applicant")
    Employment_Length: int = Field(..., ge=0, description="Length of employment in years")
    Public_Records: int = Field(..., ge=0, description="Number of public records")
    Loan_Amount: float = Field(..., gt=0, description="Requested loan amount")
    Loan_Term: int = Field(..., gt=0, description="Loan term in months")
    Loan_Purpose: int = Field(..., ge=0, le=3, description="Purpose of the loan (0: medical, 1: debt_consolidation, 2: car, 3: home_improvement)")
    
    @field_validator('Loan_Purpose')
    def validate_loan_purpose(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError('Loan_Purpose must be between 0 and 3')
        return v

class PredictionResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "loan_status": 1,
                "probability": 0.95,
                "message": "Loan approved"
            }
        }
    )
    
    loan_status: int = Field(..., description="1 if approved, 0 if not approved")
    probability: float = Field(..., ge=0, le=1, description="Confidence score of the prediction")
    message: str = Field(..., description="Human-readable prediction result")

# Dependency to get the predictor
async def get_predictor():
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    return predictor

# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        # Get the field location, handling both body and path parameters
        loc = error["loc"]
        if len(loc) > 1 and loc[0] in ("body", "query", "path"):
            field = ".".join(str(l) for l in loc[1:] if l != "body")
        else:
            field = ".".join(str(l) for l in loc if l != "body")
            
        errors.append({
            "field": field or "request body",
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "value_error")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            "status": "error",
            "message": "Validation error"
        },
    )

# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "message": "Loan Approval Prediction API is running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": predictor is not None
    }

# Model info endpoint
@app.get("/model-info", tags=["Model"])
async def get_model_info(predictor: LoanPredictor = Depends(get_predictor)):
    """
    Get information about the loaded model.
    
    Returns:
        Dict containing model information including features, mappings, and status.
    """
    try:
        model_info = predictor.get_model_info()
        return {
            "status": "success",
            "data": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to get model information",
                "error": str(e)
            }
        )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_loan_status(
    application: LoanApplication,
    predictor: LoanPredictor = Depends(get_predictor)
):
    """
    Predict loan approval status for a single application.
    
    - **Age**: Age of the applicant (1-100)
    - **Income**: Annual income of the applicant (> 0)
    - **Employment_Length**: Length of employment in years (>= 0)
    - **Public_Records**: Number of public records (>= 0)
    - **Loan_Amount**: Requested loan amount (> 0)
    - **Loan_Term**: Loan term in months (> 0)
    - **Loan_Purpose**: Purpose of the loan (0-3)
    """
    try:
        return predictor.predict_single(application.model_dump())
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error during prediction"}
        )

# Batch prediction endpoint
@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(
    applications: List[LoanApplication],
    predictor: LoanPredictor = Depends(get_predictor)
):
    """
    Predict loan approval status for multiple applications in a single request.
    
    Accepts an array of loan applications and returns predictions for each one.
    If an error occurs for a specific application, it will be noted in the response
    without failing the entire batch.
    """
    try:
        # Convert Pydantic models to dictionaries
        app_dicts = [app.model_dump() for app in applications]
        
        # Get predictions
        results = predictor.predict_batch(app_dicts)
        
        return {"predictions": results}
        
    except ValueError as e:
        logger.error(f"Validation error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(e)}
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error during batch prediction"}
        )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 10000))
    
    # Always use 0.0.0.0 for Render
    host = "0.0.0.0"
    
    # Log the startup information
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Use 1 worker for Render
        log_level="info"
    )
