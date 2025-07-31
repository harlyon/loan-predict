# Loan Approval Prediction API

A FastAPI-based service that provides automated loan approval predictions using machine learning. This solution helps financial institutions make faster, data-driven lending decisions while reducing risk and improving operational efficiency.

## Problem Statement

Financial institutions face significant challenges in accurately assessing loan applications. Traditional processes are often slow, subjective, and prone to human bias, leading to either increased default rates or lost business opportunities. This API addresses these challenges by providing:

- **Automated Decision Making**: Instant loan approval/rejection predictions
- **Risk Assessment**: Data-driven evaluation of applicant creditworthiness
- **Consistency**: Eliminates human bias in lending decisions
- **Scalability**: Handles high volumes of applications with minimal processing time

## Demo
- App: https://stackblitzstartersvorawhbb-a4gy--8080--96435430.local-credentialless.webcontainer.io/
- Api: https://loan-predict-q89g.onrender.com/

## Features

- Predict loan approval status for individual applications
- Batch prediction for multiple applications
- Input validation using Pydantic models
- Detailed error handling and logging
- CORS enabled for cross-origin requests
- Model and API health checks

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd loan-predict
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place model files**
   - Place your trained model as `models/loan_approval_model.pkl`
   - Place your scaler as `models/scaler.pkl`
   - The application will create the `models` directory if it doesn't exist

## Running the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **GET /**
  - Check if the API is running

### 2. Model Information
- **GET /model-info**
  - Get information about the loaded model and features

### 3. Single Prediction
- **POST /predict**
  - Predict loan approval for a single application
  - **Request Body**:
    ```json
    {
        "Age": 30,
        "Income": 50000,
        "Employment_Length": 5,
        "Public_Records": 0,
        "Loan_Amount": 10000,
        "Loan_Term": 36,
        "Loan_Purpose": 1
    }
    ```
  - **Loan_Purpose Mapping**:
    - 0: medical
    - 1: debt_consolidation
    - 2: car
    - 3: home_improvement

### 4. Batch Prediction
- **POST /batch-predict**
  - Predict loan approval for multiple applications
  - **Request Body**:
    ```json
    [
        {
            "Age": 30,
            "Income": 50000,
            "Employment_Length": 5,
            "Public_Records": 0,
            "Loan_Amount": 10000,
            "Loan_Term": 36,
            "Loan_Purpose": 1
        },
        {
            "Age": 40,
            "Income": 75000,
            "Employment_Length": 8,
            "Public_Records": 1,
            "Loan_Amount": 20000,
            "Loan_Term": 60,
            "Loan_Purpose": 2
        }
    ]
    ```

## Testing

Run the test suite with:

```bash
pytest test_app.py -v
```

## API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

The following environment variables can be set:

- `MODEL_PATH`: Path to the model file (default: `models/loan_approval_model.pkl`)
- `SCALER_PATH`: Path to the scaler file (default: `models/scaler.pkl`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Deployment

For production deployment, consider using:
- Gunicorn with Uvicorn workers
- Nginx as a reverse proxy
- Environment variables for configuration
- Containerization with Docker

## License

[Your License Here]
