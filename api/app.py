"""
Customer Churn Prediction API
Built with FastAPI — Production Ready
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.churn_utils import ChurnFeatureEngineer  # noqa: F401 — needed for joblib unpickling
import joblib, numpy as np, pandas as pd

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## Churn Prediction API
    Predict whether a telecom customer will churn.
    Built as part of Data Science Portfolio Project.

    ### Endpoints:
    - `GET  /`           — Health check
    - `POST /predict`    — Predict churn for a single customer
    - `POST /predict/batch` — Predict for multiple customers
    - `GET  /model/info` — Model information
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_pipeline.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model load failed: {e}")

# ── Input Schema ───────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: str = Field("Male", description="Male or Female")
    SeniorCitizen: int = Field(0, description="0 = No, 1 = Yes")
    Partner: str = Field("Yes", description="Yes or No")
    Dependents: str = Field("No", description="Yes or No")
    tenure: int = Field(12, description="Months with company (0–72)", ge=0, le=72)
    PhoneService: str = Field("Yes", description="Yes or No")
    MultipleLines: str = Field("No", description="Yes, No, or No phone service")
    InternetService: str = Field("Fiber optic", description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field("No", description="Yes, No, or No internet service")
    OnlineBackup: str = Field("No", description="Yes, No, or No internet service")
    DeviceProtection: str = Field("No", description="Yes, No, or No internet service")
    TechSupport: str = Field("No", description="Yes, No, or No internet service")
    StreamingTV: str = Field("No", description="Yes, No, or No internet service")
    StreamingMovies: str = Field("No", description="Yes, No, or No internet service")
    Contract: str = Field("Month-to-month", description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field("Yes", description="Yes or No")
    PaymentMethod: str = Field("Electronic check",
                               description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    MonthlyCharges: float = Field(70.0, description="Monthly bill amount in USD", ge=0)
    TotalCharges: float = Field(840.0, description="Total billed so far in USD", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70, "TotalCharges": 151.65,
            }
        }

class BatchInput(BaseModel):
    customers: list[CustomerInput]

# ── Predict Helper ─────────────────────────────────────────────────────────────
def predict_customer(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([customer.dict()])
    prob = float(model.predict_proba(df)[0][1])
    pred = int(model.predict(df)[0])

    risk_level = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
    
    recommendations = []
    if customer.Contract == "Month-to-month":
        recommendations.append("Offer a discounted annual contract to increase commitment")
    if customer.tenure <= 12:
        recommendations.append("Customer is in high-risk early lifecycle — trigger retention campaign")
    if customer.MonthlyCharges > 70:
        recommendations.append("High charges — offer loyalty discount or bundle deal")
    if customer.InternetService == "Fiber optic" and customer.OnlineSecurity == "No":
        recommendations.append("Upsell security/backup — increases stickiness")
    if not recommendations:
        recommendations.append("Customer appears stable — maintain regular engagement")

    return {
        "churn_prediction": bool(pred),
        "churn_probability": round(prob, 4),
        "churn_probability_pct": f"{prob*100:.1f}%",
        "risk_level": risk_level,
        "recommendations": recommendations,
    }

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "API is running",
        "name":   "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs":   "/docs",
    }

@app.post("/predict", tags=["Prediction"])
def predict_single(customer: CustomerInput):
    """
    Predict churn for a single customer.
    Returns probability, risk level, and retention recommendations.
    """
    result = predict_customer(customer)
    return {"customer": customer.dict(), "prediction": result}

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict churn for multiple customers in one request.
    """
    results = []
    for i, customer in enumerate(batch.customers):
        res = predict_customer(customer)
        res["customer_index"] = i
        results.append(res)
    
    churn_count = sum(r["churn_prediction"] for r in results)
    return {
        "total_customers": len(results),
        "predicted_churners": churn_count,
        "churn_rate_pct": f"{churn_count/len(results)*100:.1f}%",
        "predictions": results,
    }

@app.get("/model/info", tags=["Model"])
def model_info():
    """Get information about the deployed model."""
    return {
        "model_type": "XGBoost Classifier (inside sklearn Pipeline)",
        "pipeline_steps": ["ChurnFeatureEngineer", "ColumnTransformer (Scaler+OHE)", "SMOTE", "XGBoostClassifier"],
        "dataset": "Telco Customer Churn (7043 customers)",
        "training_metrics": {
            "F1_Score": "0.59",
            "AUC_ROC": "0.84",
        },
        "features_used": [
            "tenure", "MonthlyCharges", "TotalCharges",
            "Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "SeniorCitizen", "Partner",
            "+ 4 engineered features"
        ],
        "target": "Churn (Yes/No)",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
