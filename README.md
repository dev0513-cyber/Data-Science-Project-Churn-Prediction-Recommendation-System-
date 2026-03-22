---

# 🔮 End-to-End Data Science Project (Churn Prediction + Recommendation System)

A complete **production-style Data Science project** covering the full lifecycle — from data cleaning and exploratory analysis to machine learning, model explainability, API deployment, and a hybrid recommendation system.

---

## 🚀 Overview

This project demonstrates real-world data science skills through:

* 📊 Exploratory Data Analysis (EDA) with business insights
* 🤖 Machine Learning model building & comparison
* 🛠️ Feature engineering & debugging
* 🧠 Model explainability (SHAP, LIME)
* ⚡ End-to-end ML pipeline (scikit-learn)
* 🌐 FastAPI deployment for predictions
* 🎈 Interactive Streamlit dashboard
* 🎬 Hybrid Movie Recommendation System

---

## 📁 Project Structure

```
data-science-project-churn-prediction-recommendation-system/
│
├── api/                          # FastAPI backend
│   └── app.py
│
├── data/                         # Datasets
│   ├── telco_churn.csv
│   ├── tmdb_5000_credits.csv
│   └── tmdb_5000_movies.csv
│
├── models/                       # Trained ML models & artifacts
│   ├── churn_model_fixed.pkl
│   ├── churn_pipeline.pkl
│   └── feature_info.pkl
│
├── notebooks/                    # Jupyter notebooks (development & experiments)
│   ├── 01_EDA_Data_Cleaning.ipynb
│   ├── 02_Basic_ML_Models.ipynb
│   ├── 03_Debug_Improve.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_SQL_Analysis.ipynb
│   ├── 06_End_to_End_Pipeline.ipynb
│   ├── 07_Model_Explainability.ipynb
│   └── 08_Recommendation_System.ipynb
│
├── sql/                          # SQL queries
│   └── queries.sql
│
├── src/                          # Utility & helper modules
│   ├── __init__.py
│   └── churn_utils.py
│
├── streamlit_app/                # Streamlit frontend
│   └── app.py
│
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
├── render.yaml                   # Deployment config (if using Render)
├── README.md                     # Project documentation
└── .gitignore
```

---

## 🎯 Key Features

### 📊 Churn Prediction System

* Predicts customer churn probability
* Provides risk classification (LOW / MEDIUM / HIGH)
* Generates actionable business recommendations

### 🤖 Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost (best performing)

### 🧠 Explainability

* SHAP (global + local explanations)
* LIME (instance-level interpretation)

### ⚙️ Production Pipeline

* End-to-end `sklearn Pipeline`
* Feature preprocessing + SMOTE + model
* No data leakage

### 🌐 API (FastAPI)

* Real-time prediction endpoint
* JSON-based input/output
* Swagger UI included

### 🎈 Dashboard (Streamlit)

* Interactive churn analysis
* Batch predictions
* Visual insights

### 🎬 Recommendation System

* Content-based filtering (TF-IDF + cosine similarity)
* Collaborative filtering (SVD)
* Hybrid approach (combined scoring)

---

## 📊 Model Performance

| Metric   | Score |
| -------- | ----- |
| AUC-ROC  | 0.84  |
| F1-Score | 0.59  |
| Accuracy | 0.77  |

> Note: Dataset is imbalanced (~26% churn), so F1-score and AUC-ROC are prioritized over accuracy.

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd ds_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 📓 Run Notebooks

```bash
jupyter notebook
```

Run in order:

1. EDA
2. ML Models
3. Debugging
4. Feature Engineering
5. SQL Analysis
6. Pipeline (IMPORTANT — saves model)
7. Explainability
8. Recommendation System

---

### 🎈 Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

### ⚡ Run FastAPI

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

* API: [http://localhost:8000](http://localhost:8000)
* Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🌐 API Example

### Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.70,
  "TotalCharges": 151.65
}
```

### Response

```json
{
  "churn_prediction": true,
  "churn_probability": 0.78,
  "risk_level": "HIGH"
}
```

---

## 🚀 Deployment

### Streamlit (Free)

* Deploy via Streamlit Cloud
* Connect GitHub repo
* Select `streamlit_app/app.py`

### FastAPI (Render)

* Create Web Service
* Start command:

```bash
uvicorn api.app:app --host 0.0.0.0 --port $PORT
```

---

## 💡 Key Concepts Demonstrated

* Handling missing values & outliers
* Feature engineering impact analysis
* Model evaluation (AUC, F1, ROC curve)
* Class imbalance (SMOTE)
* Avoiding data leakage
* Hyperparameter tuning
* Explainable AI (XAI)
* REST API deployment
* Recommendation systems

---

## 🛠️ Tech Stack

* **Python** (pandas, numpy)
* **Machine Learning**: scikit-learn, XGBoost
* **Visualization**: matplotlib, seaborn
* **Explainability**: SHAP, LIME
* **API**: FastAPI, Uvicorn
* **Frontend**: Streamlit
* **Database/SQL**: SQLite

---

## 📌 Highlights

* End-to-end production-ready workflow
* Covers multiple real-world data science scenarios
* Clean modular structure (notebooks → pipeline → API → UI)
* Designed for interview discussions and demonstrations

---
