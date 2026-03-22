"""
Shared utilities — ChurnFeatureEngineer must be importable
for model loading to work in both API and Streamlit.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom sklearn transformer: adds business-logic features."""
    def fit(self, X, y=None):
        self.avg_charge_ = X["MonthlyCharges"].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["ChargesPerTenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)
        X["IsEarlyLifecycle"] = (X["tenure"] <= 12).astype(int)
        X["AboveAvgCharges"]  = (X["MonthlyCharges"] > self.avg_charge_).astype(int)
        X["ValueRatio"]       = X["TotalCharges"] / (X["MonthlyCharges"] * (X["tenure"] + 1) + 1)
        return X
