import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features, FEATURE_COLS, TARGET

def train_models(data_path="data/dataset.csv"):
    df = load_and_clean(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10))]),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        results[name] = {"model": model, "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std()}
        print(f"{name}: CV R2 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")
    joblib.dump(best_model, "models/wellbeing_model.pkl")
    joblib.dump((X_train, X_test, y_train, y_test), "models/data_splits.pkl")
    return results, best_name, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_models()
