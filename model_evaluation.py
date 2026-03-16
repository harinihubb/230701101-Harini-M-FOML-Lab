import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features, FEATURE_COLS, TARGET

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

def get_feature_importance(model, feature_names):
    # Try direct, pipeline, or permutation
    try:
        m = model.named_steps["model"] if hasattr(model, "named_steps") else model
        if hasattr(m, "feature_importances_"):
            return dict(zip(feature_names, m.feature_importances_))
        elif hasattr(m, "coef_"):
            return dict(zip(feature_names, np.abs(m.coef_)))
    except Exception:
        pass
    return {}

if __name__ == "__main__":
    model = joblib.load("models/wellbeing_model.pkl")
    _, X_test, _, y_test = joblib.load("models/data_splits.pkl")
    metrics = evaluate_model(model, X_test, y_test)
    print("Evaluation:", metrics)
    fi = get_feature_importance(model, FEATURE_COLS)
    if fi:
        print("Feature importances:", sorted(fi.items(), key=lambda x: -x[1])[:5])
