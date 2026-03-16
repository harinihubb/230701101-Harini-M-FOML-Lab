import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features, FEATURE_COLS

def run_clustering(df, n_clusters=3):
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)
    return df, kmeans, scaler

def run_pca(df, scaler):
    X_scaled = scaler.transform(df[FEATURE_COLS])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df = df.copy()
    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]
    return df, pca

CLUSTER_LABELS = {
    0: "High Stimulation",
    1: "Balanced & Productive",
    2: "Rested & Focused",
}

if __name__ == "__main__":
    raw = load_and_clean()
    df = engineer_features(raw)
    df, km, sc = run_clustering(df)
    print(df.groupby("cluster")[["daily_screen_time","sleep_hours","focus_score","digital_wellbeing_score"]].mean())
