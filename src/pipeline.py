"""
================================================================================
  DIGITAL WELLBEING ML ANALYSIS  ·  src/pipeline.py
================================================================================
End-to-end pipeline:
  1. Data Loading
  2. Data Cleaning
  3. Exploratory Data Analysis (EDA)
  4. Feature Engineering
  5. Data Preparation
  6. Model Training & Evaluation
  7. Clustering (K-Means)
  8. Dimensionality Reduction (PCA)
  9. Visualisations
 10. Model Saving
================================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data",          "digital_wellbeing_dataset.csv")
VIZ    = os.path.join(BASE, "visualizations")
MDL    = os.path.join(BASE, "models")
os.makedirs(VIZ, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

FEATURES = [
    "daily_screen_time", "num_app_switches", "sleep_hours",
    "notification_count", "social_media_time_min",
    "focus_score", "mood_score", "anxiety_level",
]
TARGET = "digital_wellbeing_score"

# ─────────────────────────────────────────────────────────────────────────────
# 1 · DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1 · DATA LOADING")
print("=" * 70)
df = pd.read_csv(DATA)
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(df.dtypes)

# ─────────────────────────────────────────────────────────────────────────────
# 2 · DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2 · DATA CLEANING")
print("=" * 70)

# 2a) Missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
print("  → Filled with column medians.")

# 2b) Duplicate rows
dups = df.duplicated().sum()
print(f"\n  Duplicate rows: {dups}  → dropping")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 2c) Outlier removal using IQR
print("\n  Outlier removal (IQR method):")
before = len(df)
for col in FEATURES + [TARGET]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 3 * IQR) & (df[col] <= Q3 + 3 * IQR)]
df.reset_index(drop=True, inplace=True)
print(f"  Rows before: {before}  |  Rows after: {len(df)}")

# 2d) Data types
df = df.astype({
    "num_app_switches":  int,
    "notification_count": int,
})
print("\n  Final dtypes:")
print(df.dtypes)
print("\n  Descriptive statistics:")
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 3 · EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3 · EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 3a) Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "01_correlation_heatmap.png"), dpi=150)
plt.close()
print("  ✓ Saved: 01_correlation_heatmap.png")

# 3b) Distribution plots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(FEATURES + [TARGET]):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("muted")[i % 6])
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel("")
plt.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "02_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: 02_distributions.png")

# 3c) Scatter plots vs target
key_features = ["sleep_hours", "daily_screen_time", "mood_score",
                "anxiety_level", "focus_score", "notification_count"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    axes[i].scatter(df[feat], df[TARGET], alpha=0.4, edgecolors="none",
                    color=sns.color_palette("deep")[i])
    m, b = np.polyfit(df[feat], df[TARGET], 1)
    xs = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[i].plot(xs, m * xs + b, "r--", lw=2)
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel("Digital Wellbeing Score", fontsize=10)
    axes[i].set_title(f"{feat} vs Wellbeing", fontsize=12)
plt.suptitle("Key Feature vs Digital Wellbeing Score", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "03_scatter_plots.png"), dpi=150)
plt.close()
print("  ✓ Saved: 03_scatter_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4 · FEATURE ENGINEERING")
print("=" * 70)

# dopamine_index: proxy for stimulus-seeking behaviour.
# High values → overloaded reward system → lower wellbeing.
df["dopamine_index"] = (
    df["social_media_time_min"] + df["notification_count"] + df["num_app_switches"]
)

# focus_efficiency: quality of screen time (are you productive?).
# Higher = more focused per hour of screen use → better wellbeing.
df["focus_efficiency"] = df["focus_score"] / (df["daily_screen_time"] + 1e-9)

# sleep_deficit: deviation from recommended 8 h.
# Positive values = under-sleep → elevated anxiety & lower mood.
df["sleep_deficit"] = 8 - df["sleep_hours"]

ENGINEERED = ["dopamine_index", "focus_efficiency", "sleep_deficit"]
ALL_FEATURES = FEATURES + ENGINEERED

print("  New features:")
for f in ENGINEERED:
    print(f"    {f}: mean={df[f].mean():.2f}, std={df[f].std():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5 · DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5 · DATA PREPARATION")
print("=" * 70)

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train set: {X_train.shape}  |  Test set: {X_test.shape}")
joblib.dump(scaler, os.path.join(MDL, "scaler.pkl"))
print("  ✓ Scaler saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 6 · MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6 · MODEL TRAINING & EVALUATION")
print("=" * 70)

models = {
    "Linear Regression":       LinearRegression(),
    "Decision Tree":           DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":           RandomForestRegressor(n_estimators=150, random_state=42),
    "Support Vector Regression": SVR(kernel="rbf", C=10, epsilon=0.5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2, "model": model, "preds": preds}
    print(f"  {name:<30} MSE={mse:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

# Best model by R²
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n  ★ Best model: {best_name}  (R²={results[best_name]['R2']:.4f})")
joblib.dump(best_model, os.path.join(MDL, "best_model.pkl"))
print("  ✓ Best model saved.")

# Model comparison bar chart
metrics_df = pd.DataFrame(
    {k: {m: results[k][m] for m in ("MSE", "RMSE", "R2")} for k in results}
).T

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = sns.color_palette("deep", len(models))
for ax, metric in zip(axes, ["MSE", "RMSE", "R2"]):
    bars = ax.bar(metrics_df.index, metrics_df[metric], color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xticklabels(metrics_df.index, rotation=20, ha="right", fontsize=9)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "04_model_comparison.png"), dpi=150)
plt.close()
print("  ✓ Saved: 04_model_comparison.png")

# Actual vs Predicted (best model)
best_preds = results[best_name]["preds"]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, best_preds, alpha=0.5, edgecolors="none", color="#4C72B0")
lims = [min(y_test.min(), best_preds.min()) - 2,
        max(y_test.max(), best_preds.max()) + 2]
ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
ax.set_xlabel("Actual Wellbeing Score", fontsize=12)
ax.set_ylabel("Predicted Wellbeing Score", fontsize=12)
ax.set_title(f"Actual vs Predicted  [{best_name}]", fontsize=13, fontweight="bold")
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims)
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "05_actual_vs_predicted.png"), dpi=150)
plt.close()
print("  ✓ Saved: 05_actual_vs_predicted.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7 · FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7 · FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_)
else:
    importances = np.ones(len(ALL_FEATURES))

imp_df = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": importances})
imp_df.sort_values("Importance", ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
               color=sns.color_palette("viridis", len(imp_df)), edgecolor="white")
ax.set_xlabel("Importance", fontsize=12)
ax.set_title(f"Feature Importance  [{best_name}]", fontsize=14, fontweight="bold")
for bar, val in zip(bars, imp_df["Importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "06_feature_importance.png"), dpi=150)
plt.close()
print("  Top 5 important features:")
print(imp_df.tail(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8 · CLUSTERING (K-Means)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8 · CLUSTERING ANALYSIS")
print("=" * 70)

X_scaled = scaler.transform(X)

# Elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, "bo-", lw=2, ms=8)
ax.set_xlabel("Number of clusters (k)", fontsize=12)
ax.set_ylabel("Inertia", fontsize=12)
ax.set_title("Elbow Method for Optimal k", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "07_kmeans_elbow.png"), dpi=150)
plt.close()
print("  ✓ Saved: 07_kmeans_elbow.png")

# Fit with k=4
K_OPT = 4
km = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

cluster_summary = df.groupby("cluster")[
    ["daily_screen_time", "sleep_hours", "social_media_time_min",
     "focus_score", "mood_score", "anxiety_level", TARGET]
].mean().round(2)
print("\n  Cluster Profiles:")
print(cluster_summary.to_string())

cluster_labels = {
    0: "Balanced Users",
    1: "High-Screen Low-Sleep",
    2: "Healthy & Focused",
    3: "Anxious Overloaded",
}
# Sort by wellbeing to assign labels
order = cluster_summary[TARGET].argsort().values
label_map = {order[i]: list(cluster_labels.values())[i] for i in range(K_OPT)}
df["cluster_label"] = df["cluster"].map(label_map)

# ─────────────────────────────────────────────────────────────────────────────
# 9 · PCA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("9 · DIMENSIONALITY REDUCTION (PCA)")
print("=" * 70)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  Explained variance ratio: {pca.explained_variance_ratio_.round(4)}")
print(f"  Total explained: {pca.explained_variance_ratio_.sum():.2%}")

# PCA scatter coloured by cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
palette = sns.color_palette("Set2", K_OPT)

# By cluster
for c in range(K_OPT):
    mask = df["cluster"] == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=label_map[c], alpha=0.6, s=30, color=palette[c])
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
axes[0].set_title("PCA – Coloured by Cluster", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9)

# By wellbeing score
sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df[TARGET], cmap="RdYlGn", alpha=0.6, s=30)
plt.colorbar(sc, ax=axes[1], label="Wellbeing Score")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=11)
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=11)
axes[1].set_title("PCA – Coloured by Wellbeing Score", fontsize=13, fontweight="bold")

plt.suptitle("PCA Visualisation", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "08_pca_scatter.png"), dpi=150)
plt.close()
print("  ✓ Saved: 08_pca_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10 · CLUSTER PROFILE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
cluster_plot_df = (
    df.groupby("cluster_label")[
        ["sleep_hours", "daily_screen_time", "focus_score",
         "mood_score", "anxiety_level", TARGET]
    ].mean()
)

fig, ax = plt.subplots(figsize=(12, 6))
cluster_plot_df.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Cluster Profiles – Mean Feature Values", fontsize=14, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "09_cluster_profiles.png"), dpi=150)
plt.close()
print("  ✓ Saved: 09_cluster_profiles.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11 · SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(pca,        os.path.join(MDL, "pca.pkl"))
joblib.dump(km,         os.path.join(MDL, "kmeans.pkl"))
joblib.dump(ALL_FEATURES, os.path.join(MDL, "feature_names.pkl"))
df.to_csv(os.path.join(BASE, "data", "cleaned_dataset.csv"), index=False)
print("\n  ✓ All models, scaler, PCA, KMeans saved to models/")
print("  ✓ Cleaned dataset saved to data/cleaned_dataset.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 12 · PREDICTION SYSTEM (CLI demo)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("12 · SAMPLE PREDICTION (CLI)")
print("=" * 70)

sample = {
    "daily_screen_time":     8.0,
    "num_app_switches":      60,
    "sleep_hours":           6.0,
    "notification_count":    80,
    "social_media_time_min": 150.0,
    "focus_score":           55.0,
    "mood_score":            60.0,
    "anxiety_level":         6.0,
}
sample["dopamine_index"]   = sample["social_media_time_min"] + sample["notification_count"] + sample["num_app_switches"]
sample["focus_efficiency"] = sample["focus_score"] / (sample["daily_screen_time"] + 1e-9)
sample["sleep_deficit"]    = 8 - sample["sleep_hours"]

sample_df = pd.DataFrame([sample])[ALL_FEATURES]
sample_sc = scaler.transform(sample_df)
pred = best_model.predict(sample_sc)[0]
print(f"  Input: {sample}")
print(f"\n  ★ Predicted Digital Wellbeing Score: {pred:.2f} / 100")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — see visualizations/ for all plots")
print("=" * 70)
