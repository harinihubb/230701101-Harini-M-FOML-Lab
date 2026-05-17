# 🧠 Dopamine × Productivity — Machine Learning Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/Frontend-HTML5%20%2F%20CSS3%20%2F%20JS-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A full-stack machine learning system that analyzes students' digital behavior patterns and predicts their productivity levels — revealing how dopamine-triggering habits influence focus and academic performance.**

[Features](#-features) · [Demo](#-live-demo) · [Installation](#-installation) · [API Docs](#-api-reference) · [Results](#-model-results) · [Contributing](#-contributing)

</div>

---

## 📌 Problem Statement

Modern students are caught in a dopamine loop — social media, endless notifications, and excessive screen time fragment attention and silently erode productivity. This project builds a data-driven system to **quantify that relationship** using machine learning.

> *Can we predict a student's productivity level just by looking at their digital habits?*  
> **Yes — with 92% accuracy.**

---

## ✨ Features

- 🎯 **Productivity Predictor** — Input your daily habits, get an instant ML-powered productivity label (High / Medium / Low) with confidence probabilities
- 📊 **Exploratory Data Analysis** — Correlation heatmaps, feature distributions, PCA scatter plots
- 🤖 **3 Trained ML Models** — Logistic Regression, Decision Tree, SVM with full evaluation metrics
- 🔵 **K-Means Clustering** — Identifies 3 behavioral archetypes: High-Focus Group, Balanced Group, Heavy Digital Group
- 📉 **PCA Visualization** — 2D dimensionality reduction explaining 62.8% variance
- 🌐 **REST API** — 8 Flask endpoints serving predictions and analytics
- 💻 **Standalone Frontend** — Single HTML file, works offline, no framework required

---

## 🗂️ Project Structure

```
dopamine-productivity-ml/
│
├── backend/
│   ├── train_models.py        # Full ML training pipeline
│   └── app.py                 # Flask REST API (port 5050)
│
├── frontend/
│   └── index.html             # Standalone SPA (no build step needed)
│
├── data/
│   └── dataset.csv            # 500-student synthetic dataset
│
├── models/
│   ├── best_model.pkl         # Logistic Regression (92% acc)
│   ├── scaler.pkl             # StandardScaler
│   ├── label_encoder.pkl      # LabelEncoder
│   ├── kmeans.pkl             # K-Means (k=3)
│   ├── pca.pkl                # PCA (2 components)
│   └── analytics.json         # Pre-computed analytics cache
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dopamine-productivity-ml.git
cd dopamine-productivity-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
python backend/train_models.py
```
This generates the dataset, trains all models, runs clustering + PCA, and saves all artifacts to `/models`.

### 4. Start the API server
```bash
python backend/app.py
# API available at http://localhost:5050
```

### 5. Open the frontend
```bash
# Option A: Open directly in browser (fully offline)
open frontend/index.html

# Option B: Serve with Python
python -m http.server 8080
# Then visit http://localhost:8080/frontend/
```

---

## 📦 Requirements

```
flask>=2.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

```bash
pip install flask scikit-learn pandas numpy joblib
```

---

## 📊 Dataset

The dataset contains **500 student records** with the following features:

| Feature | Type | Description |
|---|---|---|
| `screen_time` | float | Total daily screen time (hours) |
| `social_media_hours` | float | Time spent on social media (hours) |
| `notifications` | int | Number of notifications received per day |
| `study_hours` | float | Dedicated study/work hours per day |
| `sleep_hours` | float | Sleep duration (hours) |
| `exercise_mins` | int | Physical exercise (minutes) |
| `caffeine_cups` | int | Caffeine consumption (cups/day) |

### Engineered Features

| Feature | Formula | Meaning |
|---|---|---|
| `focus_ratio` | `study_hours / (screen_time + 0.1)` | Study-to-screen efficiency |
| `dopamine_index` | `social_media × 0.6 + (notifications / 20) × 0.4` | Digital dopamine stimulation score |
| `sleep_quality` | `(sleep_hours - 4) / 6` | Normalized sleep adequacy |

### Target Classes
- 🟢 **High** — Top 34% productivity scores
- 🟡 **Medium** — Middle 33%
- 🔴 **Low** — Bottom 33%

---

## 🤖 Model Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | Status |
|---|---|---|---|---|---|
| **Logistic Regression** | **92.0%** | **92.0%** | **92.0%** | **0.920** | ✅ Deployed |
| SVM (RBF Kernel) | 92.0% | 92.2% | 92.0% | 0.921 | Trained |
| Decision Tree (depth=6) | 78.0% | 77.7% | 78.0% | 0.778 | Trained |

### Key Findings — Feature Correlations with Productivity

| Feature | Correlation | Interpretation |
|---|---|---|
| 📱 Social Media Hours | **-0.783** | Strongest negative predictor |
| 🧲 Dopamine Index | **-0.783** | High dopamine load = low productivity |
| 🖥️ Screen Time | **-0.776** | More screen = less output |
| 🔔 Notifications | **-0.770** | Attention fragmentation |
| 🎯 Focus Ratio | **+0.704** | Strongest positive predictor |
| 📚 Study Hours | **+0.499** | Obvious but secondary |
| 😴 Sleep Hours | **+0.244** | Moderate positive effect |
| 🏃 Exercise | **+0.124** | Small but consistent boost |
| ☕ Caffeine | **-0.043** | Negligible effect |

### Behavioral Clusters (K-Means, k=3)

| Cluster | Social Media | Screen Time | Focus Ratio | Sleep | Count |
|---|---|---|---|---|---|
| 🎯 High-Focus Group | 1.6h | 3.5h | 1.33 | 7.6h | 165 |
| ⚖️ Balanced Group | 1.9h | 4.1h | 1.18 | 5.2h | 148 |
| 📱 Heavy Digital Group | 7.0h | 9.1h | 0.46 | 6.4h | 87 |

---

## 🌐 API Reference

Base URL: `http://localhost:5050`

### `GET /api/health`
Check server status and deployed model accuracy.

```json
{
  "status": "ok",
  "model": "Logistic Regression",
  "accuracy": 0.92
}
```

---

### `POST /api/predict`
Predict productivity level from digital behavior inputs.

**Request Body:**
```json
{
  "screen_time": 5,
  "social_media_hours": 3,
  "notifications": 50,
  "study_hours": 4,
  "sleep_hours": 7,
  "exercise_mins": 30,
  "caffeine_cups": 2
}
```

**Response:**
```json
{
  "prediction": "High",
  "probabilities": {
    "High": 0.8812,
    "Medium": 0.1043,
    "Low": 0.0145
  },
  "cluster": "High-Focus Group",
  "engineered_features": {
    "focus_ratio": 0.784,
    "dopamine_index": 2.3,
    "sleep_quality": 0.5
  },
  "pca_projection": { "x": -1.45, "y": 0.87 },
  "insights": [
    "Great habits! Maintain your current routine for sustained performance."
  ],
  "model_used": "Logistic Regression"
}
```

---

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/analytics` | Full analytics dump (all models, clusters, PCA) |
| GET | `/api/models` | Model comparison metrics + confusion matrices |
| GET | `/api/correlations` | Feature correlations with productivity score |
| GET | `/api/clusters` | K-Means cluster statistics |
| GET | `/api/pca` | PCA 2D projection data + explained variance |
| GET | `/api/dataset/sample` | Random 50-row sample from the dataset |

---

## 🖥️ Frontend Walkthrough

The frontend is a single HTML file with **6 interactive tabs**:

| Tab | Content |
|---|---|
| **Overview** | Hero stats, correlation bars, full pipeline visualization |
| **Predict** | Live sliders → engineered features → prediction + insights |
| **Models** | Metrics table + color-coded confusion matrices |
| **EDA** | PCA scatter, feature distribution charts, correlation heatmap |
| **Clusters** | Behavioral archetype cards + grouped bar comparison |
| **API** | Live endpoint reference with sample request/response |

The frontend **embeds the analytics.json data directly** and implements the prediction heuristic in JavaScript — making it fully functional offline without the Flask server.

---

## 🔬 Methodology

```
Data Generation → Preprocessing → Feature Engineering
        ↓
   EDA + Correlation Analysis
        ↓
   Train/Test Split (80/20, stratified)
        ↓
┌──────────────────────────────────┐
│    Supervised Learning           │
│  • Logistic Regression           │
│  • Decision Tree (depth=6)       │
│  • SVM (RBF kernel)              │
└──────────────────────────────────┘
        ↓
┌──────────────────────────────────┐
│    Unsupervised Learning         │
│  • K-Means Clustering (k=3)      │
│  • PCA (2 components)            │
└──────────────────────────────────┘
        ↓
   Evaluation (Accuracy, Precision, Recall, F1, CM)
        ↓
   Flask API + Standalone Frontend
```

---

## 💡 Key Takeaways

1. **Social media is the #1 productivity killer** — stronger correlation than even study hours
2. **Focus ratio matters more than raw study time** — how you study beats how long you study  
3. **Notification overload is underrated** — 80+ notifications/day consistently predicts Low productivity
4. **Sleep quality > sleep quantity** — 7+ hours is a threshold, not just a target
5. **Exercise has a small but reliable positive effect** — even 20 min/day moves the needle

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Connect to real student data via a survey form
- [ ] Add time-series tracking (daily logs over weeks)
- [ ] Implement SHAP explainability for feature importance
- [ ] Add more models (Random Forest, XGBoost, Neural Network)
- [ ] Build a mobile-responsive PWA version
- [ ] Add a "habit improvement simulator" to the predictor

To contribute:
```bash
git fork
git checkout -b feature/your-feature-name
git commit -m "Add: your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```



*Understanding your digital habits is the first step to reclaiming your focus.*

</div>
