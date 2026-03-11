# 🧠 Digital Wellbeing ML Analysis

A complete end-to-end machine learning project that predicts a user's
**digital wellbeing score** from daily digital behaviour patterns.

---

## 📁 Project Structure

```
digital_wellbeing/
├── data/
│   ├── generate_dataset.py        # Synthetic dataset generator
│   ├── digital_wellbeing_dataset.csv
│   └── cleaned_dataset.csv        # Post-cleaning output
│
├── src/
│   ├── pipeline.py                # Full ML pipeline (run this first)
│   └── predict.py                 # CLI prediction interface
│
├── models/
│   ├── best_model.pkl             # Saved best ML model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── pca.pkl                    # Fitted PCA
│   ├── kmeans.pkl                 # Fitted KMeans
│   └── feature_names.pkl          # Ordered feature list
│
├── app/
│   └── streamlit_app.py           # Interactive web app
│
├── visualizations/
│   ├── 01_correlation_heatmap.png
│   ├── 02_distributions.png
│   ├── 03_scatter_plots.png
│   ├── 04_model_comparison.png
│   ├── 05_actual_vs_predicted.png
│   ├── 06_feature_importance.png
│   ├── 07_kmeans_elbow.png
│   ├── 08_pca_scatter.png
│   └── 09_cluster_profiles.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Run full ML pipeline
```bash
python src/pipeline.py
```

### 4. Make a prediction (CLI)
```bash
python src/predict.py \
  --screen_time 8 --app_switches 60 --sleep 6 \
  --notifications 80 --social_media 150 \
  --focus 55 --mood 60 --anxiety 6
```

### 5. Launch the Streamlit web app
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Dataset Features

| Feature | Description | Unit |
|---------|-------------|------|
| `daily_screen_time` | Total device screen time | hours |
| `num_app_switches` | Number of app switches | count |
| `sleep_hours` | Nightly sleep duration | hours |
| `notification_count` | Daily notifications received | count |
| `social_media_time_min` | Time on social platforms | minutes |
| `focus_score` | Self-rated focus ability | 0–100 |
| `mood_score` | Self-rated mood | 0–100 |
| `anxiety_level` | Self-rated anxiety | 1–10 |
| `digital_wellbeing_score` | **Target variable** | 0–100 |

---

## 🔧 Engineered Features

| Feature | Formula | Why it helps |
|---------|---------|-------------|
| `dopamine_index` | social_media + notifications + app_switches | Aggregates all stimulus-seeking signals into one score |
| `focus_efficiency` | focus_score / screen_time | Distinguishes productive vs. passive screen time |
| `sleep_deficit` | 8 − sleep_hours | Captures the accumulative sleep debt effect |

---

## 🤖 Model Results

| Model | MSE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | ~10.1 | ~3.2 | **~0.73** |
| Decision Tree | ~16.9 | ~4.1 | ~0.55 |
| Random Forest | ~11.3 | ~3.4 | ~0.70 |
| SVR | ~12.1 | ~3.5 | ~0.68 |

**Best Model: Linear Regression** (R² ≈ 0.73)

---

## 🔍 Cluster Archetypes (K-Means, k=4)

| Cluster | Label | Key Traits |
|---------|-------|-----------|
| 0 | Balanced Users | Average across all metrics |
| 1 | High-Screen Low-Sleep | Long screen sessions, poor sleep |
| 2 | Healthy & Focused | Low screen time, high focus & mood |
| 3 | Anxious Overloaded | Very high screen + social media use |

---

## 📈 Key Findings

- **Sleep hours** is the strongest single positive predictor of wellbeing.
- **Daily screen time** has the highest negative impact.
- **Mood** and **focus** scores contribute substantially to wellbeing.
- **PCA** retains ~77.5% of variance in 2 components.
- Users with a **high dopamine index** (>300) consistently show lower wellbeing.

---

## 📌 Notes

- Dataset is synthetic, designed to model realistic correlations.
- For production use, replace with real anonymised user data.
- Model can be retrained by re-running `src/pipeline.py`.
