"""
================================================================================
  app/streamlit_app.py  —  Digital Wellbeing Predictor
================================================================================
Run with:  streamlit run app/streamlit_app.py
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ── resolve paths whether run from root or from app/ ─────────────────────────
HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
MDL   = os.path.join(ROOT, "models")
DATA  = os.path.join(ROOT, "data",  "cleaned_dataset.csv")
VIZ   = os.path.join(ROOT, "visualizations")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Wellbeing Predictor",
    page_icon="🧠",
    layout="wide",
)

# ── load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load(os.path.join(MDL, "best_model.pkl"))
    scaler   = joblib.load(os.path.join(MDL, "scaler.pkl"))
    features = joblib.load(os.path.join(MDL, "feature_names.pkl"))
    df       = pd.read_csv(DATA)
    return model, scaler, features, df

model, scaler, ALL_FEATURES, df = load_artefacts()

# ── helpers ────────────────────────────────────────────────────────────────────
def score_color(score):
    if score >= 70: return "🟢"
    if score >= 45: return "🟡"
    return "🔴"

def build_input(st_time, app_sw, sleep, notif, sm_time, focus, mood, anxiety):
    row = {
        "daily_screen_time":     st_time,
        "num_app_switches":      app_sw,
        "sleep_hours":           sleep,
        "notification_count":    notif,
        "social_media_time_min": sm_time,
        "focus_score":           focus,
        "mood_score":            mood,
        "anxiety_level":         anxiety,
    }
    row["dopamine_index"]   = sm_time + notif + app_sw
    row["focus_efficiency"] = focus / (st_time + 1e-9)
    row["sleep_deficit"]    = 8 - sleep
    return pd.DataFrame([row])[ALL_FEATURES]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 Digital Wellbeing Predictor")
st.markdown(
    """
    Analyse how your **daily digital habits** affect your **mental wellbeing**.  
    Adjust the sliders, hit **Predict**, and see where you stand.
    """
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Data Insights", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Daily Digital Habits")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📱 Screen Usage")
        screen_time  = st.slider("Daily Screen Time (hrs)", 1.0, 16.0, 7.0, 0.5)
        app_switches = st.slider("App Switches per Day",    5,   200,  60,  5)
        sm_time      = st.slider("Social Media Time (min)", 0,   600,  120, 10)

    with col2:
        st.markdown("##### 🔔 Notifications & Sleep")
        notifications = st.slider("Notifications per Day",  0,   200,  80,  5)
        sleep_hours   = st.slider("Sleep Hours",            3.0, 10.0, 7.0, 0.5)

    with col3:
        st.markdown("##### 🧘 Mental State")
        focus_score = st.slider("Focus Score (0–100)",  10, 100, 65, 1)
        mood_score  = st.slider("Mood Score (0–100)",   10, 100, 65, 1)
        anxiety_lvl = st.slider("Anxiety Level (1–10)", 1,  10,  4,  1)

    st.divider()
    predict_btn = st.button("🔍 Predict My Wellbeing Score", type="primary", use_container_width=True)

    if predict_btn:
        inp    = build_input(screen_time, app_switches, sleep_hours,
                             notifications, sm_time, focus_score, mood_score, anxiety_lvl)
        inp_sc = scaler.transform(inp)
        score  = float(model.predict(inp_sc)[0])
        score  = max(0.0, min(100.0, score))
        emoji  = score_color(score)

        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.metric(f"{emoji} Predicted Digital Wellbeing Score", f"{score:.1f} / 100")
            # Progress bar
            st.progress(int(score))

            if score >= 70:
                st.success("🎉 **Excellent!** Your digital habits are supporting great wellbeing.")
            elif score >= 50:
                st.warning("🙂 **Good.** A few tweaks could further improve your wellbeing.")
            elif score >= 35:
                st.warning("😐 **Moderate.** Consider reducing screen time and improving sleep.")
            else:
                st.error("😟 **Needs attention.** High screen use and poor sleep are impacting you.")

        # Engineered features breakdown
        st.divider()
        st.markdown("#### 📐 Computed Indicators")
        ci1, ci2, ci3 = st.columns(3)
        dopamine  = sm_time + notifications + app_switches
        feff      = focus_score / (screen_time + 1e-9)
        sdef      = 8 - sleep_hours
        ci1.metric("Dopamine Index",    f"{dopamine:.0f}",  help="Social media + notifications + app switches. Lower = better.")
        ci2.metric("Focus Efficiency",  f"{feff:.1f}",      help="Focus score per screen hour. Higher = better.")
        ci3.metric("Sleep Deficit",     f"{sdef:.1f} hrs",  help="Hours short of recommended 8 hrs. Lower = better.")

        # Radar chart of inputs vs dataset average
        st.divider()
        st.markdown("#### 🕸️ Your Profile vs. Dataset Average")
        labels_radar = ["Screen Time", "App Switches", "Sleep", "Notifications",
                        "Social Media", "Focus", "Mood", "Anxiety"]
        user_vals   = [screen_time, app_switches/10, sleep_hours, notifications/20,
                       sm_time/60, focus_score/10, mood_score/10, anxiety_lvl]
        avg_vals    = [df["daily_screen_time"].mean(),
                       df["num_app_switches"].mean()/10,
                       df["sleep_hours"].mean(),
                       df["notification_count"].mean()/20,
                       df["social_media_time_min"].mean()/60,
                       df["focus_score"].mean()/10,
                       df["mood_score"].mean()/10,
                       df["anxiety_level"].mean()]

        N_r = len(labels_radar)
        angles = np.linspace(0, 2 * np.pi, N_r, endpoint=False).tolist()
        user_vals += user_vals[:1]; avg_vals += avg_vals[:1]; angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, user_vals, "o-", lw=2, color="#4C72B0", label="You")
        ax.fill(angles, user_vals, alpha=0.25, color="#4C72B0")
        ax.plot(angles, avg_vals,  "o-", lw=2, color="#DD8452", label="Average")
        ax.fill(angles, avg_vals,  alpha=0.15, color="#DD8452")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_radar, fontsize=9)
        ax.set_title("Profile Radar", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Insights")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Correlation with Wellbeing Score**")
        corr = df[["daily_screen_time", "sleep_hours", "focus_score",
                   "mood_score", "anxiety_level", "digital_wellbeing_score"]].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Wellbeing Score Distribution**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(df["digital_wellbeing_score"], kde=True, ax=ax, color="#4C72B0")
        ax.axvline(df["digital_wellbeing_score"].mean(), color="red",
                   linestyle="--", label=f"Mean: {df['digital_wellbeing_score'].mean():.1f}")
        ax.legend(); ax.set_title("Wellbeing Score Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("**Sleep vs. Wellbeing (coloured by Anxiety)**")
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(df["sleep_hours"], df["digital_wellbeing_score"],
                    c=df["anxiety_level"], cmap="RdYlGn_r", alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label="Anxiety Level")
    ax.set_xlabel("Sleep Hours"); ax.set_ylabel("Wellbeing Score")
    ax.set_title("Sleep vs Wellbeing (colour = anxiety)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### Digital Wellbeing ML Analysis

    This project uses machine learning to predict a user's **digital wellbeing score**
    based on behavioural metrics collected from their device usage and self-reported 
    mental state indicators.

    #### 🧩 Engineered Features
    | Feature | Formula | Interpretation |
    |---------|---------|----------------|
    | **Dopamine Index** | social_media_time + notifications + app_switches | Proxy for stimulus-seeking behaviour |
    | **Focus Efficiency** | focus_score / screen_time | Quality vs. quantity of screen time |
    | **Sleep Deficit** | 8 − sleep_hours | Deviation from recommended sleep |

    #### 🤖 Models Trained
    - Linear Regression  
    - Decision Tree Regressor  
    - Random Forest Regressor  
    - Support Vector Regression  

    #### 📈 Key Findings
    - **Sleep hours** and **mood score** are the strongest positive predictors.
    - **Daily screen time** and **anxiety level** are the strongest negative predictors.
    - **PCA** explains ~77.5% of variance in just 2 components.
    - **K-Means (k=4)** reveals four behavioural archetypes:
        - 🟢 *Healthy & Focused* — low screen time, high focus
        - 🟡 *Balanced Users* — average across all metrics
        - 🟠 *High-Screen Low-Sleep* — elevated usage, insufficient rest
        - 🔴 *Anxious Overloaded* — very high screen time & notifications

    ---
    *Built as a final-year ML project. Model trained on synthetic data.*
    """)
