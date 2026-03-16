import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features, FEATURE_COLS, TARGET
from src.clustering_analysis import run_clustering, run_pca, CLUSTER_LABELS
from src.model_evaluation import evaluate_model, get_feature_importance

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Outfit", color="#94A3B8"),
    margin=dict(l=20, r=20, t=44, b=20),
)
GRID = dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False)

@st.cache_data
def load_all():
    df = load_and_clean("data/dataset.csv")
    df = engineer_features(df)
    df, km, sc = run_clustering(df)
    df, pca = run_pca(df, sc)
    df["cluster_name"] = df["cluster"].map(CLUSTER_LABELS)
    return df, pca

@st.cache_resource
def load_model_and_eval():
    model = joblib.load("models/wellbeing_model.pkl")
    X_train, X_test, y_train, y_test = joblib.load("models/data_splits.pkl")
    metrics = evaluate_model(model, X_test, y_test)
    fi = get_feature_importance(model, FEATURE_COLS)
    return model, metrics, fi

def render():
    df, pca = load_all()
    model, metrics, fi = load_model_and_eval()

    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>Advanced Analytics</h1>
  <p style='color:#64748B;font-size:0.92rem;'>Model performance, feature analysis, and unsupervised learning results</p>
</div>
""", unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Model Performance","🔬 Feature Importance",
                    "🧩 Cluster Analysis","🔭 PCA Projection","🔥 Correlations"])

    # ── Tab 1: Model Performance ──────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, key, icon, color in [
            (c1,"MSE","📉","#EF4444"),(c2,"RMSE","📊","#F59E0B"),(c3,"R2","🎯","#10B981")]:
            col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {color}44;border-radius:16px;
  padding:24px;text-align:center;'>
  <div style='font-size:1.4rem;'>{icon}</div>
  <div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;
    margin:6px 0 4px;font-weight:600;'>{key}</div>
  <div style='font-size:2rem;font-weight:800;color:{color};'>{metrics[key]}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        mdf = pd.DataFrame({
            "Model":  ["Linear Regression","Decision Tree","Random Forest","SVR","Gradient Boosting"],
            "CV R²":  [0.7428, 0.6359, 0.7654, 0.7594, 0.7808],
            "CV Std": [0.0229, 0.0614, 0.0322, 0.0340, 0.0278],
        }).sort_values("CV R²", ascending=True)

        bar_colors = ["#475569","#64748B","#7C3AED","#06B6D4","#10B981"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mdf["CV R²"], y=mdf["Model"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            error_x=dict(array=mdf["CV Std"].tolist(), color="#94A3B8", thickness=2),
            text=[f"{v:.4f}" for v in mdf["CV R²"]],
            textposition="outside", textfont=dict(color="#E2E8F0", size=12)
        ))
        fig.update_layout(**PL, height=320,
                          title="Model Comparison (Cross-Validated R²)",
                          xaxis=dict(range=[0.5,0.85], **GRID),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
  border-radius:14px;padding:16px 20px;'>
  <span style='color:#10B981;font-weight:700;'>✅ Best Model: Gradient Boosting Regressor</span>
  <span style='color:#64748B;font-size:0.88rem;'>
    · CV R² = 0.7808 · Ensemble of 100 weak learners · Handles non-linear feature interactions
  </span>
</div>
""", unsafe_allow_html=True)

    # ── Tab 2: Feature Importance ─────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).sort_values("Importance")
            feat_labels = {
                "daily_screen_time":"Daily Screen Time","num_app_switches":"App Switches",
                "sleep_hours":"Sleep Hours","notification_count":"Notification Count",
                "social_media_time_min":"Social Media Time","focus_score":"Focus Score",
                "mood_score":"Mood Score","anxiety_level":"Anxiety Level",
                "dopamine_index":"Dopamine Index","focus_efficiency":"Focus Efficiency",
                "sleep_deficit":"Sleep Deficit",
            }
            fi_df["Feature"] = fi_df["Feature"].map(feat_labels)
            n = len(fi_df)
            bar_c = [f"rgba(124,58,237,{0.4+0.6*(i/n)})" for i in range(n)]
            fig2 = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                marker_color=bar_c,
                text=[f"{v:.4f}" for v in fi_df["Importance"]],
                textposition="outside", textfont=dict(color="#E2E8F0", size=11)
            ))
            fig2.update_layout(**PL, height=420,
                               title="Feature Importance (Gradient Boosting)",
                               xaxis=dict(**GRID),
                               yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig2, use_container_width=True)

            top = fi_df.nlargest(3,"Importance")
            st.markdown("<div style='font-weight:700;color:#E2E8F0;margin:16px 0 10px;'>🔑 Top Influencing Features</div>", unsafe_allow_html=True)
            explanations = {
                "Sleep Deficit":      "Sleep deprivation is the strongest predictor of poor wellbeing.",
                "Focus Efficiency":   "The ratio of focus quality to screen time captures problematic usage.",
                "Dopamine Index":     "Combined measure of social media, notifications, and app switching.",
                "Anxiety Level":      "Anxiety mediates the effect of digital overuse on wellbeing.",
                "Mood Score":         "Mood directly correlates with wellbeing and digital habits.",
                "Sleep Hours":        "Absolute sleep duration has direct positive effects on cognition.",
            }
            cols = st.columns(3)
            for col, (_, row) in zip(cols, top.iterrows()):
                col.markdown(f"""
<div style='background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.25);
  border-radius:14px;padding:16px;'>
  <div style='font-weight:700;color:#A78BFA;margin-bottom:6px;font-size:0.92rem;'>{row["Feature"]}</div>
  <div style='font-size:0.8rem;color:#64748B;line-height:1.5;'>{explanations.get(row["Feature"],"Key predictor.")}</div>
  <div style='margin-top:10px;font-size:0.75rem;color:#7C3AED;font-weight:600;'>Importance: {row["Importance"]:.4f}</div>
</div>
""", unsafe_allow_html=True)

    # ── Tab 3: Cluster Analysis ───────────────────────────────────────────────
    with tabs[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fig3 = px.scatter(df.sample(600, random_state=42), x="PC1", y="PC2",
                              color="cluster_name",
                              color_discrete_sequence=["#7C3AED","#06B6D4","#10B981"],
                              title="K-Means Clusters in PCA Space")
            fig3.update_layout(**PL, height=380,
                               xaxis=dict(**GRID), yaxis=dict(**GRID),
                               legend=dict(orientation="h", y=-0.15))
            fig3.update_traces(marker=dict(size=5, opacity=0.75))
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            cluster_stats = df.groupby("cluster_name")[
                ["daily_screen_time","sleep_hours","focus_score",
                 "notification_count","digital_wellbeing_score"]].mean()
            clabels  = ["Screen Time(h)","Sleep(h)","Focus/10","Notifs/10","Wellbeing"]
            colors_c = ["#7C3AED","#06B6D4","#10B981"]
            fig4 = go.Figure()
            for i, (cname, row) in enumerate(cluster_stats.iterrows()):
                vals = [row["daily_screen_time"], row["sleep_hours"],
                        row["focus_score"]/10, row["notification_count"]/10,
                        row["digital_wellbeing_score"]]
                fig4.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=clabels+[clabels[0]],
                    fill="toself", name=cname,
                    fillcolor=f"{colors_c[i]}33",
                    line=dict(color=colors_c[i], width=2)))
            fig4.update_layout(**PL, height=380, title="Cluster Behavioral Profiles",
                               polar=dict(bgcolor="rgba(255,255,255,0.03)",
                                          radialaxis=dict(visible=True,gridcolor="rgba(255,255,255,0.1)"),
                                          angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                               legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<div style='font-weight:700;color:#E2E8F0;margin:16px 0 10px;'>📋 Cluster Statistics</div>", unsafe_allow_html=True)
        display = df.groupby("cluster_name")[
            ["daily_screen_time","sleep_hours","focus_score",
             "notification_count","anxiety_level","digital_wellbeing_score"]
        ].mean().round(2)
        display.columns = ["Screen Time(h)","Sleep(h)","Focus","Notifications","Anxiety","Wellbeing"]
        st.dataframe(display.style.background_gradient(cmap="RdYlGn", axis=0),
                     use_container_width=True)

    # ── Tab 4: PCA ────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        var_exp = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_exp)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig5 = px.scatter(df.sample(800, random_state=99), x="PC1", y="PC2",
                              color="digital_wellbeing_score",
                              color_continuous_scale="Viridis",
                              title="PCA Projection (coloured by Wellbeing)")
            fig5.update_layout(**PL, height=360, xaxis=dict(**GRID), yaxis=dict(**GRID))
            fig5.update_traces(marker=dict(size=4, opacity=0.7))
            st.plotly_chart(fig5, use_container_width=True)

        with col_p2:
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(x=["PC1","PC2"], y=var_exp*100,
                                  marker_color=["#7C3AED","#06B6D4"], name="Explained Var %"))
            fig6.add_trace(go.Scatter(x=["PC1","PC2"], y=cum_var*100,
                                      mode="lines+markers", name="Cumulative",
                                      line=dict(color="#10B981",width=2),
                                      marker=dict(size=8)))
            fig6.update_layout(**PL, height=360,
                               title="PCA Explained Variance",
                               xaxis=dict(**GRID),
                               yaxis=dict(title="Variance Explained (%)", **GRID),
                               legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;padding:16px 20px;'>
  <span style='color:#A78BFA;font-weight:700;'>PCA Summary</span>
  <span style='color:#64748B;font-size:0.88rem;'>
    · PC1 explains <b style="color:#E2E8F0;">{var_exp[0]*100:.1f}%</b> of variance
    · PC2 explains <b style="color:#E2E8F0;">{var_exp[1]*100:.1f}%</b>
    · Together: <b style="color:#E2E8F0;">{cum_var[1]*100:.1f}%</b> in 2D space.
  </span>
</div>
""", unsafe_allow_html=True)

    # ── Tab 5: Correlations ───────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("<br>", unsafe_allow_html=True)
        corr_cols = ["daily_screen_time","sleep_hours","notification_count","focus_score",
                     "mood_score","anxiety_level","social_media_time_min","digital_wellbeing_score"]
        corr_labels = ["Screen Time","Sleep","Notifications","Focus","Mood",
                       "Anxiety","Social Media","Wellbeing"]
        corr = df[corr_cols].corr()
        fig7 = go.Figure(go.Heatmap(
            z=corr.values, x=corr_labels, y=corr_labels,
            colorscale=[[0,"#EF4444"],[0.5,"#1E293B"],[1,"#10B981"]],
            zmid=0, text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            colorbar=dict(title="r", tickfont=dict(color="#94A3B8"))
        ))
        fig7.update_layout(**PL, height=480, title="Feature Correlation Matrix",
                           xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("<div style='font-weight:700;color:#E2E8F0;margin:16px 0 10px;'>🔑 Key Correlations</div>", unsafe_allow_html=True)
        pairs = [
            (corr.loc["digital_wellbeing_score","sleep_hours"],        "Sleep ↔ Wellbeing",       "#10B981"),
            (corr.loc["digital_wellbeing_score","daily_screen_time"],  "Screen Time ↔ Wellbeing", "#EF4444"),
            (corr.loc["digital_wellbeing_score","anxiety_level"],      "Anxiety ↔ Wellbeing",     "#EF4444"),
            (corr.loc["focus_score","daily_screen_time"],              "Screen Time ↔ Focus",     "#F59E0B"),
        ]
        cols = st.columns(4)
        for col, (r, label, color) in zip(cols, pairs):
            col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {color}33;border-radius:12px;
  padding:14px;text-align:center;'>
  <div style='font-size:1.5rem;font-weight:800;color:{color};'>{r:.2f}</div>
  <div style='font-size:0.75rem;color:#64748B;margin-top:4px;'>{label}</div>
  <div style='color:{color};font-size:0.8rem;margin-top:2px;font-weight:600;'>
    {"Positive ↑" if r>0 else "Negative ↓"}
  </div>
</div>
""", unsafe_allow_html=True)
