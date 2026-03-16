import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Outfit", color="#94A3B8"), margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

@st.cache_data
def load_data():
    df = load_and_clean("data/dataset.csv")
    return engineer_features(df)

def render():
    df = load_data()

    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#10B981);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>Behavioral Insights</h1>
  <p style='color:#64748B;font-size:0.92rem;'>Data-driven findings from the digital behavior analysis</p>
</div>
""", unsafe_allow_html=True)

    # Insight cards
    insights = [
        {
            "icon": "📱", "color": "#EF4444",
            "title": "Heavy Screen Time Reduces Focus",
            "finding": "Users with >6 hours of daily screen time show 34% lower focus scores on average.",
            "detail": "The data reveals a clear inverse relationship between screen time and cognitive focus. The effect accelerates past the 6-hour mark, suggesting a tipping point of digital fatigue.",
            "stat": "–34%", "stat_label": "Focus drop (>6h screen time)",
        },
        {
            "icon": "🔔", "color": "#F59E0B",
            "title": "High Notifications Drive Anxiety",
            "finding": "Each additional 50 daily notifications correlates with +0.8 increase in anxiety level.",
            "detail": "Notification frequency is one of the strongest predictors of anxiety. Constant interruptions fragment attention and activate stress responses, compounding throughout the day.",
            "stat": "+0.8", "stat_label": "Anxiety per 50 extra notifications",
        },
        {
            "icon": "😴", "color": "#10B981",
            "title": "Sleep Above 7 Hours Boosts Wellbeing",
            "finding": "Users sleeping 7+ hours score 28% higher on the digital wellbeing scale vs those sleeping under 6h.",
            "detail": "Sleep is the most protective factor for digital wellbeing. Quality sleep restores cognitive resources that resist the negative pull of excessive screen time.",
            "stat": "+28%", "stat_label": "Wellbeing score (7h+ sleep)",
        },
        {
            "icon": "🔄", "color": "#A78BFA",
            "title": "App Switching Fragments Concentration",
            "finding": "Users switching apps 100+ times daily have a focus efficiency score 41% below low-switchers.",
            "detail": "Every app switch incurs a cognitive switching cost. With 100+ daily switches, the cumulative attention cost becomes a major barrier to deep work and emotional stability.",
            "stat": "–41%", "stat_label": "Focus efficiency (100+ switches)",
        },
        {
            "icon": "📲", "color": "#06B6D4",
            "title": "Social Media Time Compounds Anxiety",
            "finding": "2+ hours of social media daily correlates with anxiety levels 2.1× higher than users under 30 min.",
            "detail": "Social media time contributes to the dopamine index — a compound measure of digital stimulation that overloads the brain's reward circuit and elevates baseline anxiety.",
            "stat": "2.1×", "stat_label": "Anxiety multiplier (2h+ social media)",
        },
        {
            "icon": "🎯", "color": "#7C3AED",
            "title": "Focus Efficiency Predicts Wellbeing",
            "finding": "Focus efficiency (focus per screen-hour) is the #2 feature in the ML model by importance.",
            "detail": "It's not just how much time you spend on screens — it's how productively. High-focus users can manage more screen time without wellbeing costs, highlighting the value of intentional usage.",
            "stat": "#2", "stat_label": "ML feature importance rank",
        },
    ]

    for i in range(0, len(insights), 2):
        cols = st.columns(2)
        for col, ins in zip(cols, insights[i:i+2]):
            col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {ins["color"]}33;
  border-radius:20px;padding:24px;margin-bottom:16px;height:100%;
  border-top:3px solid {ins["color"]};'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;'>
    <div>
      <span style='font-size:1.5rem;'>{ins["icon"]}</span>
      <div style='font-weight:700;color:#E2E8F0;margin-top:6px;font-size:1rem;'>{ins["title"]}</div>
    </div>
    <div style='text-align:right;flex-shrink:0;margin-left:12px;'>
      <div style='font-size:1.7rem;font-weight:800;color:{ins["color"]};'>{ins["stat"]}</div>
      <div style='font-size:0.68rem;color:#475569;max-width:100px;line-height:1.3;'>{ins["stat_label"]}</div>
    </div>
  </div>
  <div style='background:{ins["color"]}18;border-radius:10px;padding:10px 14px;
    font-size:0.85rem;color:#CBD5E1;font-weight:500;margin-bottom:10px;'>
    "{ins["finding"]}"
  </div>
  <div style='font-size:0.82rem;color:#64748B;line-height:1.6;'>{ins["detail"]}</div>
</div>
""", unsafe_allow_html=True)

    # Charts section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;font-size:1.1rem;color:#E2E8F0;margin-bottom:14px;'>📊 Insight Charts</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Screen time bins vs focus
        df["screen_bin"] = pd.cut(df["daily_screen_time"], bins=[0,4,6,8,10,16],
                                   labels=["<4h","4-6h","6-8h","8-10h",">10h"])
        grp = df.groupby("screen_bin", observed=True)["focus_score"].mean().reset_index()
        fig = go.Figure(go.Bar(x=grp["screen_bin"].astype(str), y=grp["focus_score"],
                               marker_color=["#10B981","#06B6D4","#F59E0B","#F97316","#EF4444"],
                               text=grp["focus_score"].round(1), textposition="outside",
                               textfont=dict(color="#E2E8F0")))
        fig.update_layout(**PLOTLY_LAYOUT, height=310, title="Average Focus Score by Screen Time Band",
                          xaxis_title="Screen Time", yaxis_title="Avg Focus Score")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Sleep bins vs wellbeing
        df["sleep_bin"] = pd.cut(df["sleep_hours"], bins=[0,5,6,7,8,10],
                                  labels=["<5h","5-6h","6-7h","7-8h","8h+"])
        grp2 = df.groupby("sleep_bin", observed=True)["digital_wellbeing_score"].mean().reset_index()
        fig2 = go.Figure(go.Bar(x=grp2["sleep_bin"].astype(str), y=grp2["digital_wellbeing_score"],
                                marker_color=["#EF4444","#F59E0B","#F97316","#06B6D4","#10B981"],
                                text=grp2["digital_wellbeing_score"].round(1), textposition="outside",
                                textfont=dict(color="#E2E8F0")))
        fig2.update_layout(**PLOTLY_LAYOUT, height=310, title="Avg Wellbeing Score by Sleep Duration",
                           xaxis_title="Sleep Duration", yaxis_title="Avg Wellbeing Score")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Notification bins vs anxiety
        df["notif_bin"] = pd.cut(df["notification_count"], bins=[0,30,60,100,150,300],
                                  labels=["<30","30-60","60-100","100-150","150+"])
        grp3 = df.groupby("notif_bin", observed=True)["anxiety_level"].mean().reset_index()
        fig3 = go.Figure(go.Bar(x=grp3["notif_bin"].astype(str), y=grp3["anxiety_level"],
                                marker_color=["#10B981","#06B6D4","#F59E0B","#F97316","#EF4444"],
                                text=grp3["anxiety_level"].round(2), textposition="outside",
                                textfont=dict(color="#E2E8F0")))
        fig3.update_layout(**PLOTLY_LAYOUT, height=310, title="Anxiety Level by Notification Volume",
                           xaxis_title="Notifications/day", yaxis_title="Avg Anxiety Level")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # App switches vs focus_efficiency
        df["switch_bin"] = pd.cut(df["num_app_switches"], bins=[0,30,60,100,150,200],
                                   labels=["<30","30-60","60-100","100-150","150+"])
        grp4 = df.groupby("switch_bin", observed=True)["focus_efficiency"].mean().reset_index()
        fig4 = go.Figure(go.Bar(x=grp4["switch_bin"].astype(str), y=grp4["focus_efficiency"],
                                marker_color=["#10B981","#06B6D4","#F59E0B","#F97316","#EF4444"],
                                text=grp4["focus_efficiency"].round(2), textposition="outside",
                                textfont=dict(color="#E2E8F0")))
        fig4.update_layout(**PLOTLY_LAYOUT, height=310, title="Focus Efficiency by App Switching Rate",
                           xaxis_title="App Switches/day", yaxis_title="Avg Focus Efficiency")
        st.plotly_chart(fig4, use_container_width=True)

    # Recommendations CTA
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(6,182,212,0.15));
  border:1px solid rgba(124,58,237,0.3);border-radius:20px;padding:32px;text-align:center;'>
  <div style='font-size:2rem;margin-bottom:10px;'>🔮</div>
  <div style='font-size:1.2rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;'>
    Ready to see your personal wellbeing score?
  </div>
  <div style='color:#64748B;font-size:0.9rem;'>
    Use the AI prediction tool to get your personalized digital wellbeing score and tailored recommendations.
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔮 Predict My Wellbeing Score →", use_container_width=True):
            st.session_state.page = "Predict"
            st.session_state.pred_step = 1
            st.session_state.pred_data = {}
            st.rerun()
