import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Outfit", color="#94A3B8"), margin=dict(l=20,r=20,t=40,b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

TIER_COLOR = {"Excellent":"#10B981","Good":"#06B6D4","Fair":"#F59E0B","Needs Attention":"#EF4444"}
TIER_ICON  = {"Excellent":"🌟","Good":"✅","Fair":"⚠️","Needs Attention":"🚨"}

def render():
    email   = st.session_state.get("email","guest")
    history = st.session_state.get("prediction_history", {}).get(email, [])

    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>My Prediction History</h1>
  <p style='color:#64748B;font-size:0.92rem;'>Track your progress and celebrate every improvement</p>
</div>
""", unsafe_allow_html=True)

    if not history:
        st.markdown("""
<div style='background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);
  border-radius:20px;padding:48px;text-align:center;'>
  <div style='font-size:3rem;margin-bottom:12px;'>📭</div>
  <div style='font-size:1.1rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;'>No predictions yet</div>
  <div style='font-size:0.88rem;color:#64748B;'>
    Complete a wellbeing prediction and your results will appear here automatically.
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            if st.button("🔮 Make My First Prediction →", use_container_width=True):
                st.session_state.page      = "Predict"
                st.session_state.pred_step = 1
                st.session_state.pred_data = {}
                st.rerun()
        return

    # ── Summary cards ──────────────────────────────────────────────────────────
    scores = [r["score"] for r in history]
    best   = max(scores)
    latest = scores[-1]
    delta  = latest - scores[0] if len(scores) > 1 else 0
    trend  = "📈 Improving" if delta > 0 else ("📉 Declining" if delta < 0 else "➡️ Stable")
    trend_c= "#10B981" if delta > 0 else ("#EF4444" if delta < 0 else "#F59E0B")

    c1,c2,c3,c4 = st.columns(4)
    for col, icon, label, val, clr in [
        (c1,"🔮","Total Predictions", str(len(history)),       "#A78BFA"),
        (c2,"🏆","Best Score",        f"{best:.1f}/100",        "#10B981"),
        (c3,"⚡","Latest Score",      f"{latest:.1f}/100",      TIER_COLOR.get(history[-1]["tier"],"#06B6D4")),
        (c4,"📊","Overall Trend",     trend,                    trend_c),
    ]:
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {clr}33;border-radius:16px;
  padding:20px;text-align:center;'>
  <div style='font-size:1.4rem;'>{icon}</div>
  <div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;
    margin:6px 0 4px;font-weight:600;'>{label}</div>
  <div style='font-size:1.5rem;font-weight:800;color:{clr};'>{val}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Score timeline chart ───────────────────────────────────────────────────
    if len(history) > 1:
        st.markdown("<div style='font-weight:700;color:#E2E8F0;margin-bottom:10px;'>📈 Wellbeing Score Over Time</div>", unsafe_allow_html=True)
        df_h = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_h["timestamp"], y=df_h["score"],
            mode="lines+markers+text",
            line=dict(color="#7C3AED", width=2.5),
            marker=dict(size=9, color=df_h["score"],
                        colorscale="RdYlGn", cmin=10, cmax=100,
                        line=dict(color="white", width=1.5)),
            text=[f"{s:.1f}" for s in df_h["score"]],
            textposition="top center",
            textfont=dict(color="#E2E8F0", size=11),
            name="Wellbeing Score",
            fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"
        ))
        # add goal line at 70
        fig.add_hline(y=70, line_dash="dot", line_color="#10B981",
                      annotation_text="Goal: 70", annotation_font_color="#10B981")
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          yaxis=dict(range=[0,105], gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                          xaxis_title="Prediction Date", yaxis_title="Wellbeing Score")
        st.plotly_chart(fig, use_container_width=True)

    # ── Multi-metric trend ─────────────────────────────────────────────────────
    if len(history) > 1:
        st.markdown("<div style='font-weight:700;color:#E2E8F0;margin:16px 0 10px;'>🧩 Behavioral Trends</div>", unsafe_allow_html=True)
        df_h = pd.DataFrame(history)
        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            for metric, color, label in [
                ("sleep_hours","#06B6D4","Sleep (h)"),
                ("focus_score","#10B981","Focus Score"),
            ]:
                if metric in df_h.columns:
                    fig2.add_trace(go.Scatter(
                        x=df_h["timestamp"], y=df_h[metric],
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)))
            fig2.update_layout(**PLOTLY_LAYOUT, height=280, title="Sleep & Focus Trend",
                               legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure()
            for metric, color, label in [
                ("daily_screen_time","#EF4444","Screen Time (h)"),
                ("notification_count","#F59E0B","Notifications"),
            ]:
                if metric in df_h.columns:
                    y_vals = df_h[metric]
                    # normalise notification_count to same scale for dual display
                    if metric == "notification_count":
                        y_vals = y_vals / 10
                        label += " (÷10)"
                    fig3.add_trace(go.Scatter(
                        x=df_h["timestamp"], y=y_vals,
                        mode="lines+markers", name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)))
            fig3.update_layout(**PLOTLY_LAYOUT, height=280, title="Screen Time & Notifications Trend",
                               legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Individual records ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#E2E8F0;margin-bottom:12px;'>🗂️ All Records</div>", unsafe_allow_html=True)

    for i, record in enumerate(reversed(history)):
        idx    = len(history) - i
        tier   = record.get("tier","—")
        color  = TIER_COLOR.get(tier,"#64748B")
        icon   = TIER_ICON.get(tier,"🔵")
        delta_txt = ""
        if i < len(history) - 1:
            prev_score = list(reversed(history))[i+1]["score"]
            diff = record["score"] - prev_score
            delta_txt = f"<span style='color:{'#10B981' if diff>=0 else '#EF4444'};font-size:0.8rem;font-weight:700;'>{'▲' if diff>=0 else '▼'} {abs(diff):.1f} vs prev</span>"

        with st.expander(f"#{idx}  ·  {record['timestamp']}  ·  Score: {record['score']}  ·  {icon} {tier}", expanded=(i==0)):
            rc1, rc2, rc3 = st.columns(3)
            rc1.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;text-align:center;'>
  <div style='font-size:2rem;font-weight:800;color:{color};'>{record['score']}</div>
  <div style='font-size:0.75rem;color:#64748B;'>Wellbeing Score</div>
  <div style='margin-top:6px;'>{delta_txt}</div>
</div>
""", unsafe_allow_html=True)
            rc2.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;font-size:0.82rem;color:#94A3B8;line-height:1.7;'>
  📱 Screen Time: <b style='color:#E2E8F0;'>{record.get('daily_screen_time','—')}h</b><br>
  😴 Sleep: <b style='color:#E2E8F0;'>{record.get('sleep_hours','—')}h</b><br>
  🔔 Notifications: <b style='color:#E2E8F0;'>{record.get('notification_count','—')}</b><br>
  🔄 App Switches: <b style='color:#E2E8F0;'>{record.get('num_app_switches','—')}</b>
</div>
""", unsafe_allow_html=True)
            rc3.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;font-size:0.82rem;color:#94A3B8;line-height:1.7;'>
  📲 Social Media: <b style='color:#E2E8F0;'>{record.get('social_media_time_min','—')}min</b><br>
  🎯 Focus Score: <b style='color:#E2E8F0;'>{record.get('focus_score','—')}</b><br>
  😊 Mood Score: <b style='color:#E2E8F0;'>{record.get('mood_score','—')}</b><br>
  📅 Recorded: <b style='color:#E2E8F0;'>{record.get('timestamp','—')}</b>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Motivational banner based on trend
    if delta > 5:
        msg = f"🎉 Amazing progress! Your score improved by {delta:.1f} points. Keep building those healthy habits!"
        bg  = "rgba(16,185,129,0.12)"; bc = "rgba(16,185,129,0.3)"
    elif delta > 0:
        msg = f"✅ You're on the right track — up {delta:.1f} points. Small changes add up to big results!"
        bg  = "rgba(6,182,212,0.12)";  bc = "rgba(6,182,212,0.3)"
    elif delta == 0:
        msg = "➡️ Your score is consistent. Ready to push it higher? Try improving your sleep by 30 minutes tonight."
        bg  = "rgba(245,158,11,0.12)"; bc = "rgba(245,158,11,0.3)"
    else:
        msg = "💪 Don't give up — every journey has ups and downs. One better day is all it takes to turn the tide."
        bg  = "rgba(124,58,237,0.12)"; bc = "rgba(124,58,237,0.3)"

    st.markdown(f"""
<div style='background:{bg};border:1px solid {bc};border-radius:16px;
  padding:20px 24px;text-align:center;font-size:0.95rem;color:#E2E8F0;font-weight:500;'>
  {msg}
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔮 New Prediction →", use_container_width=True):
            st.session_state.page      = "Predict"
            st.session_state.pred_step = 1
            st.session_state.pred_data = {}
            st.rerun()
