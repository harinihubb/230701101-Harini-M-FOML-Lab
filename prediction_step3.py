import streamlit as st

def render():
    st.markdown("""
<div style='padding:8px 0 24px;'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
    <span style='background:linear-gradient(135deg,#7C3AED,#06B6D4);border-radius:50%;width:32px;height:32px;
      display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;color:white;'>3</span>
    <h1 style='font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#10B981);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>Behavior & Mood</h1>
  </div>
  <p style='color:#64748B;font-size:0.9rem;margin-left:44px;'>Step 3 of 3 — Social habits and emotional state</p>
</div>
""", unsafe_allow_html=True)

    st.progress(1.0)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(16,185,129,0.25);
  border-radius:20px;padding:32px;'>""", unsafe_allow_html=True)

        st.markdown("#### 📲 Social Media Time")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Minutes spent on social media platforms daily</p>", unsafe_allow_html=True)
        social = st.slider("Social Media (minutes/day)", min_value=0, max_value=360,
                           value=st.session_state.pred_data.get("social_media_time_min", 90), step=10)
        s_color = "#10B981" if social < 60 else "#F59E0B" if social < 150 else "#EF4444"
        s_label = "Minimal" if social < 60 else "Moderate" if social < 150 else "Heavy"
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:10px 16px;
  border-left:3px solid {s_color};margin:6px 0 20px;font-size:0.85rem;'>
  <span style='color:{s_color};font-weight:600;'>{s_label}</span>
  <span style='color:#64748B;'> · {social} min/day ({social/60:.1f}h)</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🎯 Focus Score")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Rate your ability to concentrate during work/study (0–100)</p>", unsafe_allow_html=True)
        focus = st.slider("Focus Score", min_value=10, max_value=100,
                          value=st.session_state.pred_data.get("focus_score", 60))

        st.markdown("#### 😊 Mood Score")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Rate your overall mood over the past week (20–100)</p>", unsafe_allow_html=True)
        mood = st.slider("Mood Score", min_value=20, max_value=100,
                         value=st.session_state.pred_data.get("mood_score", 65))

        # Anxiety derived display
        notifs = st.session_state.pred_data.get("notification_count", 80)
        screen = st.session_state.pred_data.get("daily_screen_time", 6)
        est_anxiety = min(10, (notifs * 0.08 + social * 0.03 + screen * 1.2) / 10)
        a_color = "#10B981" if est_anxiety < 4 else "#F59E0B" if est_anxiety < 7 else "#EF4444"
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:14px 16px;margin-top:16px;'>
  <div style='font-size:0.78rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>
    Estimated Anxiety Indicator
  </div>
  <div style='display:flex;align-items:center;gap:10px;'>
    <div style='flex:1;background:rgba(255,255,255,0.08);border-radius:100px;height:8px;overflow:hidden;'>
      <div style='width:{est_anxiety*10}%;height:100%;background:{a_color};border-radius:100px;'></div>
    </div>
    <span style='color:{a_color};font-weight:700;font-size:0.9rem;'>{est_anxiety:.1f}/10</span>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c_back, c_pred = st.columns(2)
        with c_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.pred_step = 2
                st.rerun()
        with c_pred:
            if st.button("🔮 Predict Wellbeing Score", use_container_width=True):
                st.session_state.pred_data["social_media_time_min"] = social
                st.session_state.pred_data["focus_score"] = focus
                st.session_state.pred_data["mood_score"] = mood
                st.session_state.pred_step = 4
                st.rerun()
