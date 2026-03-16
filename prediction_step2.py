import streamlit as st

def render():
    st.markdown("""
<div style='padding:8px 0 24px;'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
    <span style='background:linear-gradient(135deg,#7C3AED,#06B6D4);border-radius:50%;width:32px;height:32px;
      display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;color:white;'>2</span>
    <h1 style='font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#06B6D4);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>Sleep & Notifications</h1>
  </div>
  <p style='color:#64748B;font-size:0.9rem;margin-left:44px;'>Step 2 of 3 — Rest patterns & digital interruptions</p>
</div>
""", unsafe_allow_html=True)

    st.progress(0.66)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(6,182,212,0.25);
  border-radius:20px;padding:32px;'>""", unsafe_allow_html=True)

        st.markdown("#### 😴 Sleep Hours")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Average hours of sleep you get per night</p>", unsafe_allow_html=True)
        sleep = st.slider("Sleep Hours per Night", min_value=3.0, max_value=10.0,
                          value=st.session_state.pred_data.get("sleep_hours", 7.0), step=0.25)

        sleep_color = "#EF4444" if sleep < 5 else "#F59E0B" if sleep < 7 else "#10B981"
        sleep_label = "Sleep Deprived" if sleep < 5 else "Below Optimal" if sleep < 7 else "Well Rested"
        deficit = max(0, 8 - sleep)
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;
  border-left:3px solid {sleep_color};margin:8px 0 24px;'>
  <span style='color:{sleep_color};font-weight:600;'>{sleep_label}</span>
  <span style='color:#64748B;font-size:0.85rem;'> · {sleep}h sleep · Sleep deficit: {deficit:.2f}h</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🔔 Daily Notification Count")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Total notifications received per day across all apps</p>", unsafe_allow_html=True)
        notifs = st.slider("Notifications per Day", min_value=5, max_value=300,
                           value=st.session_state.pred_data.get("notification_count", 80), step=5)

        n_color = "#10B981" if notifs < 50 else "#F59E0B" if notifs < 120 else "#EF4444"
        n_label = "Low" if notifs < 50 else "Moderate" if notifs < 120 else "High Volume"
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;
  border-left:3px solid {n_color};margin:8px 0 16px;'>
  <span style='color:{n_color};font-weight:600;'>{n_label} Interruptions</span>
  <span style='color:#64748B;font-size:0.85rem;'> · {notifs} notifications/day</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        c_back, c_next = st.columns(2)
        with c_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.pred_step = 1
                st.rerun()
        with c_next:
            if st.button("Continue to Step 3 →", use_container_width=True):
                st.session_state.pred_data["sleep_hours"] = sleep
                st.session_state.pred_data["notification_count"] = notifs
                st.session_state.pred_step = 3
                st.rerun()
