import streamlit as st

def render():
    st.markdown("""
<div style='padding:8px 0 24px;'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
    <span style='background:linear-gradient(135deg,#7C3AED,#06B6D4);border-radius:50%;width:32px;height:32px;
      display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;color:white;'>1</span>
    <h1 style='font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>Digital Activity</h1>
  </div>
  <p style='color:#64748B;font-size:0.9rem;margin-left:44px;'>Step 1 of 3 — Tell us about your daily screen usage</p>
</div>
""", unsafe_allow_html=True)

    # Progress
    st.progress(0.33)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.25);
  border-radius:20px;padding:32px;'>""", unsafe_allow_html=True)

        st.markdown("#### 📱 Daily Screen Time")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>Average hours you spend on screens per day (phone, tablet, computer combined)</p>", unsafe_allow_html=True)
        screen_time = st.slider("Daily Screen Time (hours)", min_value=1.0, max_value=16.0,
                                value=st.session_state.pred_data.get("daily_screen_time", 6.0),
                                step=0.5)

        # Visual feedback
        color = "#10B981" if screen_time <= 4 else "#F59E0B" if screen_time <= 8 else "#EF4444"
        label = "Healthy" if screen_time <= 4 else "Moderate" if screen_time <= 8 else "High"
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;
  border-left:3px solid {color};margin:8px 0 24px;'>
  <span style='color:{color};font-weight:600;'>{label} Usage</span>
  <span style='color:#64748B;font-size:0.85rem;'> · {screen_time}h/day</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 🔄 App Switches Per Day")
        st.markdown("<p style='color:#64748B;font-size:0.85rem;'>How many times you switch between apps or browser tabs daily</p>", unsafe_allow_html=True)
        app_switches = st.slider("Number of App Switches", min_value=5, max_value=200,
                                 value=st.session_state.pred_data.get("num_app_switches", 60),
                                 step=5)

        sw_color = "#10B981" if app_switches < 40 else "#F59E0B" if app_switches < 100 else "#EF4444"
        sw_label = "Low" if app_switches < 40 else "Moderate" if app_switches < 100 else "High"
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:12px 16px;
  border-left:3px solid {sw_color};margin:8px 0 16px;'>
  <span style='color:{sw_color};font-weight:600;'>{sw_label} Switching</span>
  <span style='color:#64748B;font-size:0.85rem;'> · {app_switches} switches/day</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue to Step 2 →", use_container_width=True):
            st.session_state.pred_data["daily_screen_time"] = screen_time
            st.session_state.pred_data["num_app_switches"] = app_switches
            st.session_state.pred_step = 2
            st.rerun()
