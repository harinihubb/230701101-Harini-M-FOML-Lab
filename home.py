import streamlit as st

def render():
    st.markdown("""
<div style='text-align:center;padding:60px 20px 20px;'>
  <div style='display:inline-block;background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(6,182,212,0.2));
    border:1px solid rgba(124,58,237,0.4);border-radius:20px;padding:12px 24px;
    font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;color:#A78BFA;font-weight:600;margin-bottom:20px;'>
    AI-Powered Wellbeing Intelligence
  </div>
  <h1 style='font-size:3.5rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0 0%,#A78BFA 50%,#06B6D4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 16px;line-height:1.1;'>
    Understand Your<br>Digital Wellbeing
  </h1>
  <p style='font-size:1.15rem;color:#94A3B8;max-width:580px;margin:0 auto 40px;line-height:1.7;'>
    ScreenSense uses machine learning to analyze your digital behavior patterns
    and predict your wellbeing score — giving you actionable insights to reclaim balance.
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🔮", "AI Prediction",    "Gradient Boosting ML model with 78% accuracy predicts your digital wellbeing from behavioral inputs."),
        (c2, "🔍", "Behavior Clusters", "K-Means clustering identifies your behavior archetype across 3 distinct digital lifestyle groups."),
        (c3, "💡", "Smart Insights",    "Personalized AI-generated recommendations based on your unique usage patterns and wellbeing score."),
    ]:
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.2);
  border-radius:20px;padding:28px 24px;height:100%;backdrop-filter:blur(12px);'>
  <div style='font-size:2rem;margin-bottom:12px;'>{icon}</div>
  <div style='font-weight:700;font-size:1.05rem;color:#E2E8F0;margin-bottom:8px;'>{title}</div>
  <div style='font-size:0.88rem;color:#64748B;line-height:1.6;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for col, num, label in [
        (s1, "1,200", "Records Analyzed"),
        (s2, "78.1%", "Model Accuracy (R²)"),
        (s3, "11",    "Behavioral Features"),
        (s4, "3",     "Behavior Clusters"),
    ]:
        col.markdown(f"""
<div style='text-align:center;background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
  border-radius:16px;padding:24px 16px;'>
  <div style='font-size:2rem;font-weight:800;background:linear-gradient(135deg,#A78BFA,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{num}</div>
  <div style='font-size:0.82rem;color:#64748B;margin-top:4px;font-weight:500;'>{label}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if not st.session_state.logged_in:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔑 Log In", use_container_width=True):
                    st.session_state.page = "Login"
                    st.rerun()
            with col_b:
                if st.button("✨ Get Started Free", use_container_width=True):
                    st.session_state.page = "Signup"
                    st.rerun()
        else:
            if st.button("🔮 Predict My Wellbeing Score →", use_container_width=True):
                st.session_state.page = "Predict"
                st.session_state.pred_step = 1
                st.session_state.pred_data = {}
                st.rerun()
