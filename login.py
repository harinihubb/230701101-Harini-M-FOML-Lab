import streamlit as st

def render():
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.markdown("""
<div style='text-align:center;padding:40px 0 32px;'>
  <div style='font-size:2.5rem;'>🔑</div>
  <h2 style='font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:8px 0 6px;'>Welcome back</h2>
  <p style='color:#64748B;font-size:0.9rem;'>Sign in to your ScreenSense account</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.25);
  border-radius:20px;padding:32px;backdrop-filter:blur(12px);'>""", unsafe_allow_html=True)

        email    = st.text_input("📧 Email", placeholder="you@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True):
            users = st.session_state.users
            if email in users and users[email]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username  = users[email]["name"]
                st.session_state.email     = email
                st.session_state.page      = "Dashboard"
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid email or password.")

        st.markdown("""
<div style='text-align:center;margin-top:20px;font-size:0.85rem;color:#64748B;'>
  Demo credentials: <span style='color:#A78BFA;'>demo@screensense.ai</span> /
  <span style='color:#A78BFA;'>demo123</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Don't have an account? Sign up →", use_container_width=True):
            st.session_state.page = "Signup"
            st.rerun()
