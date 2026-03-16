import streamlit as st, re

def render():
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        st.markdown("""
<div style='text-align:center;padding:40px 0 32px;'>
  <div style='font-size:2.5rem;'>✨</div>
  <h2 style='font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:8px 0 6px;'>Create account</h2>
  <p style='color:#64748B;font-size:0.9rem;'>Start your ScreenSense wellbeing journey</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(6,182,212,0.25);
  border-radius:20px;padding:32px;'>""", unsafe_allow_html=True)

        name    = st.text_input("👤 Full Name", placeholder="Your name")
        email   = st.text_input("📧 Email",     placeholder="you@example.com")
        password= st.text_input("🔒 Password",  type="password", placeholder="Min 6 characters")
        confirm = st.text_input("🔒 Confirm Password", type="password", placeholder="Repeat password")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True):
            if not all([name, email, password, confirm]):
                st.error("Please fill in all fields.")
            elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("Please enter a valid email address.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif email in st.session_state.users:
                st.error("An account with this email already exists.")
            else:
                st.session_state.users[email] = {"password": password, "name": name}
                st.session_state.logged_in = True
                st.session_state.username  = name
                st.session_state.email     = email
                st.session_state.page      = "Dashboard"
                st.success(f"🎉 Welcome, {name}!")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Already have an account? Sign in →", use_container_width=True):
            st.session_state.page = "Login"
            st.rerun()
