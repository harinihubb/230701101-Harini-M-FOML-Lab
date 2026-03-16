import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="ScreenSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background: #080B14 !important;
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1E 0%, #080B14 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.2) !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
h1, h2, h3 { font-family: 'Outfit', sans-serif !important; font-weight: 700 !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 12px 28px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    font-size: 15px !important; cursor: pointer !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}
.stTextInput input, .stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
    font-family: 'Outfit', sans-serif !important;
}
.stProgress .st-bo { background: linear-gradient(90deg,#7C3AED,#06B6D4) !important; }
hr { border-color: rgba(124,58,237,0.2) !important; }
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important; padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-family: 'Outfit', sans-serif !important;
    color: #94A3B8 !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7C3AED,#06B6D4) !important;
    color: white !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080B14; }
::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

inject_css()

# ── Session state init ──────────────────────────────────────────────────────────
defaults = {
    "logged_in": False,
    "username": "",
    "email": "",
    "page": "Home",
    "pred_step": 1,
    "pred_data": {},
    "prediction_history": {},   # email -> list of prediction dicts
    "users": {"demo@screensense.ai": {"password": "demo123", "name": "Demo User"}},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

from app import (login, signup, home, dashboard, prediction_step1,
                 prediction_step2, prediction_step3, prediction_result,
                 analytics, insights, history, affirmations)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='text-align:center;padding:20px 0 24px;'>
  <div style='font-size:2.2rem;'>🧠</div>
  <div style='font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#A78BFA,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    letter-spacing:-0.5px;'>ScreenSense</div>
  <div style='font-size:0.72rem;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-top:2px;'>AI Wellbeing Platform</div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.logged_in:
        st.markdown(f"""
<div style='background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.3);
  border-radius:12px;padding:12px 14px;margin-bottom:16px;'>
  <div style='font-size:0.72rem;color:#7C3AED;font-weight:600;text-transform:uppercase;letter-spacing:1px;'>Signed in as</div>
  <div style='font-weight:600;color:#E2E8F0;margin-top:2px;'>{st.session_state.username}</div>
</div>
""", unsafe_allow_html=True)

        nav_items = ["🏠 Home","📊 Dashboard","🔮 Predict Wellbeing",
                     "📈 Analytics","💡 Insights","📜 My History","✨ Affirmations"]
        nav_map = {
            "🏠 Home":"Home","📊 Dashboard":"Dashboard",
            "🔮 Predict Wellbeing":"Predict","📈 Analytics":"Analytics",
            "💡 Insights":"Insights","📜 My History":"History",
            "✨ Affirmations":"Affirmations"
        }
        choice = st.radio("Navigation", nav_items, label_visibility="collapsed")
        if nav_map[choice] != st.session_state.page:
            st.session_state.page = nav_map[choice]
            if nav_map[choice] == "Predict":
                st.session_state.pred_step = 1
                st.session_state.pred_data = {}

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "Home"
            st.rerun()
    else:
        nav_items = ["🏠 Home","🔑 Login","✨ Sign Up"]
        nav_map = {"🏠 Home":"Home","🔑 Login":"Login","✨ Sign Up":"Signup"}
        choice = st.radio("Navigation", nav_items, label_visibility="collapsed")
        st.session_state.page = nav_map[choice]

    st.markdown("""
<div style='position:fixed;bottom:16px;left:0;right:0;text-align:center;
  font-size:0.68rem;color:#334155;'>ScreenSense v1.0 · Built with AI</div>
""", unsafe_allow_html=True)

# ── Page render ─────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "Home":
    home.render()
elif page == "Login":
    login.render()
elif page == "Signup":
    signup.render()
elif not st.session_state.logged_in:
    st.warning("Please log in to access this page.")
    login.render()
elif page == "Dashboard":
    dashboard.render()
elif page == "Predict":
    step = st.session_state.pred_step
    if step == 1:    prediction_step1.render()
    elif step == 2:  prediction_step2.render()
    elif step == 3:  prediction_step3.render()
    else:            prediction_result.render()
elif page == "Analytics":
    analytics.render()
elif page == "Insights":
    insights.render()
elif page == "History":
    history.render()
elif page == "Affirmations":
    affirmations.render()
