"""
================================================================================
  Digital Wellbeing Predictor  ·  Full Streamlit App  v3
  ────────────────────────────────────────────────────────
  • Modern glassmorphism UI with smooth hover animations
  • Mobile-responsive layout
  • Login / Signup with SHA-256+salt hashing & password rules
  • 5-step wizard (Intro → Screen → Notifications → Mental → Predict)
  • Full analysis results page
  • PDF report with proper text alignment (ReportLab)
  • User profile & history
  • Logout & Delete account
  • Dark mode only (no toggle)
================================================================================
  Run:  python -m streamlit run app/streamlit_app.py
================================================================================
"""

import os, json, hashlib, secrets, re, io, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MDL  = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data", "cleaned_dataset.csv")
DB   = os.path.join(ROOT, "data", "users.json")

ALL_FEATURES = [
    "daily_screen_time", "num_app_switches", "sleep_hours",
    "notification_count", "social_media_time_min",
    "focus_score", "mood_score", "anxiety_level",
    "dopamine_index", "focus_efficiency", "sleep_deficit",
]

# ── ML artefacts ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml():
    m  = joblib.load(os.path.join(MDL, "best_model.pkl"))
    sc = joblib.load(os.path.join(MDL, "scaler.pkl"))
    df = pd.read_csv(DATA)
    return m, sc, df

model, scaler, dataset = load_ml()

# ══════════════════════════════════════════════════════════════════════════════
#  USER DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def load_db():
    if not os.path.exists(DB):
        return {}
    with open(DB) as f:
        return json.load(f)

def save_db(db):
    with open(DB, "w") as f:
        json.dump(db, f, indent=2)

def hash_pw(pw: str) -> str:
    salt = secrets.token_hex(16)
    return f"{salt}${hashlib.sha256((salt+pw).encode()).hexdigest()}"

def verify_pw(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
        return hashlib.sha256((salt+pw).encode()).hexdigest() == h
    except Exception:
        return False

def validate_pw(pw: str):
    e = []
    if len(pw) < 8:                               e.append("At least 8 characters")
    if not re.search(r"[A-Z]", pw):               e.append("At least one uppercase letter")
    if not re.search(r"[0-9]", pw):               e.append("At least one digit")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-]', pw): e.append("At least one special character")
    return e

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Digital Wellbeing",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — glassmorphism + mobile-responsive + animations
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #060818 !important;
    color: #e2e8f8 !important;
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(139,92,246,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 60% 30%, rgba(59,130,246,0.10) 0%, transparent 55%);
    pointer-events: none;
    z-index: 0;
}
.block-container { position: relative; z-index: 1; padding-top: 1.5rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div {
    background: rgba(10, 12, 30, 0.85) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #e2e8f8 !important; }

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(99,102,241,0.18), inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.hero-sub {
    font-size: clamp(0.88rem, 2vw, 1.05rem);
    color: #94a3b8;
    line-height: 1.75;
}

/* ── Step badge ── */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 50px;
    padding: 4px 16px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.8rem;
}

/* ── Section headings ── */
.sec-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(1.2rem, 3vw, 1.5rem);
    font-weight: 700;
    color: #e2e8f8;
    margin-bottom: 0.25rem;
}
.sec-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.4rem;
    line-height: 1.6;
}

/* ── KPI card ── */
.kpi {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: all 0.25s ease;
    cursor: default;
}
.kpi:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.35);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.2);
}
.kpi-val {
    font-family: 'Sora', sans-serif;
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.kpi-lbl {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Recommendation card ── */
.rec {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #6366f1;
    border-top: 1px solid rgba(255,255,255,0.07);
    border-right: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    border-radius: 0 14px 14px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}
.rec:hover {
    background: rgba(99,102,241,0.08);
    border-left-color: #a5b4fc;
    transform: translateX(4px);
}

/* ── Score ring ── */
.score-hero {
    text-align: center;
    padding: 2rem;
}
.score-number {
    font-family: 'Sora', sans-serif;
    font-size: clamp(3.5rem, 8vw, 6rem);
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: clamp(1rem, 2.5vw, 1.4rem);
    font-weight: 700;
    margin-top: 0.4rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.93rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Form inputs ── */
.stTextInput input, .stSelectbox > div > div, .stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stSelectbox > div > div:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0.4rem 0; }
.stSlider [data-testid="stThumbValue"] { color: #a5b4fc !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(99,102,241,0.4),rgba(139,92,246,0.35)) !important;
    color: #a5b4fc !important;
}

/* ── Progress steps ── */
.prog-step { text-align:center; font-size:0.72rem; color:#475569; padding:0.3rem 0.1rem; }
.prog-step.done  { color:#4ade80; }
.prog-step.curr  { color:#a5b4fc; font-weight:700; }
.prog-step.ahead { color:#334155; }

/* ── Auth page ── */
.auth-logo { text-align:center; padding:2rem 0 1.5rem; }
.auth-logo .icon { font-size:3.5rem; }
.auth-logo .name {
    font-family:'Sora',sans-serif;
    font-size:clamp(1.5rem,4vw,2rem);
    font-weight:800;
    background:linear-gradient(135deg,#a5b4fc,#c4b5fd);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin-top:0.4rem;
}
.auth-logo .tag { font-size:0.88rem; color:#475569; margin-top:0.25rem; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 1rem 0 !important; }

/* ── DataFrames ── */
.stDataFrame { border-radius:14px !important; overflow:hidden !important; }

/* ── Alerts ── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .glass { padding: 1.2rem 1.4rem !important; }
    .hero  { padding: 1.5rem 1.4rem !important; }
    .block-container { padding: 0.8rem !important; }
    .kpi-val { font-size: 1.5rem !important; }
}
@media (max-width: 480px) {
    .hero-title { font-size: 1.6rem !important; }
    .sec-title  { font-size: 1.1rem !important; }
    .glass { padding: 1rem !important; border-radius: 14px !important; }
}

/* ── Sidebar nav button ── */
.sidebar-btn button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
.sidebar-btn button:hover {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.4) !important;
    color: #a5b4fc !important;
    transform: translateX(4px) translateY(0) !important;
    box-shadow: none !important;
}

/* ── Markdown table ── */
table { width:100% !important; border-collapse:collapse !important; }
th { background:rgba(99,102,241,0.2) !important; color:#a5b4fc !important;
     padding:8px 12px !important; font-size:0.82rem !important; }
td { padding:7px 12px !important; font-size:0.88rem !important;
     border-bottom:1px solid rgba(255,255,255,0.05) !important; }
tr:hover td { background:rgba(255,255,255,0.03) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_D = dict(logged_in=False, username="", page="auth",
          inputs={}, prediction=None, confirm_delete=False)
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def go(p):
    st.session_state.page = p
    st.rerun()

def get_user():
    return load_db().get(st.session_state.username, {})

def upd_user(d):
    db = load_db(); db[st.session_state.username].update(d); save_db(db)

def predict_score(inp):
    r = dict(inp)
    r["dopamine_index"]   = inp["social_media_time_min"] + inp["notification_count"] + inp["num_app_switches"]
    r["focus_efficiency"] = inp["focus_score"] / (inp["daily_screen_time"] + 1e-9)
    r["sleep_deficit"]    = 8 - inp["sleep_hours"]
    X = pd.DataFrame([r])[ALL_FEATURES]
    s = float(model.predict(scaler.transform(X))[0])
    return max(0.0, min(100.0, s)), r

def score_meta(s):
    if s >= 70: return "Excellent",       "#4ade80", "🟢"
    if s >= 50: return "Good",            "#facc15", "🟡"
    if s >= 35: return "Moderate",        "#fb923c", "🟠"
    return       "Needs Attention",       "#f87171", "🔴"

def kpi_html(val, lbl):
    return f"""<div class='kpi'>
      <div class='kpi-val'>{val}</div>
      <div class='kpi-lbl'>{lbl}</div>
    </div>"""

def kpi(val, lbl):
    st.markdown(kpi_html(val, lbl), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT  — fixed alignment throughout
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(user, inp, score, eng):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm
    )
    W = 17*cm
    lbl, chex, _ = score_meta(score)
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    # ── Styles ──────────────────────────────────────────────────────────────
    TITLE = ParagraphStyle("TITLE",
        fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#5b63d3"),
        alignment=TA_CENTER, spaceAfter=4, leading=28)
    SUB = ParagraphStyle("SUB",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#666680"),
        alignment=TA_CENTER, spaceAfter=20, leading=14)
    H2 = ParagraphStyle("H2",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1f3c"),
        spaceAfter=8, spaceBefore=16, leading=18)
    BODY = ParagraphStyle("BODY",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#2c2c54"),
        spaceAfter=5, leading=16, leftIndent=0)
    CELL_H = ParagraphStyle("CELL_H",
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_LEFT, leading=13)
    CELL = ParagraphStyle("CELL",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#2c2c54"),
        alignment=TA_LEFT, leading=13)
    CELL_KEY = ParagraphStyle("CELL_KEY",
        fontSize=9, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#5b63d3"),
        alignment=TA_LEFT, leading=13)
    CELL_VAL = ParagraphStyle("CELL_VAL",
        fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#1a1f3c"),
        alignment=TA_LEFT, leading=13)
    SCORE_BIG = ParagraphStyle("SCORE_BIG",
        fontSize=30, fontName="Helvetica-Bold",
        textColor=colors.HexColor(chex),
        alignment=TA_CENTER, leading=36)
    SCORE_LBL = ParagraphStyle("SCORE_LBL",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor(chex),
        alignment=TA_CENTER, leading=18)
    FOOTER = ParagraphStyle("FOOTER",
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#aaaaaa"),
        alignment=TA_CENTER, leading=12)

    ACCENT  = colors.HexColor("#5b63d3")
    LIGHT   = colors.HexColor("#f5f6ff")
    WHITE   = colors.white
    BORDER  = colors.HexColor("#dde3ff")
    ROW_ALT = colors.HexColor("#f9f9ff")

    # ── Helper: styled table ─────────────────────────────────────────────────
    def make_table(rows, col_widths, has_header=True):
        t = Table(rows, colWidths=col_widths, repeatRows=1 if has_header else 0)
        base = [
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("TOPPADDING",  (0,0), (-1,-1), 7),
            ("BOTTOMPADDING",(0,0),(-1,-1), 7),
            ("LEFTPADDING", (0,0), (-1,-1), 9),
            ("RIGHTPADDING",(0,0), (-1,-1), 9),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
            ("GRID",      (0,0), (-1,-1), 0.4, BORDER),
        ]
        if has_header:
            base += [
                ("BACKGROUND", (0,0), (-1,0), ACCENT),
                ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, WHITE]),
            ]
        else:
            base += [("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT, WHITE])]
        t.setStyle(TableStyle(base))
        return t

    story = []

    # ── Page 1: Header + User Info + Score ───────────────────────────────────
    story.append(Paragraph("🧠  Digital Wellbeing Report", TITLE))
    story.append(Paragraph(f"Generated on {now}", SUB))
    story.append(HRFlowable(width=W, color=ACCENT, thickness=2, spaceAfter=12))

    # User info grid (2-col key-value)
    story.append(Paragraph("User Information", H2))
    ui_rows = [
        [Paragraph("Full Name",  CELL_KEY), Paragraph(user.get("name","—"),     CELL_VAL),
         Paragraph("Username",   CELL_KEY), Paragraph(user.get("username","—"), CELL_VAL)],
        [Paragraph("Age",        CELL_KEY), Paragraph(str(user.get("age","—")), CELL_VAL),
         Paragraph("Email",      CELL_KEY), Paragraph(user.get("email","—"),    CELL_VAL)],
        [Paragraph("Gender",     CELL_KEY), Paragraph(user.get("gender","—"),   CELL_VAL),
         Paragraph("Report Date",CELL_KEY), Paragraph(now.split(",")[0],        CELL_VAL)],
    ]
    story.append(make_table(ui_rows, [3.4*cm, 5.1*cm, 3.4*cm, 5.1*cm], has_header=False))
    story.append(Spacer(1, 0.5*cm))

    # Score box
    story.append(HRFlowable(width=W, color=BORDER, thickness=1, spaceAfter=10))
    story.append(Paragraph("Predicted Digital Wellbeing Score", H2))
    sc_row = [[Paragraph(f"{score:.1f} / 100", SCORE_BIG),
               Paragraph(lbl, SCORE_LBL)]]
    sc_t = Table(sc_row, colWidths=[8.5*cm, 8.5*cm])
    sc_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
        ("GRID",          (0,0), (-1,-1), 0.4, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(sc_t)
    story.append(Spacer(1, 0.6*cm))

    # ── Daily habit inputs ────────────────────────────────────────────────────
    story.append(Paragraph("Daily Habit Inputs", H2))
    habit_rows = [
        [Paragraph("Metric",              CELL_H), Paragraph("Your Value", CELL_H),
         Paragraph("Metric",              CELL_H), Paragraph("Your Value", CELL_H)],
        [Paragraph("Daily Screen Time",   CELL),   Paragraph(f"{inp['daily_screen_time']} hrs", CELL),
         Paragraph("Sleep Hours",         CELL),   Paragraph(f"{inp['sleep_hours']} hrs",       CELL)],
        [Paragraph("App Switches",        CELL),   Paragraph(str(inp['num_app_switches']),       CELL),
         Paragraph("Notifications / Day", CELL),   Paragraph(str(inp['notification_count']),    CELL)],
        [Paragraph("Social Media",        CELL),   Paragraph(f"{inp['social_media_time_min']} min", CELL),
         Paragraph("Focus Score",         CELL),   Paragraph(f"{inp['focus_score']} / 100",    CELL)],
        [Paragraph("Mood Score",          CELL),   Paragraph(f"{inp['mood_score']} / 100",     CELL),
         Paragraph("Anxiety Level",       CELL),   Paragraph(f"{inp['anxiety_level']} / 10",  CELL)],
    ]
    story.append(make_table(habit_rows, [4.5*cm, 4*cm, 4.5*cm, 4*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Engineered indicators ─────────────────────────────────────────────────
    story.append(Paragraph("Computed Wellness Indicators", H2))
    eng_rows = [
        [Paragraph("Indicator",        CELL_H),
         Paragraph("Your Value",       CELL_H),
         Paragraph("What it means",    CELL_H)],
        [Paragraph("Dopamine Index",   CELL),
         Paragraph(f"{eng['dopamine_index']:.0f}", CELL),
         Paragraph("Social media + notifications + app switches. Aim < 200.", CELL)],
        [Paragraph("Focus Efficiency", CELL),
         Paragraph(f"{eng['focus_efficiency']:.2f}", CELL),
         Paragraph("Focus score per screen hour. Higher values are better.", CELL)],
        [Paragraph("Sleep Deficit",    CELL),
         Paragraph(f"{eng['sleep_deficit']:.1f} hrs", CELL),
         Paragraph("Hours short of recommended 8 hrs. Zero or negative is ideal.", CELL)],
    ]
    story.append(make_table(eng_rows, [4.4*cm, 3*cm, 9.6*cm]))
    story.append(Spacer(1, 0.5*cm))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, color=BORDER, thickness=1, spaceAfter=10))
    story.append(Paragraph("Personalised Recommendations", H2))

    recs = []
    if inp["sleep_hours"]           < 7:
        recs.append(("Improve Sleep",
                     "You are averaging less than 7 hours of sleep. Aim for 7 to 9 hours nightly for optimal cognitive recovery and mood regulation."))
    if inp["daily_screen_time"]     > 8:
        recs.append(("Reduce Screen Time",
                     "Daily screen usage exceeds 8 hours. Schedule regular screen-free breaks of at least 20 minutes every 2 hours."))
    if inp["notification_count"]    > 80:
        recs.append(("Batch Notifications",
                     "Over 80 daily notifications fragment your attention span. Enable Do Not Disturb during focus and sleep hours."))
    if inp["anxiety_level"]         >= 7:
        recs.append(("Manage Anxiety",
                     "High anxiety detected. Consider mindfulness, deep-breathing exercises, or speaking with a mental health professional."))
    if inp["social_media_time_min"] > 120:
        recs.append(("Limit Social Media",
                     "Cap social media usage to under 2 hours per day to lower your dopamine index and improve sustained focus."))
    if inp["focus_score"]           < 50:
        recs.append(("Boost Focus",
                     "Try the Pomodoro technique (25 min work / 5 min break) or use app-blocking tools like Freedom during work sessions."))
    if not recs:
        recs.append(("Excellent Habits",
                     "Your digital behaviours are well-balanced. Continue maintaining your healthy screen habits and sleep schedule."))

    rec_data = []
    for title, body in recs:
        rec_data.append([
            Paragraph(f"<b>{title}</b>", ParagraphStyle("RT",
                fontSize=9, fontName="Helvetica-Bold",
                textColor=ACCENT, leading=14)),
            Paragraph(body, BODY)
        ])
    if rec_data:
        rt = Table(rec_data, colWidths=[3.5*cm, 13.5*cm])
        rt.setStyle(TableStyle([
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT, WHITE]),
            ("GRID",         (0,0), (-1,-1), 0.3, BORDER),
            ("LINEAFTER",    (0,0), (0,-1),  2,   ACCENT),
        ]))
        story.append(rt)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width=W, color=ACCENT, thickness=1, spaceAfter=8))
    story.append(Paragraph(
        "Digital Wellbeing ML Predictor  ·  For educational and research purposes only.",
        FOOTER
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    user = get_user()
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center;padding:1.4rem 0 0.6rem'>
          <div style='font-size:2.8rem'>👤</div>
          <div style='font-family:Sora,sans-serif;font-weight:700;font-size:1rem;
               color:#e2e8f8'>{user.get('name','User')}</div>
          <div style='font-size:0.78rem;color:#475569;margin-top:3px'>
               @{st.session_state.username}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        nav = [("🏠","Home","home"),("📊","New Analysis","step2"),
               ("👤","Profile","profile"),("⚙️","Settings","settings")]
        for icon, label, key in nav:
            st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                go(key)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚪  Logout", use_container_width=True, key="nav_logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            for k, v in _D.items(): st.session_state[k] = v
            st.rerun()

        if st.button("🗑️  Delete Account", use_container_width=True, key="nav_del"):
            st.session_state.confirm_delete = True; st.rerun()

        if st.session_state.confirm_delete:
            st.warning("⚠️ This cannot be undone!")
            c1, c2 = st.columns(2)
            if c1.button("Yes", key="del_yes"):
                db = load_db(); db.pop(st.session_state.username, None); save_db(db)
                for k in list(st.session_state.keys()): del st.session_state[k]
                for k, v in _D.items(): st.session_state[k] = v
                st.rerun()
            if c2.button("No",  key="del_no"):
                st.session_state.confirm_delete = False; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  WIZARD PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════
def wizard_prog(cur):
    steps = [("1","Intro"),("2","Screen"),("3","Notifs & Sleep"),("4","Mental"),("5","Predict")]
    cols = st.columns(len(steps))
    for i, (col, (n, lbl)) in enumerate(zip(cols, steps)):
        c = i+1
        if   c < cur: cls, icon = "done",  "✅"
        elif c == cur: cls, icon = "curr",  "●"
        else:          cls, icon = "ahead", "○"
        col.markdown(f"<div class='prog-step {cls}'>{icon}<br>{lbl}</div>",
                     unsafe_allow_html=True)
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AUTH
# ══════════════════════════════════════════════════════════════════════════════
def page_auth():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div class='auth-logo'>
          <div class='icon'>🧠</div>
          <div class='name'>Digital Wellbeing</div>
          <div class='tag'>Understand your digital health, one habit at a time.</div>
        </div>""", unsafe_allow_html=True)

        t_li, t_su = st.tabs(["🔑  Login", "✨  Sign Up"])

        # ── LOGIN ──────────────────────────────────────────────────────────
        with t_li:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("#### Welcome back!")
            uname = st.text_input("Username", key="li_u", placeholder="e.g. harini_m")
            pwd   = st.text_input("Password", key="li_p", type="password",
                                  placeholder="Your password")
            if st.button("Login  →", key="btn_li", use_container_width=True):
                db = load_db()
                if uname not in db:
                    st.error("❌ Username not found.")
                elif not verify_pw(pwd, db[uname].get("password","")):
                    st.error("❌ Incorrect password.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    go("home")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── SIGN UP ────────────────────────────────────────────────────────
        with t_su:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("#### Create your account")

            c1, c2 = st.columns(2)
            su_name  = c1.text_input("Full Name *",  key="su_nm",
                                     placeholder="e.g. John Cena")
            su_age   = c2.text_input("Age *",        key="su_ag",
                                     placeholder="e.g. 46")
            su_email = st.text_input("Email Address *", key="su_em",
                                     placeholder="e.g. johncena@gmail.com")
            c3, c4   = st.columns(2)
            su_gender= c3.selectbox("Gender", ["Prefer not to say","Female","Male","Non-binary"],
                                    key="su_gd")
            su_user  = c4.text_input("Username *",   key="su_un",
                                     placeholder="e.g. john_cena16")
            su_pwd   = st.text_input("Password *",   key="su_pw", type="password",
                                     placeholder="e.g. YouCantSeeMe@1")
            su_pwd2  = st.text_input("Confirm Password *", key="su_p2", type="password",
                                     placeholder="Repeat your password")

            st.markdown("""
            <div style='font-size:0.76rem;color:#475569;margin-top:0.2rem;
                        padding:8px 12px;background:rgba(99,102,241,0.08);
                        border-radius:8px;border:1px solid rgba(99,102,241,0.15)'>
            🔒 Password rules: ≥ 8 chars · 1 uppercase (A–Z) · 1 digit (0–9) · 1 special char (!@#$…)
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account  →", key="btn_su", use_container_width=True):
                db   = load_db()
                errs = []
                if not all([su_name.strip(), su_age.strip(), su_email.strip(),
                            su_user.strip(), su_pwd, su_pwd2]):
                    errs.append("All fields marked * are required.")
                if su_user.strip() in db:
                    errs.append("Username already taken — try another.")
                if su_pwd != su_pwd2:
                    errs.append("Passwords do not match.")
                errs += validate_pw(su_pwd)
                if not re.match(r"[^@]+@[^@]+\.[^@]+", su_email):
                    errs.append("Invalid email format.")
                try:
                    if not (1 <= int(su_age.strip()) <= 120): raise ValueError
                except ValueError:
                    errs.append("Age must be a number between 1 and 120.")

                if errs:
                    for e in errs: st.error(e)
                else:
                    db[su_user.strip()] = {
                        "username": su_user.strip(),
                        "name":     su_name.strip(),
                        "age":      su_age.strip(),
                        "email":    su_email.strip(),
                        "gender":   su_gender,
                        "password": hash_pw(su_pwd),
                        "created":  datetime.datetime.now().isoformat(),
                        "history":  [],
                    }
                    save_db(db)
                    st.success("✅ Account created! Switch to Login to continue.")
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    user    = get_user()
    fname   = user.get("name","there").split()[0]
    history = user.get("history",[])

    st.markdown(f"""
    <div class='hero'>
      <div class='hero-title'>Hello, {fname}! 👋</div>
      <div class='hero-sub'>
        Welcome to your <b>Digital Wellbeing Dashboard</b>.<br>
        Analyse how your daily digital habits affect your mental wellbeing.<br>
        Adjust the sliders, hit <b>Predict</b>, and see where you stand.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    last_s = f"{history[-1]['score']:.1f}" if history else "—"
    with c1: kpi("🧠", "Wellbeing Tracker")
    with c2: kpi(last_s, "Last Score")
    with c3: kpi(str(len(history)), "Analyses Done")

    st.markdown("---")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 🚀 Start a New Analysis")
    st.markdown("Walk through **5 quick steps** to assess your digital wellbeing score.")
    if st.button("Begin Analysis  →", use_container_width=True, key="home_start"):
        go("step2")
    st.markdown('</div>', unsafe_allow_html=True)

    if history:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 📈 Score History")
        last10  = history[-10:]
        scores  = [h["score"] for h in last10]
        dates   = [h["date"][:10] for h in last10]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(len(scores)), scores, "o-", color="#818cf8", lw=2.5, ms=8,
                markerfacecolor="#a5b4fc", markeredgecolor="#6366f1", markeredgewidth=1.5)
        ax.fill_between(range(len(scores)), [max(0,s-3) for s in scores],
                        [min(100,s+3) for s in scores],
                        alpha=0.15, color="#818cf8")
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=8, color="#94a3b8")
        ax.set_ylabel("Score", fontsize=9, color="#94a3b8")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#1e293b")
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.grid(axis="y", alpha=0.15, color="#334155")
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: INTRO
# ══════════════════════════════════════════════════════════════════════════════
def page_step2():
    wizard_prog(1)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">⚡ Step 1 of 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Enter the Quantities of Your Daily Habits</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">We will ask about your screen usage, notifications, '
                'sleep, and mental state across the next few steps.</div>', unsafe_allow_html=True)
    st.markdown("""
    | Category | What we measure |
    |---|---|
    | 📱 **Screen Usage** | Screen time · App switches · Social media |
    | 🔔 **Notifications & Sleep** | Daily interruptions · Sleep hours |
    | 🧘 **Mental State** | Focus level · Mood · Anxiety |

    Your responses are **private** — only you can see them. Each step takes under **30 seconds**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    _, cc, _ = st.columns([1,2,1])
    with cc:
        if st.button("Let's Start  →", use_container_width=True, key="s2_go"):
            go("step3")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: SCREEN USAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_step3():
    wizard_prog(2)
    inp = st.session_state.inputs
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">📱 Step 2 of 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Screen Usage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">How much time do you spend on your devices each day?</div>',
                unsafe_allow_html=True)
    screen = st.slider("Daily Screen Time (hours)",        1.0, 16.0,
                        float(inp.get("daily_screen_time",7.0)), 0.5)
    apps   = st.slider("Number of App Switches per Day",   5,   200,
                        int(inp.get("num_app_switches",60)), 5)
    sm     = st.slider("Social Media Time (minutes/day)",  0,   600,
                        int(inp.get("social_media_time_min",120)), 10)
    st.info("💡 **Tip:** App switches and social media time fuel your *Dopamine Index*. "
            "High scores indicate stimulus-seeking behaviour linked to reduced focus.")
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("← Back", key="s3b", use_container_width=True): go("step2")
    if c2.button("Next →", key="s3n", use_container_width=True):
        inp.update({"daily_screen_time":screen,"num_app_switches":apps,"social_media_time_min":sm})
        go("step4")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: NOTIFICATIONS & SLEEP
# ══════════════════════════════════════════════════════════════════════════════
def page_step4():
    wizard_prog(3)
    inp = st.session_state.inputs
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">🔔 Step 3 of 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Notifications & Sleep</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">How many interruptions do you receive, '
                'and how much sleep are you getting?</div>', unsafe_allow_html=True)
    notifs = st.slider("Notifications Received per Day", 0,   200,
                        int(inp.get("notification_count",80)), 5)
    sleep  = st.slider("Hours of Sleep Last Night",      3.0, 10.0,
                        float(inp.get("sleep_hours",7.0)), 0.25)
    deficit = 8 - sleep
    if deficit > 2:   st.error(  f"😴 Sleep Deficit: **{deficit:.1f} hrs** — Significantly under target!")
    elif deficit > 0: st.warning(f"⚡ Sleep Deficit: **{deficit:.1f} hrs** — Slightly below 8 hrs.")
    else:             st.success(f"✅ Sleep Surplus: **{abs(deficit):.1f} hrs** — Great sleep!")
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("← Back", key="s4b", use_container_width=True): go("step3")
    if c2.button("Next →", key="s4n", use_container_width=True):
        inp.update({"notification_count":notifs,"sleep_hours":sleep})
        go("step5")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: MENTAL STATE
# ══════════════════════════════════════════════════════════════════════════════
def page_step5():
    wizard_prog(4)
    inp = st.session_state.inputs
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="step-pill">🧘 Step 4 of 5</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Mental State</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Rate your focus, mood, and anxiety honestly — '
                'this is completely private to you.</div>', unsafe_allow_html=True)
    focus = st.slider("Focus Score  (0 = distracted → 100 = laser focused)", 10, 100,
                       int(inp.get("focus_score",65)), 1)
    mood  = st.slider("Mood Score   (0 = very low → 100 = excellent)",        10, 100,
                       int(inp.get("mood_score",65)), 1)
    anx   = st.slider("Anxiety Level (1 = calm → 10 = very anxious)",         1,  10,
                       int(inp.get("anxiety_level",4)), 1)
    st.markdown("---")
    st.markdown("**Quick summary of your inputs so far:**")
    mc = st.columns(4)
    prev = {**inp, "focus_score":focus, "mood_score":mood, "anxiety_level":anx}
    with mc[0]: kpi(f"{prev.get('daily_screen_time','—')} hrs","Screen Time")
    with mc[1]: kpi(f"{prev.get('sleep_hours','—')} hrs",      "Sleep")
    with mc[2]: kpi(f"{focus}/100",                             "Focus")
    with mc[3]: kpi(f"{anx}/10",                                "Anxiety")
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("← Back", key="s5b", use_container_width=True): go("step4")
    if c2.button("🔍  Predict My Wellbeing Score →", key="s5p", use_container_width=True):
        inp.update({"focus_score":focus,"mood_score":mood,"anxiety_level":anx})
        score, eng = predict_score(inp)
        st.session_state.prediction = {"score":score,"eng":eng}
        db = load_db()
        db[st.session_state.username].setdefault("history",[]).append({
            "date":datetime.datetime.now().isoformat(),
            "score":round(score,2),
            "inputs":inp,
        })
        save_db(db)
        go("result")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_result():
    pred = st.session_state.prediction
    if not pred: go("home"); return
    score = pred["score"]; eng = pred["eng"]; inp = st.session_state.inputs
    lbl, chex, emj = score_meta(score)
    user = get_user()

    # Score hero
    st.markdown(f"""
    <div class='glass' style='text-align:center'>
      <div style='font-size:.75rem;text-transform:uppercase;letter-spacing:.14em;
                  color:#475569;margin-bottom:.5rem'>Your Digital Wellbeing Score</div>
      <div class='score-number' style='color:{chex}'>{score:.1f}</div>
      <div class='score-label' style='color:{chex}'>{emj} {lbl}</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: kpi(f"{eng['dopamine_index']:.0f}",    "Dopamine Index")
    with c2: kpi(f"{eng['focus_efficiency']:.1f}",  "Focus Efficiency")
    with c3: kpi(f"{eng['sleep_deficit']:.1f} hrs", "Sleep Deficit")
    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 🕸️ Your Profile vs. Average")
        RL   = ["Screen","App Sw.","Sleep","Notifs","SocMedia","Focus","Mood","Anxiety"]
        uv   = [inp["daily_screen_time"], inp["num_app_switches"]/10,
                inp["sleep_hours"],        inp["notification_count"]/20,
                inp["social_media_time_min"]/60,
                inp["focus_score"]/10,     inp["mood_score"]/10, inp["anxiety_level"]]
        av   = [dataset["daily_screen_time"].mean(),   dataset["num_app_switches"].mean()/10,
                dataset["sleep_hours"].mean(),          dataset["notification_count"].mean()/20,
                dataset["social_media_time_min"].mean()/60,
                dataset["focus_score"].mean()/10,       dataset["mood_score"].mean()/10,
                dataset["anxiety_level"].mean()]
        ang  = np.linspace(0,2*np.pi,len(RL),endpoint=False).tolist()
        uv2  = uv+uv[:1]; av2 = av+av[:1]; ang2 = ang+ang[:1]
        fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
        ax.plot(ang2,uv2,"o-",lw=2.5,color="#818cf8",label="You",ms=7,
                markerfacecolor="#a5b4fc")
        ax.fill(ang2,uv2,alpha=0.22,color="#818cf8")
        ax.plot(ang2,av2,"o-",lw=2,color="#fb923c",label="Avg",ms=6,alpha=0.8)
        ax.fill(ang2,av2,alpha=0.10,color="#fb923c")
        ax.set_xticks(ang); ax.set_xticklabels(RL,fontsize=7.5,color="#94a3b8")
        ax.tick_params(colors="#94a3b8"); ax.grid(color="#1e293b",alpha=0.6)
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=8,
                  labelcolor="#94a3b8",facecolor="#0d1117",edgecolor="#1e293b")
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📊 Key Metrics vs. Average")
        keys  = ["Screen\n(hrs)","Sleep\n(hrs)","Focus\n(/10)","Mood\n(/10)","Anxiety\n(/10)"]
        ubars = [inp["daily_screen_time"],   inp["sleep_hours"],
                 inp["focus_score"]/10,       inp["mood_score"]/10, inp["anxiety_level"]]
        abars = [dataset["daily_screen_time"].mean(), dataset["sleep_hours"].mean(),
                 dataset["focus_score"].mean()/10,     dataset["mood_score"].mean()/10,
                 dataset["anxiety_level"].mean()]
        x = np.arange(len(keys)); w = 0.34
        fig, ax = plt.subplots(figsize=(4.5,4.5))
        ax.bar(x-w/2,ubars,w,label="You",    color="#818cf8",alpha=0.9)
        ax.bar(x+w/2,abars,w,label="Average",color="#fb923c",alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(keys,fontsize=7.5,color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.legend(fontsize=8,labelcolor="#94a3b8",facecolor="#0d1117",edgecolor="#1e293b")
        ax.grid(axis="y",alpha=0.15,color="#334155")
        ax.spines[:].set_color("#1e293b")
        ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 💡 Personalised Recommendations")
    recs = []
    if inp["sleep_hours"]           < 7:    recs.append(("🌙","Improve Sleep","Aim for 7–9 hrs nightly for optimal recovery."))
    if inp["daily_screen_time"]     > 8:    recs.append(("📱","Reduce Screen Time","Over 8 hrs detected. Schedule device-free breaks."))
    if inp["notification_count"]    > 80:   recs.append(("🔕","Batch Notifications","Enable Do Not Disturb during focus & sleep hours."))
    if inp["anxiety_level"]         >= 7:   recs.append(("🧘","Manage Anxiety","Try mindfulness or speak with a professional."))
    if inp["social_media_time_min"] > 120:  recs.append(("🌐","Limit Social Media","Cap to under 2 hrs/day."))
    if inp["focus_score"]           < 50:   recs.append(("🎯","Boost Focus","Try Pomodoro or app-blocking tools."))
    if not recs:                            recs.append(("🎉","Excellent Habits!","Keep it up — your digital habits are well-balanced!"))
    for icon, title, body in recs:
        st.markdown(f"""<div class='rec'>
          <b>{icon} {title}</b><br>
          <span style='font-size:.88rem;color:#94a3b8'>{body}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📄 Download Your Report")
    pdf_data = build_pdf(user, inp, score, eng)
    fname = f"wellbeing_{st.session_state.username}_{datetime.date.today()}.pdf"
    st.download_button("⬇️  Download PDF Report", data=pdf_data,
                       file_name=fname, mime="application/pdf",
                       use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("🔄 New Analysis", use_container_width=True): go("step2")
    if c2.button("🏠 Back to Home", use_container_width=True): go("home")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def page_profile():
    user = get_user()
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:1.2rem;margin-bottom:1.5rem'>
      <div style='font-size:3.2rem'>👤</div>
      <div>
        <div class='sec-title'>{user.get('name','User')}</div>
        <div style='font-size:0.82rem;color:#475569'>
          @{st.session_state.username} &nbsp;·&nbsp;
          Member since {user.get('created','—')[:10]}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("#### ✏️ Edit Profile")
    c1, c2 = st.columns(2)
    n_name   = c1.text_input("Full Name",   value=user.get("name",""),      key="pn")
    n_age    = c2.text_input("Age",         value=str(user.get("age","")),  key="pa")
    n_email  = st.text_input("Email",       value=user.get("email",""),     key="pe")
    glist    = ["Prefer not to say","Female","Male","Non-binary"]
    n_gender = st.selectbox("Gender", glist,
                            index=glist.index(user.get("gender","Prefer not to say")), key="pg")
    st.markdown("#### 🔑 Change Password")
    op = st.text_input("Current Password",     type="password", key="pop")
    np1= st.text_input("New Password",         type="password", key="pnp")
    np2= st.text_input("Confirm New Password", type="password", key="pnp2")
    if st.button("💾 Save Changes", use_container_width=True, key="psave"):
        errs = []
        if not n_name.strip(): errs.append("Name cannot be empty.")
        try:
            if not (1 <= int(n_age.strip()) <= 120): raise ValueError
        except: errs.append("Age must be a valid number.")
        if not re.match(r"[^@]+@[^@]+\.[^@]+", n_email): errs.append("Invalid email.")
        pp = {}
        if op or np1 or np2:
            if not verify_pw(op, user.get("password","")):  errs.append("Current password is incorrect.")
            elif np1 != np2:                                 errs.append("New passwords do not match.")
            else:
                pe = validate_pw(np1)
                if pe: errs += pe
                else:  pp["password"] = hash_pw(np1)
        if errs:
            for e in errs: st.error(e)
        else:
            upd_user({"name":n_name.strip(),"age":n_age.strip(),
                      "email":n_email.strip(),"gender":n_gender,**pp})
            st.success("✅ Profile updated successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

    history = user.get("history",[])
    if history:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📋 Analysis History")
        rows = [{"Date":h["date"][:10],"Time":h["date"][11:16],
                 "Score":f"{h['score']:.1f}","Status":score_meta(h["score"])[0]}
                for h in reversed(history[-20:])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SETTINGS  (no dark/light toggle — always dark)
# ══════════════════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown("#### 🤖 Model Information")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | Algorithm | Linear Regression (best R² on test set) |
    | R² Score | ~0.73 |
    | Training Samples | ~752 |
    | Features Used | 11 (8 raw + 3 engineered) |
    | Dataset | Synthetic — 1000 records |
    """)
    st.markdown("---")
    st.markdown("#### 🔐 Privacy & Data")
    st.info("All your data is stored locally in `data/users.json`. No data is sent to any external server.", icon="🔒")
    st.markdown("---")
    st.markdown("#### 🧩 Engineered Features Explained")
    st.markdown("""
    | Feature | Formula | Purpose |
    |---|---|---|
    | Dopamine Index | social_media + notifications + app_switches | Aggregates stimulus-seeking |
    | Focus Efficiency | focus_score / screen_time | Quality vs quantity of screen use |
    | Sleep Deficit | 8 − sleep_hours | Deviation from recommended sleep |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    page_auth()
else:
    render_sidebar()
    pg = st.session_state.page
    if   pg == "home":     page_home()
    elif pg == "step2":    page_step2()
    elif pg == "step3":    page_step3()
    elif pg == "step4":    page_step4()
    elif pg == "step5":    page_step5()
    elif pg == "result":   page_result()
    elif pg == "profile":  page_profile()
    elif pg == "settings": page_settings()
    else:                  page_home()
