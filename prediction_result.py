import streamlit as st
import numpy as np
import joblib, io, datetime
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import FEATURE_COLS
from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(family="Outfit", color="#94A3B8"),
                     margin=dict(l=0, r=0, t=20, b=0))

@st.cache_resource
def load_model():
    return joblib.load("models/wellbeing_model.pkl")

@st.cache_data
def get_population_stats():
    df = load_and_clean("data/dataset.csv")
    df = engineer_features(df)
    return df

def build_input(data):
    screen   = data["daily_screen_time"]
    switches = data["num_app_switches"]
    sleep    = data["sleep_hours"]
    notifs   = data["notification_count"]
    social   = data["social_media_time_min"]
    focus    = data["focus_score"]
    mood     = data["mood_score"]
    anxiety  = min(10, (notifs*0.08 + social*0.03 + screen*1.2) / 10)
    dopamine = social + notifs + switches
    focus_eff= focus / (screen + 0.01)
    sleep_def= max(0, 8 - sleep)
    return pd.DataFrame([[screen, switches, sleep, notifs, social, focus, mood,
                           anxiety, dopamine, focus_eff, sleep_def]], columns=FEATURE_COLS)

def get_score_tier(score):
    if score >= 70: return "Excellent",  "#10B981", "🌟"
    if score >= 55: return "Good",        "#06B6D4", "✅"
    if score >= 40: return "Fair",        "#F59E0B", "⚠️"
    return "Needs Attention", "#EF4444", "🚨"

def generate_insights(data, score):
    msgs = []
    if data["daily_screen_time"] > 8:
        msgs.append(("📱","High screen time detected",
                     "Reducing screen time by 2+ hours could meaningfully improve your focus and wellbeing.","#EF4444"))
    if data["sleep_hours"] < 6.5:
        msgs.append(("😴","Sleep deficit detected",
                     f"You're sleeping {data['sleep_hours']}h. Targeting 7–8 hours could add 10–15 points to your wellbeing.","#F59E0B"))
    if data["notification_count"] > 100:
        msgs.append(("🔔","High notification load",
                     "Enable Do Not Disturb and batch-check alerts to lower anxiety significantly.","#F59E0B"))
    if data["social_media_time_min"] > 120:
        msgs.append(("📲","Heavy social media usage",
                     f"{data['social_media_time_min']:.0f} min/day correlates with lower mood. Try a 30-day usage reduction.","#EF4444"))
    if data["num_app_switches"] > 100:
        msgs.append(("🔄","Frequent app switching",
                     "Try time-blocking with single-app focus sessions to rebuild deep concentration.","#F59E0B"))
    if data["focus_score"] < 50:
        msgs.append(("🎯","Low focus score",
                     "Pomodoro technique, screen limits and notification batching may help restore focus.","#A78BFA"))
    if data["sleep_hours"] >= 7.5 and score >= 55:
        msgs.append(("🌙","Great sleep hygiene",
                     "Your sleep pattern positively contributes to wellbeing. Keep it up!","#10B981"))
    if not msgs:
        msgs.append(("🎉","Well-balanced digital habits",
                     "Your digital behavior patterns indicate healthy wellbeing. Maintain these habits!","#10B981"))
    return msgs

def generate_personal_pdf(data, score, username, pop_df):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles   = getSampleStyleSheet()
    PURPLE   = colors.HexColor("#7C3AED")
    CYAN     = colors.HexColor("#06B6D4")
    GREEN    = colors.HexColor("#10B981")
    RED      = colors.HexColor("#EF4444")
    AMBER    = colors.HexColor("#F59E0B")
    MUTED    = colors.HexColor("#64748B")
    tier, tier_color_hex, _ = get_score_tier(score)
    tier_color = colors.HexColor(tier_color_hex)

    title_style = ParagraphStyle("t",  parent=styles["Title"],  fontSize=24, textColor=PURPLE,
                                  spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("s",  parent=styles["Normal"], fontSize=9.5, textColor=MUTED,
                                  spaceAfter=2, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, textColor=PURPLE,
                                  spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("b",  parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#334155"), leading=13)
    ins_style   = ParagraphStyle("i",  parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#1E293B"),
                                  leading=13, leftIndent=10, spaceBefore=3)

    story = []
    now = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    # Header
    story.append(Paragraph("🧠 ScreenSense", title_style))
    story.append(Paragraph("Personal Digital Wellbeing Report", sub_style))
    story.append(Paragraph(f"Prepared for <b>{username}</b>  ·  {now}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE))
    story.append(Spacer(1, 0.4*cm))

    # Score banner
    story.append(Paragraph("Your Wellbeing Score", h2_style))
    score_data = [
        ["Your Score", "Status", "Confidence"],
        [f"{score:.1f} / 100", tier, "~78%"]
    ]
    st_tbl = Table(score_data, colWidths=[5.5*cm, 5.5*cm, 4.5*cm])
    st_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), PURPLE),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 11),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND",   (0,1), (-1,1), colors.HexColor("#F8FAFC")),
        ("FONTNAME",     (0,1), (-1,1), "Helvetica-Bold"),
        ("TEXTCOLOR",    (0,1), (0,1),  tier_color),
        ("TEXTCOLOR",    (1,1), (1,1),  tier_color),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
    ]))
    story.append(st_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Your inputs vs population averages
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Your Inputs vs. Population Averages", h2_style))
    story.append(Paragraph(
        "The table below shows your reported values alongside population averages "
        "so you can see where you stand and identify areas for improvement.",
        body_style))
    story.append(Spacer(1, 0.2*cm))

    anxiety_est = min(10, (data["notification_count"]*0.08 +
                           data["social_media_time_min"]*0.03 +
                           data["daily_screen_time"]*1.2) / 10)

    rows_def = [
        ("Daily Screen Time (h)",    data["daily_screen_time"],         pop_df["daily_screen_time"].mean(),     3, 6,   False),
        ("Sleep Hours",              data["sleep_hours"],                pop_df["sleep_hours"].mean(),           7, 8,   True),
        ("Notification Count",       data["notification_count"],        pop_df["notification_count"].mean(),    50,100,  False),
        ("App Switches / day",       data["num_app_switches"],          pop_df["num_app_switches"].mean(),      40,80,   False),
        ("Social Media (min/day)",   data["social_media_time_min"],     pop_df["social_media_time_min"].mean(), 60,120,  False),
        ("Focus Score",              data["focus_score"],               pop_df["focus_score"].mean(),           60,80,   True),
        ("Mood Score",               data["mood_score"],                pop_df["mood_score"].mean(),            60,80,   True),
        ("Estimated Anxiety Level",  round(anxiety_est,2),             pop_df["anxiety_level"].mean(),         3, 6,    False),
    ]

    comp_data = [["Metric", "Your Value", "Population Avg", "Assessment"]]
    for label, your_val, pop_avg, good_thresh, ok_thresh, higher_better in rows_def:
        if higher_better:
            status = "✓ Good" if your_val >= good_thresh else ("~ OK" if your_val >= ok_thresh else "↑ Improve")
        else:
            status = "✓ Good" if your_val <= good_thresh else ("~ OK" if your_val <= ok_thresh else "↓ Reduce")
        comp_data.append([label,
                          f"{your_val:.1f}" if isinstance(your_val, float) else str(your_val),
                          f"{pop_avg:.1f}",
                          status])

    comp_tbl = Table(comp_data, colWidths=[5.5*cm, 2.8*cm, 3.2*cm, 3.8*cm])
    comp_tbl_style = [
        ("BACKGROUND",   (0,0), (-1,0), CYAN),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("ALIGN",        (0,1), (0,-1),  "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]
    # colour the assessment column
    for i, (_, _, _, g, ok_t, hb) in enumerate(rows_def, start=1):
        your_v = rows_def[i-1][1]
        if hb:
            c = GREEN if your_v >= g else (AMBER if your_v >= ok_t else RED)
        else:
            c = GREEN if your_v <= g else (AMBER if your_v <= ok_t else RED)
        comp_tbl_style.append(("TEXTCOLOR", (3,i), (3,i), c))
        comp_tbl_style.append(("FONTNAME",  (3,i), (3,i), "Helvetica-Bold"))
    comp_tbl.setStyle(TableStyle(comp_tbl_style))
    story.append(comp_tbl)
    story.append(Spacer(1, 0.4*cm))

    # AI Insights
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Personalised AI Insights", h2_style))
    insights_list = generate_insights(data, score)
    for icon, title, desc, _ in insights_list:
        story.append(Paragraph(f"<b>{icon} {title}</b>", ins_style))
        story.append(Paragraph(desc, ins_style))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.3*cm))

    # Affirmation
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Your Daily Affirmation", h2_style))
    aff = ('"Every small step toward healthier digital habits compounds into a better, '
           'more focused, and more joyful version of you. You have the power to change."')
    story.append(Paragraph(aff, ParagraphStyle("aff", parent=styles["Normal"],
        fontSize=10, textColor=PURPLE, leading=15, leftIndent=12,
        borderPad=8, borderColor=PURPLE, borderWidth=1, borderRadius=4)))

    # Footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "ScreenSense AI · Your Digital Wellbeing Partner · Confidential Personal Report",
        ParagraphStyle("ft", parent=styles["Normal"], fontSize=7.5, textColor=MUTED,
                       alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()

def save_to_history(data, score):
    email = st.session_state.get("email", "guest")
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = {}
    if email not in st.session_state.prediction_history:
        st.session_state.prediction_history[email] = []
    tier, _, _ = get_score_tier(score)
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score": round(score, 1),
        "tier":  tier,
        **{k: round(v, 2) if isinstance(v, float) else v for k, v in data.items()}
    }
    st.session_state.prediction_history[email].append(record)

def render():
    data = st.session_state.pred_data
    if len(data) < 7:
        st.error("Incomplete input. Please restart the prediction wizard.")
        if st.button("← Start Over"):
            st.session_state.pred_step = 1
            st.session_state.pred_data = {}
            st.rerun()
        return

    model   = load_model()
    pop_df  = get_population_stats()
    X       = build_input(data)
    raw     = float(model.predict(X)[0])
    score   = np.clip(raw, 10, 100)
    tier, tier_color, tier_icon = get_score_tier(score)

    # Auto-save to history (only once per prediction)
    if not st.session_state.get("_last_saved_score") or st.session_state._last_saved_score != score:
        save_to_history(data, score)
        st.session_state._last_saved_score = score

    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>
    Your Wellbeing Prediction
  </h1>
  <p style='color:#64748B;font-size:0.9rem;'>AI-powered analysis of your digital behavior</p>
</div>
""", unsafe_allow_html=True)

    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            domain={'x':[0,1],'y':[0,1]},
            number={'suffix':"/100",'font':{'size':44,'family':'Outfit','color':'#E2E8F0'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#475569','tickfont':{'color':'#64748B'}},
                'bar':{'color':tier_color,'thickness':0.25},
                'bgcolor':"rgba(255,255,255,0.04)",'bordercolor':"rgba(255,255,255,0.1)",
                'steps':[
                    {'range':[0,40],'color':'rgba(239,68,68,0.15)'},
                    {'range':[40,55],'color':'rgba(245,158,11,0.15)'},
                    {'range':[55,70],'color':'rgba(6,182,212,0.15)'},
                    {'range':[70,100],'color':'rgba(16,185,129,0.15)'},
                ],
                'threshold':{'line':{'color':tier_color,'width':4},'thickness':0.75,'value':score}
            }
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {tier_color}44;
  border-radius:20px;padding:28px;'>
  <div style='font-size:2.5rem;'>{tier_icon}</div>
  <div style='font-size:1.5rem;font-weight:800;color:{tier_color};'>{tier}</div>
  <div style='color:#94A3B8;font-size:0.9rem;margin:6px 0 20px;'>Digital Wellbeing Status</div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
    <div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;text-align:center;'>
      <div style='font-size:1.6rem;font-weight:800;color:{tier_color};'>{score:.1f}</div>
      <div style='font-size:0.72rem;color:#64748B;'>Your Score</div>
    </div>
    <div style='background:rgba(255,255,255,0.04);border-radius:12px;padding:14px;text-align:center;'>
      <div style='font-size:1.6rem;font-weight:800;color:#A78BFA;'>~78%</div>
      <div style='font-size:0.72rem;color:#64748B;'>Model Confidence</div>
    </div>
  </div>
  <div style='margin-top:14px;background:rgba(255,255,255,0.04);border-radius:12px;padding:12px;'>
    <div style='font-size:0.72rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Your Inputs</div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:0.78rem;color:#94A3B8;'>
      <span>📱 Screen: {data['daily_screen_time']}h</span>
      <span>😴 Sleep: {data['sleep_hours']}h</span>
      <span>🔔 Notifs: {data['notification_count']}</span>
      <span>📲 Social: {data['social_media_time_min']:.0f}min</span>
      <span>🎯 Focus: {data['focus_score']}</span>
      <span>😊 Mood: {data['mood_score']}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Population comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;font-size:1.05rem;color:#E2E8F0;margin-bottom:10px;'>📊 How You Compare</div>", unsafe_allow_html=True)

    anxiety_est = min(10, (data["notification_count"]*0.08 +
                            data["social_media_time_min"]*0.03 +
                            data["daily_screen_time"]*1.2) / 10)
    compare_rows = [
        ("📱 Screen Time",     f"{data['daily_screen_time']}h",         f"{pop_df['daily_screen_time'].mean():.1f}h",         data["daily_screen_time"],        pop_df["daily_screen_time"].mean(),        False),
        ("😴 Sleep",           f"{data['sleep_hours']}h",               f"{pop_df['sleep_hours'].mean():.1f}h",               data["sleep_hours"],               pop_df["sleep_hours"].mean(),               True),
        ("🔔 Notifications",   f"{data['notification_count']}",         f"{pop_df['notification_count'].mean():.0f}",         data["notification_count"],       pop_df["notification_count"].mean(),        False),
        ("🎯 Focus Score",     f"{data['focus_score']}",                f"{pop_df['focus_score'].mean():.1f}",                data["focus_score"],               pop_df["focus_score"].mean(),               True),
        ("😊 Mood Score",      f"{data['mood_score']}",                 f"{pop_df['mood_score'].mean():.1f}",                 data["mood_score"],                pop_df["mood_score"].mean(),                True),
        ("😰 Anxiety (est.)",  f"{anxiety_est:.1f}",                    f"{pop_df['anxiety_level'].mean():.1f}",              anxiety_est,                       pop_df["anxiety_level"].mean(),             False),
    ]

    cols = st.columns(len(compare_rows))
    for col, (label, your_v, pop_v, yv_raw, pop_raw, hb) in zip(cols, compare_rows):
        better = (yv_raw >= pop_raw) if hb else (yv_raw <= pop_raw)
        indicator_color = "#10B981" if better else "#EF4444"
        arrow = "↑" if better else "↓"
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
  border-radius:14px;padding:14px 10px;text-align:center;'>
  <div style='font-size:0.75rem;color:#64748B;margin-bottom:6px;font-weight:600;'>{label}</div>
  <div style='font-size:1.3rem;font-weight:800;color:{indicator_color};'>{your_v}</div>
  <div style='font-size:0.7rem;color:#475569;margin-top:3px;'>avg {pop_v}</div>
  <div style='font-size:0.75rem;color:{indicator_color};font-weight:700;margin-top:4px;'>{arrow} {"Better" if better else "Room to grow"}</div>
</div>
""", unsafe_allow_html=True)

    # AI Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;font-size:1.05rem;color:#E2E8F0;margin-bottom:10px;'>🤖 AI Insights & Recommendations</div>", unsafe_allow_html=True)
    insights_list = generate_insights(data, score)
    ins_cols = st.columns(min(len(insights_list), 2))
    for i, (icon, title, desc, color) in enumerate(insights_list):
        ins_cols[i % 2].markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {color}33;border-left:3px solid {color};
  border-radius:14px;padding:18px;margin-bottom:12px;'>
  <div style='font-size:1.2rem;margin-bottom:6px;'>{icon}</div>
  <div style='font-weight:600;color:#E2E8F0;font-size:0.92rem;margin-bottom:4px;'>{title}</div>
  <div style='color:#64748B;font-size:0.82rem;line-height:1.5;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

    # Radar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;font-size:1.05rem;color:#E2E8F0;margin-bottom:10px;'>📡 Behavioral Radar Profile</div>", unsafe_allow_html=True)
    categories = ['Screen Time','App Switches','Sleep','Notifications','Social Media','Focus','Mood']
    raw_vals = [data['daily_screen_time']/16, data['num_app_switches']/200,
                data['sleep_hours']/10, data['notification_count']/300,
                data['social_media_time_min']/360, data['focus_score']/100, data['mood_score']/100]
    vals = raw_vals + [raw_vals[0]]
    cats = categories + [categories[0]]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats, fill='toself',
                                    fillcolor='rgba(124,58,237,0.2)',
                                    line=dict(color='#7C3AED', width=2)))
    fig_r.update_layout(**PLOTLY_LAYOUT, height=320,
                        polar=dict(bgcolor="rgba(255,255,255,0.03)",
                                   radialaxis=dict(visible=True,range=[0,1],gridcolor="rgba(255,255,255,0.1)"),
                                   angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")))
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.plotly_chart(fig_r, use_container_width=True)

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(6,182,212,0.12));
  border:1px solid rgba(124,58,237,0.35);border-radius:20px;padding:24px 28px;'>
  <div style='font-size:1.1rem;font-weight:700;color:#E2E8F0;margin-bottom:6px;'>📄 Download Your Personal Report</div>
  <div style='font-size:0.85rem;color:#64748B;'>
    Get a full PDF with your wellbeing score, personal inputs vs. population averages, 
    AI insights, and an affirmation — all tailored to <em>your</em> data.
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("⬇️  Generate My Personal PDF Report", use_container_width=True):
            with st.spinner("Building your personalised report..."):
                username  = st.session_state.get("username","User")
                pdf_bytes = generate_personal_pdf(data, score, username, pop_df)
                fname = f"screensense_{username.replace(' ','_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="📥  Save My Wellbeing Report",
                data=pdf_bytes, file_name=fname,
                mime="application/pdf", use_container_width=True)
            st.success("✅ Report ready!")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🔄 Predict Again", use_container_width=True):
            st.session_state.pred_step = 1
            st.session_state.pred_data = {}
            st.session_state._last_saved_score = None
            st.rerun()
    with col_b:
        if st.button("📜 My History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()
    with col_c:
        if st.button("📊 View Analytics →", use_container_width=True):
            st.session_state.page = "Analytics"
            st.rerun()
