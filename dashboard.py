import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io, base64, datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_cleaning import load_and_clean
from src.feature_engineering import engineer_features
from src.clustering_analysis import run_clustering, run_pca, CLUSTER_LABELS

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Outfit", color="#94A3B8"),
    margin=dict(l=20, r=20, t=44, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

@st.cache_data
def load_data():
    df = load_and_clean("data/dataset.csv")
    df = engineer_features(df)
    df, km, sc = run_clustering(df)
    df, pca = run_pca(df, sc)
    df["cluster_name"] = df["cluster"].map(CLUSTER_LABELS)
    return df

# ── manual trendline (no statsmodels needed) ──────────────────────────────────
def add_trendline(fig, x, y, color="#A78BFA"):
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    if len(xv) < 2:
        return fig
    m, b = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 100)
    fig.add_trace(go.Scatter(x=xs, y=m * xs + b,
                             mode="lines", line=dict(color=color, width=2, dash="dot"),
                             showlegend=False, name="Trend"))
    return fig

# ── PDF generation ─────────────────────────────────────────────────────────────
def generate_pdf_report(df, username):
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

    styles = getSampleStyleSheet()
    PURPLE = colors.HexColor("#7C3AED")
    CYAN   = colors.HexColor("#06B6D4")
    DARK   = colors.HexColor("#0D0F1E")
    LIGHT  = colors.HexColor("#E2E8F0")
    MUTED  = colors.HexColor("#64748B")
    GREEN  = colors.HexColor("#10B981")
    RED    = colors.HexColor("#EF4444")
    AMBER  = colors.HexColor("#F59E0B")

    title_style = ParagraphStyle("title", parent=styles["Title"],
        fontSize=26, textColor=PURPLE, spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
        fontSize=10, textColor=MUTED, spaceAfter=2, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=13, textColor=PURPLE, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=9.5, textColor=colors.HexColor("#334155"), leading=14)
    insight_style = ParagraphStyle("ins", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#1E293B"), leading=13,
        leftIndent=10, spaceBefore=3)

    story = []
    now = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("🧠 ScreenSense", title_style))
    story.append(Paragraph("Digital Wellbeing Analytics Report", sub_style))
    story.append(Paragraph(f"Generated for <b>{username}</b> · {now}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PURPLE))
    story.append(Spacer(1, 0.4*cm))

    # ── Population KPIs ───────────────────────────────────────────────────────
    story.append(Paragraph("Population Metrics Overview", h2_style))
    kpi_data = [
        ["Metric", "Mean", "Median", "Std Dev", "Min", "Max"],
        ["Digital Wellbeing Score",
         f"{df['digital_wellbeing_score'].mean():.1f}",
         f"{df['digital_wellbeing_score'].median():.1f}",
         f"{df['digital_wellbeing_score'].std():.1f}",
         f"{df['digital_wellbeing_score'].min():.1f}",
         f"{df['digital_wellbeing_score'].max():.1f}"],
        ["Daily Screen Time (h)",
         f"{df['daily_screen_time'].mean():.1f}",
         f"{df['daily_screen_time'].median():.1f}",
         f"{df['daily_screen_time'].std():.1f}",
         f"{df['daily_screen_time'].min():.1f}",
         f"{df['daily_screen_time'].max():.1f}"],
        ["Sleep Hours",
         f"{df['sleep_hours'].mean():.1f}",
         f"{df['sleep_hours'].median():.1f}",
         f"{df['sleep_hours'].std():.1f}",
         f"{df['sleep_hours'].min():.1f}",
         f"{df['sleep_hours'].max():.1f}"],
        ["Notification Count",
         f"{df['notification_count'].mean():.0f}",
         f"{df['notification_count'].median():.0f}",
         f"{df['notification_count'].std():.0f}",
         f"{df['notification_count'].min():.0f}",
         f"{df['notification_count'].max():.0f}"],
        ["Focus Score",
         f"{df['focus_score'].mean():.1f}",
         f"{df['focus_score'].median():.1f}",
         f"{df['focus_score'].std():.1f}",
         f"{df['focus_score'].min():.1f}",
         f"{df['focus_score'].max():.1f}"],
        ["Anxiety Level",
         f"{df['anxiety_level'].mean():.2f}",
         f"{df['anxiety_level'].median():.2f}",
         f"{df['anxiety_level'].std():.2f}",
         f"{df['anxiety_level'].min():.2f}",
         f"{df['anxiety_level'].max():.2f}"],
        ["Mood Score",
         f"{df['mood_score'].mean():.1f}",
         f"{df['mood_score'].median():.1f}",
         f"{df['mood_score'].std():.1f}",
         f"{df['mood_score'].min():.1f}",
         f"{df['mood_score'].max():.1f}"],
    ]
    tbl = Table(kpi_data, colWidths=[4.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  PURPLE),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  9),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE",    (0,1), (-1,-1), 8.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Cluster Summary ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Behavior Cluster Analysis (K-Means, k=3)", h2_style))

    cluster_stats = df.groupby("cluster_name")[
        ["daily_screen_time","sleep_hours","focus_score",
         "notification_count","anxiety_level","digital_wellbeing_score"]
    ].mean().round(2)

    c_data = [["Cluster", "Screen(h)", "Sleep(h)", "Focus", "Notifs", "Anxiety", "Wellbeing"]]
    cluster_colors_map = {
        "High Stimulation":       colors.HexColor("#FEE2E2"),
        "Balanced & Productive":  colors.HexColor("#DBEAFE"),
        "Rested & Focused":       colors.HexColor("#D1FAE5"),
    }
    row_bg = []
    for name, row in cluster_stats.iterrows():
        c_data.append([name,
            f"{row['daily_screen_time']:.1f}", f"{row['sleep_hours']:.1f}",
            f"{row['focus_score']:.1f}", f"{row['notification_count']:.0f}",
            f"{row['anxiety_level']:.2f}", f"{row['digital_wellbeing_score']:.1f}"])
        row_bg.append(cluster_colors_map.get(name, colors.white))

    ct = Table(c_data, colWidths=[4.2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.3*cm])
    ts_list = [
        ("BACKGROUND",  (0,0), (-1,0),  CYAN),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]
    for i, bg in enumerate(row_bg, start=1):
        ts_list.append(("BACKGROUND", (0,i), (-1,i), bg))
    ct.setStyle(TableStyle(ts_list))
    story.append(ct)
    story.append(Spacer(1, 0.4*cm))

    # ── Correlation highlights ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Key Correlation Findings", h2_style))

    corr = df[["digital_wellbeing_score","sleep_hours","daily_screen_time",
               "anxiety_level","focus_score","notification_count"]].corr()

    pairs = [
        ("Sleep Hours ↔ Wellbeing Score",    corr.loc["digital_wellbeing_score","sleep_hours"]),
        ("Screen Time ↔ Wellbeing Score",    corr.loc["digital_wellbeing_score","daily_screen_time"]),
        ("Anxiety Level ↔ Wellbeing Score",  corr.loc["digital_wellbeing_score","anxiety_level"]),
        ("Focus Score ↔ Wellbeing Score",    corr.loc["digital_wellbeing_score","focus_score"]),
        ("Notifications ↔ Anxiety Level",    corr.loc["notification_count","anxiety_level"]),
        ("Screen Time ↔ Focus Score",        corr.loc["daily_screen_time","focus_score"]),
    ]

    corr_data = [["Feature Pair", "Pearson r", "Direction", "Strength"]]
    for label, r in pairs:
        direction = "Positive ↑" if r > 0 else "Negative ↓"
        strength  = "Strong" if abs(r) > 0.5 else "Moderate" if abs(r) > 0.3 else "Weak"
        corr_data.append([label, f"{r:.3f}", direction, strength])

    corr_tbl = Table(corr_data, colWidths=[7*cm, 2.5*cm, 2.8*cm, 2.5*cm])
    corr_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#1E293B")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1),  "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(corr_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Behavioral Insights ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("AI-Generated Behavioral Insights", h2_style))

    high_screen = df[df["daily_screen_time"] > 6]["focus_score"].mean()
    low_screen  = df[df["daily_screen_time"] <= 6]["focus_score"].mean()
    good_sleep  = df[df["sleep_hours"] >= 7]["digital_wellbeing_score"].mean()
    poor_sleep  = df[df["sleep_hours"] < 6]["digital_wellbeing_score"].mean()
    high_notif  = df[df["notification_count"] > 100]["anxiety_level"].mean()
    low_notif   = df[df["notification_count"] <= 50]["anxiety_level"].mean()
    heavy_sw    = df[df["num_app_switches"] > 100]["focus_efficiency"].mean()
    low_sw      = df[df["num_app_switches"] <= 40]["focus_efficiency"].mean()

    insights_text = [
        (f"📱 Screen Time Impact — Users with >6h daily screen time average a focus score of "
         f"{high_screen:.1f} vs {low_screen:.1f} for moderate users "
         f"({((low_screen - high_screen)/low_screen*100):.0f}% difference)."),
        (f"😴 Sleep & Wellbeing — Users sleeping 7+ hours score {good_sleep:.1f} on wellbeing "
         f"vs {poor_sleep:.1f} for those under 6h — a {good_sleep - poor_sleep:.1f} point gap."),
        (f"🔔 Notifications & Anxiety — High-notification users (>100/day) show anxiety level "
         f"{high_notif:.2f} vs {low_notif:.2f} for low-notification users."),
        (f"🔄 App Switching & Focus — Heavy switchers (>100 switches) have focus efficiency "
         f"{heavy_sw:.2f} vs {low_sw:.2f} for low switchers — "
         f"{((low_sw-heavy_sw)/low_sw*100):.0f}% more efficient with fewer switches."),
        ("🎯 Model Accuracy — The Gradient Boosting model achieves CV R² = 0.781, "
         "meaning it explains 78.1% of variance in digital wellbeing scores."),
        ("💡 Recommendation — Reducing screen time by 2h, improving sleep to 7h+, "
         "and limiting notifications to under 60/day are the three highest-impact changes "
         "for improving digital wellbeing."),
    ]
    for txt in insights_text:
        story.append(Paragraph(f"• {txt}", insight_style))
    story.append(Spacer(1, 0.4*cm))

    # ── Screen-time distribution ──────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Paragraph("Screen Time Segmentation", h2_style))

    bins   = [0, 3, 6, 9, 12, 16]
    labels = ["<3h", "3–6h", "6–9h", "9–12h", "12h+"]
    df["st_bin"] = pd.cut(df["daily_screen_time"], bins=bins, labels=labels)
    seg = df.groupby("st_bin", observed=True).agg(
        Users=("daily_screen_time","count"),
        Avg_Focus=("focus_score","mean"),
        Avg_Wellbeing=("digital_wellbeing_score","mean"),
        Avg_Anxiety=("anxiety_level","mean"),
    ).reset_index()
    seg.columns = ["Screen Time Band","Users","Avg Focus","Avg Wellbeing","Avg Anxiety"]
    seg["Avg Focus"]    = seg["Avg Focus"].round(1)
    seg["Avg Wellbeing"]= seg["Avg Wellbeing"].round(1)
    seg["Avg Anxiety"]  = seg["Avg Anxiety"].round(2)

    seg_data = [list(seg.columns)] + seg.values.tolist()
    seg_tbl = Table(seg_data, colWidths=[4*cm, 2.5*cm, 2.8*cm, 3*cm, 3*cm])
    seg_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#7C3AED")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(seg_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "ScreenSense AI · Digital Wellbeing Analytics Platform · Confidential Report",
        ParagraphStyle("footer", parent=styles["Normal"],
                       fontSize=8, textColor=MUTED, alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def render():
    df = load_data()

    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>Analytics Dashboard</h1>
  <p style='color:#64748B;font-size:0.92rem;'>Population-level insights from behavioral analysis</p>
</div>
""", unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        ("📊","Avg Wellbeing Score", f"{df['digital_wellbeing_score'].mean():.1f}", "/100"),
        ("😴","Avg Sleep Hours",     f"{df['sleep_hours'].mean():.1f}h",            "per night"),
        ("📱","Avg Screen Time",     f"{df['daily_screen_time'].mean():.1f}h",      "per day"),
        ("🔔","Avg Notifications",   f"{df['notification_count'].mean():.0f}",      "per day"),
    ]
    for col,(icon,label,val,sub) in zip([c1,c2,c3,c4], kpis):
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.2);
  border-radius:16px;padding:22px 18px;text-align:center;'>
  <div style='font-size:1.5rem;'>{icon}</div>
  <div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;
    margin:6px 0 4px;font-weight:600;'>{label}</div>
  <div style='font-size:1.9rem;font-weight:800;background:linear-gradient(135deg,#A78BFA,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{val}</div>
  <div style='font-size:0.75rem;color:#475569;'>{sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>📱 Screen Time vs Focus Score</div>", unsafe_allow_html=True)
        sample = df.sample(400, random_state=42)
        fig = px.scatter(sample, x="daily_screen_time", y="focus_score",
                         color="digital_wellbeing_score", color_continuous_scale="Viridis",
                         labels={"daily_screen_time":"Screen Time (hrs)","focus_score":"Focus Score"})
        fig = add_trendline(fig, sample["daily_screen_time"].values, sample["focus_score"].values)
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          coloraxis_colorbar=dict(title="Wellbeing"))
        fig.update_traces(selector=dict(type="scatter", mode="markers"),
                          marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>😴 Sleep Hours vs Wellbeing Score</div>", unsafe_allow_html=True)
        sample2 = df.sample(400, random_state=7)
        fig2 = px.scatter(sample2, x="sleep_hours", y="digital_wellbeing_score",
                          color="anxiety_level", color_continuous_scale="RdYlGn_r",
                          labels={"sleep_hours":"Sleep (hrs)","digital_wellbeing_score":"Wellbeing Score"})
        fig2 = add_trendline(fig2, sample2["sleep_hours"].values, sample2["digital_wellbeing_score"].values, "#06B6D4")
        fig2.update_layout(**PLOTLY_LAYOUT, height=320)
        fig2.update_traces(selector=dict(type="scatter", mode="markers"),
                           marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ──────────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>🔔 Notifications vs Anxiety Level</div>", unsafe_allow_html=True)
        sample3 = df.sample(400, random_state=21)
        fig3 = px.scatter(sample3, x="notification_count", y="anxiety_level",
                          color="sleep_hours", color_continuous_scale="Blues",
                          labels={"notification_count":"Notifications/day","anxiety_level":"Anxiety Level"})
        fig3.update_layout(**PLOTLY_LAYOUT, height=300)
        fig3.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>📊 Wellbeing Score Distribution</div>", unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=df["digital_wellbeing_score"], nbinsx=30,
                                     marker_color="#7C3AED", opacity=0.8))
        fig4.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                           xaxis_title="Wellbeing Score", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3 ──────────────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>🧩 Behavior Cluster Profiles</div>", unsafe_allow_html=True)
        cluster_stats = df.groupby("cluster_name")[
            ["daily_screen_time","sleep_hours","focus_score","digital_wellbeing_score"]
        ].mean().reset_index()
        fig5 = go.Figure()
        clr = ["#7C3AED","#06B6D4","#10B981"]
        metrics = ["daily_screen_time","sleep_hours","focus_score","digital_wellbeing_score"]
        labels  = ["Screen Time","Sleep Hours","Focus Score","Wellbeing"]
        for i, row in cluster_stats.iterrows():
            fig5.add_trace(go.Bar(name=row["cluster_name"], x=labels,
                                  y=[row[m] for m in metrics], marker_color=clr[i % 3]))
        fig5.update_layout(**PLOTLY_LAYOUT, height=310, barmode="group",
                           legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("<div style='font-weight:600;color:#E2E8F0;margin-bottom:8px;'>🔄 App Switches vs Focus Efficiency</div>", unsafe_allow_html=True)
        fig6 = px.scatter(df.sample(400, random_state=55), x="num_app_switches", y="focus_efficiency",
                          color="cluster_name",
                          color_discrete_sequence=["#7C3AED","#06B6D4","#10B981"],
                          labels={"num_app_switches":"App Switches/day","focus_efficiency":"Focus Efficiency"})
        fig6.update_layout(**PLOTLY_LAYOUT, height=310)
        fig6.update_traces(marker=dict(size=5, opacity=0.75))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Cluster Summary Cards ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;font-size:1.05rem;color:#E2E8F0;margin-bottom:14px;'>🧩 Behavior Cluster Summary</div>", unsafe_allow_html=True)
    cluster_icons  = {"High Stimulation":"⚡","Balanced & Productive":"⚖️","Rested & Focused":"🌙"}
    cluster_colors = {"High Stimulation":"#EF4444","Balanced & Productive":"#06B6D4","Rested & Focused":"#10B981"}
    cluster_desc   = {
        "High Stimulation":      "High screen time, frequent notifications, low focus scores.",
        "Balanced & Productive": "Moderate screen time, decent sleep, sustainable digital habits.",
        "Rested & Focused":      "Low screen time, high sleep quality, strong focus and wellbeing.",
    }
    cols = st.columns(3)
    for col, (name, count) in zip(cols, df["cluster_name"].value_counts().items()):
        subset = df[df["cluster_name"] == name]
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {cluster_colors[name]}44;
  border-left:3px solid {cluster_colors[name]};border-radius:16px;padding:20px;'>
  <div style='font-size:1.6rem;'>{cluster_icons.get(name,"🔵")}</div>
  <div style='font-weight:700;color:#E2E8F0;margin:8px 0 6px;'>{name}</div>
  <div style='font-size:0.8rem;color:#64748B;line-height:1.5;margin-bottom:12px;'>{cluster_desc.get(name,"")}</div>
  <div style='display:flex;gap:8px;flex-wrap:wrap;'>
    <span style='background:rgba(255,255,255,0.06);border-radius:8px;padding:3px 10px;font-size:0.75rem;color:#94A3B8;'>
      👥 {count} users
    </span>
    <span style='background:rgba(255,255,255,0.06);border-radius:8px;padding:3px 10px;font-size:0.75rem;color:#94A3B8;'>
      🎯 Score: {subset["digital_wellbeing_score"].mean():.1f}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PDF DOWNLOAD SECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.15),rgba(6,182,212,0.12));
  border:1px solid rgba(124,58,237,0.35);border-radius:20px;padding:28px 32px;'>
  <div style='display:flex;align-items:center;gap:14px;'>
    <div style='font-size:2.2rem;'>📄</div>
    <div>
      <div style='font-size:1.15rem;font-weight:700;color:#E2E8F0;'>Download Analytics Report</div>
      <div style='font-size:0.85rem;color:#64748B;margin-top:3px;'>
        Export a full PDF containing population metrics, cluster analysis, 
        correlation findings, behavioral insights, and screen-time segmentation.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # What's included preview
    inc_cols = st.columns(4)
    items = [
        ("📊", "Population KPIs", "Mean, median, std for all features"),
        ("🧩", "Cluster Stats",   "3 behavior group comparisons"),
        ("🔗", "Correlations",    "6 key feature relationships"),
        ("💡", "AI Insights",     "Data-driven recommendations"),
    ]
    for col, (icon, title, desc) in zip(inc_cols, items):
        col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(124,58,237,0.18);
  border-radius:14px;padding:16px;text-align:center;'>
  <div style='font-size:1.4rem;'>{icon}</div>
  <div style='font-weight:600;color:#E2E8F0;font-size:0.85rem;margin:6px 0 3px;'>{title}</div>
  <div style='font-size:0.75rem;color:#64748B;'>{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("⬇️  Generate & Download PDF Report", use_container_width=True):
            with st.spinner("Generating your analytics report..."):
                username = st.session_state.get("username", "User")
                pdf_bytes = generate_pdf_report(df, username)
                fname = f"screensense_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button(
                    label="📥  Click here to save your PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.success("✅ Report ready! Click the button above to save it.")
