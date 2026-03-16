import streamlit as st
import random
import datetime

AFFIRMATIONS = {
    "💪 Willpower": [
        "Every time you choose your long-term wellbeing over a quick scroll, your willpower grows stronger.",
        "You are not controlled by your phone — you hold the power to put it down at any moment.",
        "Discipline is choosing what you want most over what you want right now.",
        "Your ability to delay gratification is one of the most powerful tools you possess.",
        "Each small act of self-control rewires your brain for greater strength tomorrow.",
        "The hardest step is always the first. Once you start, momentum carries you forward.",
        "Resistance builds resilience. Every urge you overcome makes the next one easier.",
    ],
    "🧠 Mental Clarity": [
        "A calm, uncluttered mind is your greatest competitive advantage in a noisy world.",
        "When you protect your attention, you protect your most valuable resource — your thoughts.",
        "Deep focus is a superpower. Guard it fiercely by reducing digital noise.",
        "Mental clarity is not a luxury — it's the foundation of every great decision you make.",
        "Your brain is most powerful when it has space to breathe. Give it that space today.",
        "Single-tasking is not inefficiency — it's the fastest path to meaningful progress.",
        "A rested mind solves in minutes what an exhausted mind struggles with for hours.",
    ],
    "😴 Sleep & Recovery": [
        "Sleep is not wasted time — it is the secret weapon of every high performer.",
        "Tonight's sleep is tomorrow's energy, creativity, and emotional resilience.",
        "Protecting your sleep is one of the most loving things you can do for yourself.",
        "Your brain detoxifies, heals, and consolidates memories while you sleep. Honour that process.",
        "Rest is not laziness — it is the foundation upon which all great effort is built.",
        "Every hour of quality sleep is an investment that pays dividends all day long.",
        "The best version of you shows up after a full night of restorative rest.",
    ],
    "🎯 Focus & Concentration": [
        "Where your attention goes, your life grows. Choose wisely what you focus on.",
        "Deep work is the path to mastery. Every distraction you resist is a skill sharpened.",
        "You don't need more hours in the day — you need fewer distractions in your hours.",
        "Concentration is a muscle. The more you train it, the stronger it becomes.",
        "The ability to focus is the difference between dreaming about success and achieving it.",
        "One meaningful hour of deep focus is worth more than five hours of scattered activity.",
        "Turn off the noise, tune in to your purpose, and watch what you're capable of.",
    ],
    "❤️ Confidence & Self-Worth": [
        "You are more than your screen time. Your worth is not measured by likes or notifications.",
        "Confidence grows every time you honour the commitments you make to yourself.",
        "You have overcome every difficult day you've ever faced. Today is no different.",
        "Your value is intrinsic — not defined by productivity metrics or social validation.",
        "Believe in the person you are becoming. Progress, not perfection, is the goal.",
        "You deserve a life that feels as good as your best moments offline.",
        "Every positive change, no matter how small, is proof that you are capable of growth.",
    ],
    "🌱 Growth & Progress": [
        "Progress is not always linear. Even a flat day is a day you didn't go backwards.",
        "You are not the same person you were last month. Growth is happening, even when unseen.",
        "Small consistent improvements compound into extraordinary transformations over time.",
        "The gap between where you are and where you want to be is closed one habit at a time.",
        "Your future self is watching your choices today — make them proud.",
        "Don't compare your chapter one to someone else's chapter twenty.",
        "Every step forward, however small, deserves acknowledgment and celebration.",
    ],
    "🌊 Balance & Wellbeing": [
        "Balance is not found — it is created, one intentional choice at a time.",
        "A life well-lived is not measured by how much you consumed, but by how deeply you connected.",
        "Your wellbeing is not selfish — it is the foundation from which you serve others best.",
        "Technology is a tool. You are the craftsman. Use it with intention.",
        "The richest experiences in life are rarely on a screen. Seek them out today.",
        "Digital detox moments are not escapes from life — they are returns to it.",
        "Invest in your inner world as much as you curate your digital one.",
    ],
}

def get_daily_affirmation():
    """Returns a deterministic daily affirmation based on date."""
    day_idx = datetime.date.today().toordinal()
    all_affs = [(cat, aff) for cat, affs in AFFIRMATIONS.items() for aff in affs]
    return all_affs[day_idx % len(all_affs)]

def render():
    st.markdown("""
<div style='padding:8px 0 28px;'>
  <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#E2E8F0,#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 4px;'>Daily Affirmations</h1>
  <p style='color:#64748B;font-size:0.92rem;'>Words that build confidence, clarity, and wellbeing — one day at a time</p>
</div>
""", unsafe_allow_html=True)

    # Daily affirmation hero card
    daily_cat, daily_aff = get_daily_affirmation()
    username = st.session_state.get("username", "Friend")
    today = datetime.date.today().strftime("%A, %B %d")

    st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(6,182,212,0.15));
  border:1px solid rgba(124,58,237,0.4);border-radius:24px;padding:40px;text-align:center;
  margin-bottom:28px;'>
  <div style='font-size:0.75rem;color:#7C3AED;font-weight:700;text-transform:uppercase;
    letter-spacing:2px;margin-bottom:12px;'>✨ Your Affirmation for Today · {today}</div>
  <div style='font-size:1.35rem;font-weight:600;color:#E2E8F0;line-height:1.7;
    max-width:640px;margin:0 auto 16px;'>
    "{daily_aff}"
  </div>
  <div style='font-size:0.85rem;color:#64748B;'>— {daily_cat} · For you, {username} 💜</div>
</div>
""", unsafe_allow_html=True)

    # Category filter
    st.markdown("<div style='font-weight:700;color:#E2E8F0;margin-bottom:12px;font-size:1.05rem;'>🗂️ Explore by Category</div>", unsafe_allow_html=True)

    selected_cat = st.selectbox("Choose a category", list(AFFIRMATIONS.keys()),
                                 label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    affs = AFFIRMATIONS[selected_cat]
    cat_colors = {
        "💪 Willpower":           "#7C3AED",
        "🧠 Mental Clarity":      "#06B6D4",
        "😴 Sleep & Recovery":    "#10B981",
        "🎯 Focus & Concentration":"#F59E0B",
        "❤️ Confidence & Self-Worth": "#EC4899",
        "🌱 Growth & Progress":   "#A78BFA",
        "🌊 Balance & Wellbeing": "#14B8A6",
    }
    color = cat_colors.get(selected_cat, "#7C3AED")

    for i in range(0, len(affs), 2):
        cols = st.columns(2)
        for col, aff in zip(cols, affs[i:i+2]):
            col.markdown(f"""
<div style='background:rgba(255,255,255,0.04);border:1px solid {color}33;
  border-top:3px solid {color};border-radius:16px;padding:22px;margin-bottom:14px;
  min-height:120px;display:flex;align-items:center;'>
  <div style='font-size:0.95rem;color:#CBD5E1;line-height:1.7;font-style:italic;'>
    "{aff}"
  </div>
</div>
""", unsafe_allow_html=True)

    # Shuffle / random pick
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#E2E8F0;margin-bottom:10px;'>🎲 Need a Random Boost?</div>", unsafe_allow_html=True)

    if "rand_aff" not in st.session_state:
        st.session_state.rand_aff = None

    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("✨ Give Me a Random Affirmation", use_container_width=True):
            all_flat = [(cat, aff) for cat, affs_list in AFFIRMATIONS.items() for aff in affs_list]
            cat, aff = random.choice(all_flat)
            st.session_state.rand_aff = (cat, aff)

    if st.session_state.rand_aff:
        rcat, raff = st.session_state.rand_aff
        rcol = cat_colors.get(rcat, "#7C3AED")
        st.markdown(f"""
<div style='background:rgba(255,255,255,0.05);border:2px solid {rcol}55;
  border-radius:18px;padding:28px;text-align:center;margin-top:12px;'>
  <div style='font-size:0.72rem;color:{rcol};font-weight:700;text-transform:uppercase;
    letter-spacing:2px;margin-bottom:10px;'>{rcat}</div>
  <div style='font-size:1.1rem;font-weight:600;color:#E2E8F0;line-height:1.7;max-width:500px;margin:0 auto;'>
    "{raff}"
  </div>
</div>
""", unsafe_allow_html=True)

    # Wellbeing tip of the day
    tips = [
        ("📵","Phone-free mornings","Keep your phone out of reach for the first 30 minutes after waking. Start your day on your terms, not the algorithm's."),
        ("⏱️","The 20-20-20 rule","Every 20 minutes of screen time, look at something 20 feet away for 20 seconds. Your eyes — and mind — will thank you."),
        ("🚶","Walk without your phone","Take one short walk today without your phone. Notice the world. Reconnect with the present moment."),
        ("🔕","Notification audit","Delete or mute every app notification that doesn't require immediate action. Reclaim your attention."),
        ("📖","Read before sleep","Replace your pre-sleep scroll with 10 minutes of reading. Better sleep quality begins with a screen-free wind-down."),
        ("🎯","Single-task for 25 min","Pick one task and work on it exclusively for 25 minutes. No tabs, no phone, no multitasking."),
        ("🌿","Digital Sabbath hour","Set aside one hour per day as a complete screen-free zone. Use it to create, connect, or simply breathe."),
    ]
    tip_idx = datetime.date.today().toordinal() % len(tips)
    tip_icon, tip_title, tip_desc = tips[tip_idx]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);
  border-radius:18px;padding:24px 28px;'>
  <div style='font-size:0.72rem;color:#10B981;font-weight:700;text-transform:uppercase;
    letter-spacing:2px;margin-bottom:8px;'>💡 Wellbeing Tip of the Day</div>
  <div style='font-size:1.05rem;font-weight:700;color:#E2E8F0;margin-bottom:6px;'>
    {tip_icon} {tip_title}
  </div>
  <div style='font-size:0.88rem;color:#64748B;line-height:1.6;'>{tip_desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔮 Check My Wellbeing Score →", use_container_width=True):
            st.session_state.page      = "Predict"
            st.session_state.pred_step = 1
            st.session_state.pred_data = {}
            st.rerun()
