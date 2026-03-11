"""
================================================================================
  src/predict.py  —  Command-line prediction interface
================================================================================
Usage:
    python src/predict.py \
        --screen_time 8 --app_switches 60 --sleep 6 \
        --notifications 80 --social_media 150 \
        --focus 55 --mood 60 --anxiety 6
================================================================================
"""

import argparse
import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDL  = os.path.join(BASE, "models")

def predict(screen_time, app_switches, sleep, notifications,
            social_media, focus, mood, anxiety):
    model    = joblib.load(os.path.join(MDL, "best_model.pkl"))
    scaler   = joblib.load(os.path.join(MDL, "scaler.pkl"))
    features = joblib.load(os.path.join(MDL, "feature_names.pkl"))

    row = {
        "daily_screen_time":     screen_time,
        "num_app_switches":      app_switches,
        "sleep_hours":           sleep,
        "notification_count":    notifications,
        "social_media_time_min": social_media,
        "focus_score":           focus,
        "mood_score":            mood,
        "anxiety_level":         anxiety,
    }
    row["dopamine_index"]   = social_media + notifications + app_switches
    row["focus_efficiency"] = focus / (screen_time + 1e-9)
    row["sleep_deficit"]    = 8 - sleep

    X = pd.DataFrame([row])[features]
    X_sc = scaler.transform(X)
    score = float(model.predict(X_sc)[0])
    score = max(0.0, min(100.0, score))

    print("\n" + "=" * 50)
    print("  DIGITAL WELLBEING PREDICTOR")
    print("=" * 50)
    print(f"  Screen time       : {screen_time} hrs/day")
    print(f"  App switches      : {app_switches}")
    print(f"  Sleep             : {sleep} hrs")
    print(f"  Notifications     : {notifications}")
    print(f"  Social media      : {social_media} min")
    print(f"  Focus score       : {focus}/100")
    print(f"  Mood score        : {mood}/100")
    print(f"  Anxiety level     : {anxiety}/10")
    print("-" * 50)
    print(f"  Dopamine Index    : {row['dopamine_index']:.0f}")
    print(f"  Focus Efficiency  : {row['focus_efficiency']:.2f}")
    print(f"  Sleep Deficit     : {row['sleep_deficit']:.1f} hrs")
    print("=" * 50)
    print(f"  ★ Predicted Wellbeing Score: {score:.1f} / 100")
    if   score >= 70: print("  → EXCELLENT digital wellbeing 🟢")
    elif score >= 50: print("  → GOOD – minor improvements possible 🟡")
    elif score >= 35: print("  → MODERATE – consider adjusting habits 🟠")
    else:             print("  → NEEDS ATTENTION – high-risk digital habits 🔴")
    print("=" * 50 + "\n")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Digital Wellbeing Score")
    parser.add_argument("--screen_time",   type=float, default=7.0)
    parser.add_argument("--app_switches",  type=int,   default=60)
    parser.add_argument("--sleep",         type=float, default=7.0)
    parser.add_argument("--notifications", type=int,   default=80)
    parser.add_argument("--social_media",  type=float, default=120.0)
    parser.add_argument("--focus",         type=float, default=65.0)
    parser.add_argument("--mood",          type=float, default=65.0)
    parser.add_argument("--anxiety",       type=float, default=4.0)
    args = parser.parse_args()
    predict(args.screen_time, args.app_switches, args.sleep,
            args.notifications, args.social_media,
            args.focus, args.mood, args.anxiety)
