import pandas as pd

def engineer_features(df):
    df = df.copy()
    # dopamine_index: approximates digital stimulation intensity
    df["dopamine_index"] = df["social_media_time_min"] + df["notification_count"] + df["num_app_switches"]
    # focus_efficiency: productivity relative to screen usage
    df["focus_efficiency"] = df["focus_score"] / (df["daily_screen_time"] + 0.01)
    # sleep_deficit: deviation from recommended 8h sleep
    df["sleep_deficit"] = (8 - df["sleep_hours"]).clip(0)
    return df

FEATURE_COLS = [
    "daily_screen_time", "num_app_switches", "sleep_hours",
    "notification_count", "social_media_time_min", "focus_score",
    "mood_score", "anxiety_level", "dopamine_index",
    "focus_efficiency", "sleep_deficit"
]
TARGET = "digital_wellbeing_score"
