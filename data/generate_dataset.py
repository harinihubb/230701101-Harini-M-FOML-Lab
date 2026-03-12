"""
generate_dataset.py
-------------------
Generates a synthetic dataset simulating digital behavior and wellbeing metrics
for 1000 users. Data is designed to reflect realistic correlations:
  - More sleep → better wellbeing
  - High social media / notification overload → lower wellbeing
  - Better focus & mood → higher wellbeing
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

sleep_hours        = np.random.normal(6.5, 1.2, N).clip(3, 10)
daily_screen_time  = np.random.normal(7, 2, N).clip(1, 16)
social_media_time  = (daily_screen_time * 0.35 + np.random.normal(0, 0.5, N)).clip(0, 10) * 60
notification_count = (social_media_time / 60 * 12 + np.random.normal(0, 5, N)).clip(0, 200).astype(int)
num_app_switches   = (daily_screen_time * 8 + np.random.normal(0, 5, N)).clip(5, 200).astype(int)
focus_score        = (80 - daily_screen_time * 2 + sleep_hours * 2 + np.random.normal(0, 5, N)).clip(10, 100)
mood_score         = (70 + sleep_hours * 2 - social_media_time / 60 + np.random.normal(0, 5, N)).clip(10, 100)
anxiety_level      = (50 + notification_count * 0.1 - sleep_hours * 2 + np.random.normal(0, 5, N)).clip(1, 10)

# Target: digital_wellbeing_score (0-100)
digital_wellbeing_score = (
    mood_score * 0.25
    + focus_score * 0.20
    + sleep_hours * 3
    - anxiety_level * 1.5
    - daily_screen_time * 1.2
    - notification_count * 0.05
    + np.random.normal(0, 3, N)
).clip(10, 100)

df = pd.DataFrame({
    "daily_screen_time":       daily_screen_time.round(2),
    "num_app_switches":        num_app_switches,
    "sleep_hours":             sleep_hours.round(2),
    "notification_count":      notification_count,
    "social_media_time_min":   social_media_time.round(2),
    "focus_score":             focus_score.round(2),
    "mood_score":              mood_score.round(2),
    "anxiety_level":           anxiety_level.round(2),
    "digital_wellbeing_score": digital_wellbeing_score.round(2),
})

# Inject ~2% missing values for realism
for col in ["sleep_hours", "focus_score", "mood_score"]:
    idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[idx, col] = np.nan

# Inject a few duplicate rows
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)

df.to_csv("digital_wellbeing_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df.shape[1]} columns")
print(df.head())
