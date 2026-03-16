import pandas as pd
import numpy as np

def load_and_clean(path="data/dataset.csv"):
    df = pd.read_csv(path)
    # Handle missing values with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # IQR outlier clipping
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

if __name__ == "__main__":
    df = load_and_clean()
    print("Cleaned shape:", df.shape)
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
