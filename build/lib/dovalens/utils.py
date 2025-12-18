import numpy as np
import pandas as pd


def is_categorical(series: pd.Series) -> bool:
    """Return True if the column is categorical."""
    return series.dtype == "object" or series.nunique(dropna=True) <= 20


def safe_json(o):
    """Convert numpy types to pure Python for JSON serialization."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning — remove unnamed cols and convert numeric strings."""
    df = df.copy()

    drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        if all(x.replace(".", "", 1).isdigit() for x in s):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df
