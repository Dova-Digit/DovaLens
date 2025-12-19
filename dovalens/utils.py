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

    # Drop columns like 'Unnamed: 0'
    drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Convert numeric-looking values per-column WITHOUT using deprecated errors="ignore"
    for col in df.columns:
        s = df[col]
        # prova la conversione numerica; i non convertibili diventano NaN
        converted = pd.to_numeric(s, errors="coerce")

        # sostituisci solo dove la conversione è riuscita
        # (se è NaN, lascia il valore originale — stessa semantica di "ignore")
        df[col] = s.where(converted.isna(), converted)

    return df
