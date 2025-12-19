# dovalens/report.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union, Dict, Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

from .utils import clean_dataframe, safe_json


def _numeric_cols(df: pd.DataFrame) -> Iterable[str]:
    """Ritorna la lista delle colonne numeriche del DataFrame."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _value_counts(series: pd.Series, top: int | None = None) -> Dict[str, int]:
    """
    Value counts robusto: cast a string, sostituisce NaN con 'nan', e restituisce un dict {valore: conteggio}.
    Se top è valorizzato, tronca ai primi N.
    """
    s = series.copy()
    s = s.astype("string")
    s = s.fillna("nan")
    vc = s.value_counts(dropna=False)
    if top is not None:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


def _bimodality_coeff(x: pd.Series) -> float:
    """
    Indicatore semplice di non-unimodalità (robusto come 'flag', NON per inferenza statistica fine).
    Restituisce NaN se meno di 5 valori validi.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.size < 5:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return float("nan")
    # Misure di forma: skewness (g1) e excess kurtosis (g2)
    g1 = ((x - m) ** 3).mean() / (s ** 3)
    g2 = ((x - m) ** 4).mean() / (s ** 4) - 3
    # Heuristica: cresce con asimmetria e leptocurtosi
    return float(abs(g1) + max(0.0, g2))


def _kmeans_cluster_sizes(df: pd.DataFrame, max_k: int = 3) -> Dict[str, int]:
    """
    Calcola la dimensione dei cluster con KMeans su tutte le numeriche.
    Per allineamento col report storico: se ci sono >=3 feature numeriche, forza k=3 (limitato da max_k),
    altrimenti usa k=2.
    """
    num_cols = _numeric_cols(df)
    if len(num_cols) == 0:
        return {}
    X = df[num_cols].dropna()
    if X.shape[0] < 10:
        return {}
    Xs = StandardScaler().fit_transform(X.values)

    k = 3 if Xs.shape[1] >= 3 else 2
    k = min(k, max_k)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    vc = pd.Series(labels).value_counts().sort_index()
    return {str(i): int(vc.get(i, 0)) for i in range(k)}


def _isoforest_anomalies(df: pd.DataFrame, n: int = 10) -> list[int]:
    """
    Seleziona le top-N anomalie usando IsolationForest, ordinate per 'anomalia' (decision_function crescente).
    Restituisce gli indici di riga originali (int).
    """
    num_cols = _numeric_cols(df)
    if len(num_cols) == 0:
        return []
    X = df[num_cols].dropna()
    if X.shape[0] < 20:
        return []
    iso = IsolationForest(random_state=42, n_estimators=200, contamination="auto")
    iso.fit(X.values)

    # decision_function: valori più bassi => più anomali
    scores = iso.decision_function(X.values)
    order = np.argsort(scores)  # crescente
    anomal_idx = X.index[order[:n]].tolist()
    return list(map(int, anomal_idx))


def _maybe_ks_duration(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Esegue un KS-test su 'Duration' separando per 'Sex' se presenti, altrimenti non applicabile.
    Restituisce {} se mancano le colonne o dati sufficienti.
    """
    if "Duration" not in df.columns:
        return {}
    if "Sex" not in df.columns:
        return {}
    a = pd.to_numeric(df.loc[df["Sex"] == "male", "Duration"], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df["Sex"] == "female", "Duration"], errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10:
        return {}
    stat, p = ks_2samp(a.values, b.values)
    return {"ks_stat": float(np.round(stat, 3)), "p_value": float(p)}


def _analysis_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Costruisce il dizionario 'analysis' con:
    - statistiche e bimodalità per tutte le numeriche,
    - distibuzioni per categoriche comuni (german-credit e covid_de),
    - semplice drift-test su Duration per Sex (se applicabile),
    - KMeans cluster sizes,
    - top-N anomalie ordinate.
    """
    out: Dict[str, Any] = {}

    # --- statistiche + bimodalità per TUTTE le numeriche ---
    for col in _numeric_cols(df):
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.size >= 5:
            out.setdefault("numeric_stats", {})
            out["numeric_stats"][col] = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            out.setdefault("bimodality", {})
            out["bimodality"][col] = _bimodality_coeff(x)

    # --- compat con german-credit (se presenti) ---
    if "Age" in df.columns:
        x = pd.to_numeric(df["Age"], errors="coerce").dropna()
        if len(x) > 0:
            out["age_stats"] = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            out["bimodality_age"] = _bimodality_coeff(x)

    if "Credit amount" in df.columns:
        x = pd.to_numeric(df["Credit amount"], errors="coerce").dropna()
        if len(x) > 0:
            out["credit_stats"] = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            out["bimodality_credit amount"] = _bimodality_coeff(x)

    if "Duration" in df.columns:
        x = pd.to_numeric(df["Duration"], errors="coerce").dropna()
        if len(x) > 0:
            out["duration_stats"] = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "min": float(x.min()),
                "max": float(x.max()),
            }
            out["bimodality_duration"] = _bimodality_coeff(x)

    # --- distribuzioni categoriche comuni + covid_de ---
    for col in (
        "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose",
        "state", "county", "age_group", "gender", "date"
    ):
        if col in df.columns:
            out[f"dist_{col}"] = _value_counts(df[col])

    # --- drift semplice per Duration tra Sex (se applicabile) ---
    drift = _maybe_ks_duration(df)
    if drift:
        out["drift_duration"] = drift

    # --- KMeans cluster sizes ---
    cs = _kmeans_cluster_sizes(df)
    if cs:
        out["cluster_sizes"] = cs

    # --- anomalie ordinate ---
    an = _isoforest_anomalies(df)
    if an:
        out["anomalies"] = an

    return out


def _render_html(df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """
    Rendering HTML minimale in tema scuro con:
    - preview del dataset (prime 10 righe),
    - JSON dell'analisi.
    """
    preview = df.head(10).to_string()
    analysis_json = json.dumps(analysis, indent=4, default=safe_json)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DovaLens Automated Report</title>
<style>
  :root {{
    --bg: #0b0b0b;
    --panel: #111214;
    --text: #e9eef5;
    --muted: #a9b3be;
    --accent: #3fa9f5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px 32px;
    background: var(--bg); color: var(--text);
    font: 15px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  }}
  h1 {{
    color: var(--accent);
    margin: 0 0 20px; font-size: 32px; letter-spacing: .3px;
  }}
  h2 {{
    margin: 28px 0 12px; font-size: 24px; color: var(--accent);
  }}
  .card {{
    background: var(--panel);
    border: 1px solid #1c1f24;
    border-radius: 12px; padding: 14px 16px; overflow: auto;
    box-shadow: 0 0 0 1px rgb(255 255 255 / 0.02) inset, 0 8px 24px rgb(0 0 0 / 0.35);
  }}
  pre {{ margin: 0; white-space: pre; }}
</style>
</head>
<body>
  <h1>DovaLens Automated Report</h1>

  <h2>Dataset Preview</h2>
  <div class="card"><pre>{preview}</pre></div>

  <h2>Analysis Output</h2>
  <div class="card"><pre>{analysis_json}</pre></div>
</body>
</html>
"""


def generate_report(
    df_or_path: Union[pd.DataFrame, str, os.PathLike],
    output: Union[str, os.PathLike]
) -> Dict[str, Any]:
    """
    Accetta un DataFrame OPPURE un path CSV. Restituisce il dict 'analysis' e scrive l'HTML.
    """
    # 1) normalizza input
    if isinstance(df_or_path, (str, os.PathLike, Path)):
        df = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        raise TypeError("generate_report: first argument must be a DataFrame or a CSV path")

    # 2) cleaning
    df = clean_dataframe(df)

    # 3) analisi
    analysis = _analysis_dict(df)

    # 4) render + write
    html = _render_html(df, analysis)
    out_path = Path(output)
    out_path.write_text(html, encoding="utf-8")

    return analysis
