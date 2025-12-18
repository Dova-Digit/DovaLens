# ===============================================================
# DOVALENS v6 — Fully Working Automated Report (Plotly Fixed)
# ===============================================================

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp, chi2_contingency

# ===============================================================
# Helpers
# ===============================================================

def is_categorical(s: pd.Series) -> bool:
    return s.dtype == "object" or s.nunique(dropna=True) <= 20


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [c for c in df.columns if "unnamed" in c.lower() or c.lower() == "index"]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace(["NA", "N/A", "null", ""], np.nan)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    return df

# ===============================================================
# Insight Object
# ===============================================================

class Insight:
    def __init__(self, kind, title, description, score, figure_html, meta):
        self.kind = kind
        self.title = title
        self.description = description
        self.score = score
        self.figure_html = figure_html
        self.meta = meta

# ===============================================================
# Distribution
# ===============================================================

def insight_distribution(col: str, series: pd.Series) -> Insight:
    s = series.dropna()

    if is_categorical(series):
        counts = s.astype(str).value_counts(dropna=False)
        order = counts.index.tolist()

        fig = px.bar(
            x=order,
            y=[counts[k] for k in order],
            title=f"Distribution of {col}",
            labels={"x": col, "y": "Count"},
        )

        fig.update_layout(template="plotly_dark", height=400)

        return Insight(
            "distribution",
            f"Distribution: {col}",
            f"Distribution analysis for {col}.",
            0.3,
            fig.to_html(include_plotlyjs="cdn"),
            {"counts": {str(k): int(v) for k, v in counts.items()}},
        )

    fig = px.histogram(
        s,
        nbins=40,
        title=f"Distribution of {col}",
        labels={"value": col, "count": "Count"},
    )
    fig.update_layout(template="plotly_dark", height=400)

    return Insight(
        "distribution",
        f"Distribution: {col}",
        f"Distribution analysis for {col}.",
        0.3,
        fig.to_html(include_plotlyjs="cdn"),
        {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        },
    )

# ===============================================================
# Bimodality
# ===============================================================

def insight_bimodality(col, s):
    s = s.dropna()
    if not np.issubdtype(s.dtype, np.number):
        return None
    if s.nunique() <= 20 or len(s) < 200:
        return None

    data = s.values.reshape(-1, 1)

    gm1 = GaussianMixture(1, random_state=42).fit(data)
    gm2 = GaussianMixture(2, random_state=42).fit(data)
    delta = gm1.bic(data) - gm2.bic(data)

    if delta < 50:
        return None

    fig = px.histogram(s, nbins=40, title=f"Bimodality {col} (ΔBIC={delta:.1f})")
    fig.update_layout(template="plotly_dark", height=400)

    return Insight(
        "bimodality",
        f"Bimodality: {col}",
        f"{col} may contain two distinct groups.",
        min(1.0, delta / 300),
        fig.to_html(include_plotlyjs="cdn"),
        {"bic_delta": float(delta)},
    )

# ===============================================================
# Drift
# ===============================================================

def _plot_hist_diff(col, s1, s2):
    df_plot = pd.DataFrame({col: pd.concat([s1, s2]), "split": ["first"] * len(s1) + ["second"] * len(s2)})
    fig = px.histogram(df_plot, x=col, color="split", nbins=40, barmode="overlay", title=f"Drift – {col}")
    fig.update_layout(template="plotly_dark", height=400)
    return fig.to_html(include_plotlyjs="cdn")


def _plot_bar_diff(col, s1, s2):
    df_plot = pd.DataFrame({col: pd.concat([s1, s2]), "split": ["first"] * len(s1) + ["second"] * len(s2)})
    counts = df_plot.groupby(["split", col]).size().reset_index(name="Count")
    fig = px.bar(counts, x=col, y="Count", color="split", barmode="group", title=f"Drift – {col}")
    fig.update_layout(template="plotly_dark", height=400)
    return fig.to_html(include_plotlyjs="cdn")


def insight_drift(col, s):
    s = s.dropna()
    if len(s) < 20:
        return None

    mid = len(s) // 2
    s1, s2 = s.iloc[:mid], s.iloc[mid:]

    if np.issubdtype(s.dtype, np.number):
        stat, p = ks_2samp(s1, s2)
        if p < 0.05:
            return Insight(
                "drift",
                f"Feature Drift: {col}",
                f"Numeric drift detected (p={p:.4f}).",
                min(1, stat),
                _plot_hist_diff(col, s1, s2),
                {"ks_stat": float(stat), "p_value": float(p)},
            )
        return None

    contingency = pd.crosstab(s1, s2)
    if contingency.empty:
        return None

    chi2, p, *_ = chi2_contingency(contingency)
    if p < 0.05:
        return Insight(
            "drift",
            f"Feature Drift: {col}",
            f"Categorical drift detected (p={p:.4f}).",
            min(1, chi2 / 20),
            _plot_bar_diff(col, s1, s2),
            {"chi2": float(chi2), "p_value": float(p)},
        )

    return None

# ===============================================================
# Clusters
# ===============================================================

def insight_clusters(df):
    num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    if num.shape[1] < 2:
        return None

    km = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(num.values)
    labels = km.labels_

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit_transform(num.values)

    fig = px.scatter(x=pca[:, 0], y=pca[:, 1], color=labels.astype(str), title="Cluster Analysis (PCA)")
    fig.update_layout(template="plotly_dark", height=400)

    return Insight(
        "clusters",
        "Cluster Analysis",
        "3 clusters detected.",
        0.7,
        fig.to_html(include_plotlyjs="cdn"),
        {"cluster_sizes": dict(pd.Series(labels).value_counts())},
    )

# ===============================================================
# Anomalies
# ===============================================================

def insight_anomalies(df):
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] == 0:
        return None

    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(num)

    scores = -iso.score_samples(num)
    top = np.argsort(scores)[-10:]

    fig = px.scatter(x=num.index, y=scores, title="Anomalies (IsolationForest)", labels={"x": "Index", "y": "Score"})
    fig.update_layout(template="plotly_dark", height=400)

    return Insight(
        "anomalies",
        "Anomalies",
        "Possible anomalies detected.",
        0.6,
        fig.to_html(include_plotlyjs="cdn"),
        {"indices": [int(i) for i in top]},
    )

# ===============================================================
# Report
# ===============================================================

def build_report(title, filename, df, insights):
    html = [
        "<html><head><meta charset='utf-8'>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>body{background:#111;color:#ddd;font-family:Arial;padding:20px;} h1,h2{color:#6cf;} table{color:#ddd;border-collapse:collapse;} th,td{border:1px solid #444;padding:6px;}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p><b>{filename}</b> — {df.shape[0]} rows × {df.shape[1]} columns</p>",
        df.head(10).to_html(index=False),
        "<hr>",
    ]

    for ins in insights:
        html.append(f"<h2>{ins.title}</h2>")
        html.append(f"<p>{ins.description}</p>")
        html.append(ins.figure_html)
        html.append("<pre>" + str(ins.meta) + "</pre><hr>")

    html.append("</body></html>")
    return "".join(html)

# ===============================================================
# Main Pipeline
# ===============================================================

def run_dovalens(path_input, path_output):
    df = pd.read_csv(path_input)
    df = clean_dataframe(df)

    insights = []

    for col in df.columns:
        s = df[col]
        insights.append(insight_distribution(col, s))

        bim = insight_bimodality(col, s)
        if bim:
            insights.append(bim)

        drift = insight_drift(col, s)
        if drift:
            insights.append(drift)

    cl = insight_clusters(df)
    if cl:
        insights.append(cl)

    an = insight_anomalies(df)
    if an:
        insights.append(an)

    html = build_report("DovaLens Automated Report", path_input, df, insights)
    with open(path_output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Report generated → {path_output}")

# ===============================================================
# CLI
# ===============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    run_dovalens(args.input, args.output)
