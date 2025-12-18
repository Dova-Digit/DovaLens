import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp, chi2_contingency

# IMPORT CORRETTI
from dovalens.utils import is_categorical, clean_dataframe


def analyze(df: pd.DataFrame) -> dict:

    df = clean_dataframe(df)
    results = {}

    # ==========================
    # BASIC STATS
    # ==========================
    if "Age" in df.columns:
        s = df["Age"]
        results["age_stats"] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max())
        }

    if "Credit amount" in df.columns:
        s = df["Credit amount"]
        results["credit_stats"] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max())
        }

    if "Duration" in df.columns:
        s = df["Duration"]
        results["duration_stats"] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max())
        }

    # ==========================
    # CATEGORICAL COUNTS
    # ==========================
    for col in df.columns:
        s = df[col]
        if is_categorical(s):
            counts = s.astype(str).value_counts(dropna=False).to_dict()
            results[f"dist_{col}"] = {k: int(v) for k, v in counts.items()}

    # ==========================
    # BIMODALITY
    # ==========================
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) > 200:
            try:
                gm1 = GaussianMixture(1).fit(s.values.reshape(-1, 1))
                gm2 = GaussianMixture(2).fit(s.values.reshape(-1, 1))
                delta = gm1.bic(s.values.reshape(-1, 1)) - gm2.bic(s.values.reshape(-1, 1))
                results[f"bimodality_{col.lower()}"] = float(abs(delta))
            except:
                pass

    # ==========================
    # DRIFT (Duration)
    # ==========================
    if "Duration" in df.columns:
        s = df["Duration"].dropna()
        mid = len(s) // 2
        s1, s2 = s.iloc[:mid], s.iloc[mid:]
        stat, p = ks_2samp(s1, s2)
        results["drift_duration"] = {"ks_stat": float(stat), "p_value": float(p)}

    # ==========================
    # CLUSTERS
    # ==========================
    num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    if num.shape[1] >= 2:
        km = KMeans(n_clusters=3, n_init="auto").fit(num.values)
        labels = km.labels_
        sizes = pd.Series(labels).value_counts()
        results["cluster_sizes"] = {str(k): int(v) for k, v in sizes.items()}

    # ==========================
    # ANOMALIES
    # ==========================
    if num.shape[1] > 0:
        iso = IsolationForest(contamination=0.02)
        iso.fit(num.values)
        scores = -iso.score_samples(num.values)
        top = np.argsort(scores)[-10:]
        results["anomalies"] = [int(i) for i in top]

    return results
