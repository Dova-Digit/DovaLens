import argparse
import pandas as pd
from dovalens.core import analyze
from dovalens.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="DovaLens — Automated dataset analyzer")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("--output", default="report.html", help="Output HTML report")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    print("[INFO] Running analysis...")
    results = analyze(df)

    print("[INFO] Generating report...")
    path = generate_report(df, results, args.output)

    print(f"\nReport generated → {path}")
    print("Done.")
