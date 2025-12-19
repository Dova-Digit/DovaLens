# Changelog

## 1.0.5 — 2025-12-19
- Report: added a `numeric_stats` section (mean/std/min/max) for numeric columns.
- Report: added a `bimodality` metric for numeric fields (simple indicator of non-unimodality).
- Report: the `anomalies` list (IsolationForest) now exposes the DataFrame’s original indices and is capped to the first N (default 10).
- Report: more robust KMeans logic on “wide” datasets and consistent `cluster_sizes` output.
- UI: small CSS tweaks to the report (dark theme) and more stable rendering of `pre` blocks.
- Dev: examples/README updated in line with the new report sections.

## 1.0.4 — 2025-12-19
- CLI: clearer mode message (`[INFO] Using DataFrame mode`) and clean support for CSV path or DataFrame input.
- Packaging: build/metadata tidied up; minor internal cleanups.

## 1.0.3 — 2025-12-19
- CLI: `--version` + path/DataFrame fallback.
- Fix: pandas warning on `to_numeric`.

## 1.0.1 — 2025-12-19
- README fixed, packaging cleaned up.

## 1.0.0 — 2025-12-18
- Initial release.
