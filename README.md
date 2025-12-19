# DovaLens â€“ Automated Data Profiling & Drift Detection

[![GitHub Release](https://img.shields.io/github/v/release/Dova-Digit/DovaLens?display_name=tag&sort=semver)](https://github.com/Dova-Digit/DovaLens/releases/latest)
[![PyPI version](https://img.shields.io/pypi/v/dovalens.svg)](https://pypi.org/project/dovalens/)
[![Python versions](https://img.shields.io/pypi/pyversions/dovalens.svg)](https://pypi.org/project/dovalens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

DovaLens is a command-line tool that turns a raw CSV into a clean, visual **HTML report**.

- Dataset profiling (schema, preview, missing values)
- Summary statistics for numeric and categorical features
- Distribution breakdowns (by state/county/date, etc. when present)
- **Bimodality checks** on numeric targets (Pearson's coefficient)
- **Unsupervised clustering** (K-Means) for quick segmentation
- **Anomaly detection** (Isolation Forest) on multivariate signals
- **Drift signals** via two-sample **Kolmogorovâ€“Smirnov** tests
- A single, shareable **report.html**

> Built for fast EDA on small to very large CSV files.  
> Works out of the box, no notebook required.

---

## Installation

```bash
pip install dovalens

##If you are developing locally from the repo:

pip install -e .

##Quick Start

# Basic
dovalens path/to/your_dataset.csv

# Custom output path
dovalens path/to/your_dataset.csv --output path/to/report.html

If --output is omitted, the report is saved as ./report.html in the current working directory.

Works from any folder: pass either a relative path (.\examples\german_credit_data.csv on Windows) or an absolute one.

##CLI
usage: dovalens [-h] [--output OUTPUT] input

DovaLens â€” Automated dataset analyzer

positional arguments:
  input            Input CSV file

options:
  -h, --help       Show help and exit
  --output OUTPUT  Output HTML report path (default: ./report.html)

##Examples
# From the project root (Windows PowerShell)
dovalens .\examples\german_credit_data.csv

# Custom name and folder
dovalens .\examples\covid_de.csv --output .\covid_report.html

# From anywhere with absolute path

dovalens D:\data\sales_2024.csv --output D:\reports\sales_2024_report.html

##What's in the Report

Dataset preview (head, dtypes, inferred categorical columns)
Cleaning rules applied (remove Unnamed:*, numeric coercion)
Distributions for main fields (value counts / histograms)
Bimodality coefficients for selected numeric columns
Correlations (Pearson) on numeric features
Unsupervised clustering (K-Means, k auto-selected heuristically)
Anomalies via IsolationForest (top outliers)
Drift checks (two-sample KS) across common grouping keys when present (e.g., by state/county/date)

##How It Works (Technical Overview)
Loading & Cleaning
Drops columns like Unnamed:*
Safe numeric coercion for string-encoded numbers
Low-cardinality columns are treated as categorical

##Profiling & Statistics
Head/preview, dtypes, missingness
Summary stats for numeric & categorical features

##Signals & Metrics
Distributions / value counts
Pearson correlations for numeric pairs
Bimodality coefficient to flag multi-modal shapes

##Unsupervised Structure
K-Means on standardized numeric subsets to expose coarse segments
Cluster sizes reported to highlight dominant patterns

##Anomalies
Isolation Forest surfaces atypical rows based on multivariate behavior

##Drift
Two-sample KS tests compare distributions across groups (when sensible grouping keys exist)

##Report
Everything is assembled into a single, portable HTML file you can open and share.

##Performance Notes
Handles very large CSVs; if you hit memory limits, consider:

Running on a machine with more RAM
Pre-filtering columns not needed for EDA
Sampling rows for a quick first look

##Limitations
CSV schema inference may need manual cleanup for exotic formats
KS drift checks rely on meaningful grouping keys

##License
MIT â€” see LICENSE [blocked].

## DovaLens â€“ Profilazione automatica del dataset & Rilevamento del Drift (IT)
#DovaLens Ã¨ un tool da riga di comando che trasforma un CSV grezzo in un report HTML leggibile.

Profilazione dataset (schema, anteprima, valori mancanti)
Statistiche descrittive per feature numeriche e categoriche
Distribuzioni (per stato/provincia/data, quando presenti)
Controlli di bimodalitÃ  su target numerici (coeff. di Pearson)
Clustering non supervisionato (K-Means) per segmentazioni rapide
Rilevamento anomalie (Isolation Forest) su segnali multivariati
Drift con test Kolmogorovâ€“Smirnov a due campioni
Un unico report.html condivisibile


##Installazione
pip install dovalens

##Per sviluppo locale dal repository:

pip install -e .

##Avvio Rapido
# Base
dovalens path/al/tuo_dataset.csv

# Output personalizzato
dovalens path/al/tuo_dataset.csv --output path/al/report.html

Se --output non Ã¨ specificato, il report viene salvato come ./report.html nella cartella corrente.

Puoi usare un percorso relativo (.\examples\german_credit_data.csv) o assoluto.

##Cosa Contiene il Report
Anteprima dataset (head, dtypes, colonne categoriche inferite)
Regole di pulizia (rimozione Unnamed:*, coercizione numerica)
Distribuzioni dei campi principali (conteggi / istogrammi)
BimodalitÃ  per colonne numeriche selezionate
Correlazioni (Pearson)
Clustering (K-Means, k scelto euristicamente)
Anomalie con IsolationForest (outlier principali)
Drift (test KS) su chiavi di raggruppamento quando presenti

##Come Funziona (Overview)
Caricamento & pulizia â†’ Profilazione â†’ Segnali (distribuzioni, correlazioni, bimodalitÃ ) â†’ Clustering (K-Means) â†’ Anomalie (Isolation Forest) â†’ Drift (KS) â†’ report HTML unico.

##Note di Performance
Gestisce CSV molto grandi; in caso di limiti di memoria valuta:

Macchina con piÃ¹ RAM
Selezione delle sole colonne utili
Campionamento righe per una prima occhiata

##Limitazioni
L'inferenza dello schema puÃ² richiedere fix manuali per formati atipici
I controlli di drift richiedono chiavi di gruppo significative

##Licenza
MIT â€” vedi LICENSE [blocked].
