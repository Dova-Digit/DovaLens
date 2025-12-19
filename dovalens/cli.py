import argparse
import sys
import inspect
import pandas as pd
from . import __version__

def _resolve_pipeline():
    # Preferisci report.generate_report
    try:
        from .report import generate_report
        return generate_report
    except Exception:
        pass
    # Fallback su core
    for name in ("generate_report", "run", "run_analysis"):
        try:
            from . import core  # type: ignore
            return getattr(core, name)  # type: ignore
        except Exception:
            continue
    raise ImportError(
        "Impossibile trovare una funzione di pipeline. "
        "Attese una di: report.generate_report, core.generate_report, core.run, core.run_analysis"
    )

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dovalens",
        description="DovaLens — Automated dataset analyzer"
    )
    p.add_argument("--version", action="version", version=f"dovalens {__version__}")
    p.add_argument("input", help="Input CSV file")
    p.add_argument("--output", default="./report.html",
                   help="Output HTML report path (default: ./report.html)")
    return p

def _call_with_keywords(func, a, b):
    """
    Prova a chiamare func con 2 argomenti usando i NOMI della firma:
    - prima (df, output)
    - poi (path, output)
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    if len(params) != 2:
        # cade al positional
        func(a, b)
        return

    # 1) tenta (df, output) per nome
    try:
        kwargs = {params[0]: a, params[1]: b}
        func(**kwargs)
        return
    except Exception:
        pass

    # 2) tenta (path, output) per nome
    try:
        kwargs = {params[0]: str(a), params[1]: b}
        func(**kwargs)
        return
    except Exception:
        # ultimo tentativo: posizionale
        func(a, b)
        return

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        pipeline = _resolve_pipeline()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    # Proviamo a leggere il CSV UNA sola volta
    df = None
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        df = None

    # 1) PRIORITARIO: (DataFrame, output)
    if df is not None:
        try:
            print("[INFO] Using DataFrame mode")
            _call_with_keywords(pipeline, df, args.output)
            return
        except TypeError:
            # se davvero non è compatibile, andremo al fallback
            pass
        except Exception as e:
            # errore reale durante l'analisi → meglio fallire esplicitamente
            print(f"[ERROR] Pipeline failed with DataFrame: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)

    # 2) FALLBACK: (path, output)
    try:
        print("[INFO] Using path mode")
        _call_with_keywords(pipeline, args.input, args.output)
        return
    except Exception as e2:
        print(f"[ERROR] Pipeline failed with path: {type(e2).__name__}: {e2}", file=sys.stderr)
        sys.exit(1)
