"""Run the whole pipeline end to end: data -> extraction -> scoring -> recommendations ->
fairness audit -> evaluation -> tests. Then launch the app with `streamlit run app.py`.

    python run_all.py                 # offline: mock extractor unless the provider's key is set
    python run_all.py --mode mock     # force the offline extractor
    python run_all.py --mode llm      # force the LLM extractor (provider + key set in config/env)
    python run_all.py --mode llm --limit 10   # trial run: extract from only 10 documents first
    python run_all.py --no-embeddings # skip sentence-transformers entirely
    python run_all.py --skip-tests
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "mock", "llm"], default=None)
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="trial run: extract from only the first N documents (also caps the fairness "
                         "and evaluation samples at N)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.no_embeddings:
        os.environ["SPA_NO_EMBEDDINGS"] = "1"

    from src import evaluate, extract, fairness, generate_data, recommend, score

    steps = [
        ("1/6 Generate synthetic data", generate_data.main),
        ("2/6 Extract evidence",
         lambda: extract.run(args.mode, use_embeddings=not args.no_embeddings, limit=args.limit)),
        ("3/6 Score officers x roles", score.run),
        ("4/6 Recommend development", lambda: recommend.run(use_embeddings=not args.no_embeddings)),
        ("5/6 Fairness audit", lambda: fairness.run(args.mode, limit=args.limit)),
        ("6/6 Evaluate extraction", lambda: evaluate.run(args.mode, limit=args.limit)),
    ]
    for label, fn in steps:
        t = time.time()
        print(f"\n== {label}")
        fn()
        print(f"   ({time.time() - t:.1f}s)")

    if not args.skip_tests:
        print("\n== Tests")
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q"])
        if rc != 0:
            print("Tests FAILED")
            return rc
    print("\nDone. Launch the app with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
