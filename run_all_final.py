#!/usr/bin/env python3
"""
Convenience runner for final selected submissions.

Default mode:
    Reproduce v16 and v19 from included reference submissions v2 and v6.

Usage:
    pip install -r requirements.txt
    python run_all_final.py

Outputs:
    outputs/submission_v16_LINESEARCH_public_private_g092.csv
    outputs/submission_v19_LINESEARCH_public_private_g138.csv

Optional full raw-data pipeline:
    If you need to regenerate v2/v6 from train.csv and sample_submission.csv, place
    the competition data in data/ and run the scripts manually:

        python scripts/01_generate_v2_variants.py
        python scripts/02_generate_v6_to_v9.py

    Then move/copy the generated v2 and v6 CSVs into reference_submissions/ and run
    this file again.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "03_generate_final_v16_v19.py"
    cmd = [
        sys.executable,
        str(script),
        "--reference-dir",
        str(root / "reference_submissions"),
        "--output-dir",
        str(root / "outputs"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
