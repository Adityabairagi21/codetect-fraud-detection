"""
CoDetect - central configuration for Stage 1 (notebook + inference module).

Paths, dataset choice, the chronological split and run modes live here.
Column mapping and feature definitions live in services/feature_engineering.py
so Flask can reuse them without this file.
"""
from pathlib import Path

from services.feature_engineering import IBM_AML_COLUMN_MAP, TIMESTAMP_FORMAT  # noqa: F401

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------- dataset
# Source: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.
#   Official project : https://github.com/IBM/AML-Data  (README links the data)
#   Data host        : https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
DATASET_NAME = "IBM AML HI-Small"
TRANSACTIONS_FILE = DATA_RAW / "HI-Small_Trans.csv"
PATTERNS_FILE = DATA_RAW / "HI-Small_Patterns.txt"
ACCOUNTS_FILE = DATA_RAW / "HI-Small_accounts.csv"
COLUMN_MAP = IBM_AML_COLUMN_MAP
STRING_COLUMNS = ["sender_bank", "sender_account", "receiver_bank", "receiver_account"]

# ---------------------------------------------------------------- chronological split
# Half-open [start, end). Decided from scripts/inspect_dataset.py output:
#  * Sep 1 only builds account history (cold start) - not trained on or scored.
#  * Sep 11-18 (1,108 rows, ~58% laundering) is a simulation tail artefact and
#    is excluded from modelling; it is still shown in EDA.
SPLIT_WINDOWS = {
    "warmup":     ("2022-09-01", "2022-09-02"),
    "train":      ("2022-09-02", "2022-09-07"),
    "validation": ("2022-09-07", "2022-09-09"),
    "test":       ("2022-09-09", "2022-09-11"),
}
# FX rates are fitted on data up to the end of training (warm-up + train).
FX_FIT_SPLITS = ("warmup", "train")

# ---------------------------------------------------------------- execution
RANDOM_SEED = 42
RUN_MODE = "quick"   # "quick" | "full"
RUN_MODES = {
    # Features are always computed on every row (history counts require it).
    # Quick mode trains each Isolation Forest on a fixed random sample of the
    # training rows and draws graphs from a bounded window.
    "quick": {"train_sample_rows": 300_000},
    "full": {"train_sample_rows": None},
}
