import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import feature_engineering as fe  # noqa: E402

RAW = ROOT / "data" / "raw" / "HI-Small_Trans.csv"


@pytest.fixture(scope="session")
def full_clean():
    """The whole real HI-Small file, cleaned, plus FX rates fitted on it."""
    if not RAW.exists():
        pytest.skip("HI-Small_Trans.csv not downloaded")
    tx, _ = fe.clean_transactions(fe.load_transactions(RAW), reference_prefix="HI-Small")
    return tx, fe.fit_fx_rates(tx)


@pytest.fixture(scope="session")
def real_sample(full_clean):
    """Contiguous real window Sep 5 00:00-09:00 (many distinct minutes, ordinary activity).

    The file head is a poor sample: it covers only the first ~30 minutes of Sep 1.
    """
    tx, fx = full_clean
    m = (tx["timestamp"] >= "2022-09-05 00:00") & (tx["timestamp"] < "2022-09-05 09:00")
    return tx[m].reset_index(drop=True), fx


@pytest.fixture(scope="session")
def real_multiday_subset(full_clean):
    """54 hours of real rows (Sep 4 00:00 - Sep 6 06:00) whose sender is in a fixed
    random 3% of accounts. Any subset is a valid self-contained input, so this
    exercises the 24h window expiry on real data at NetworkX-checkable size."""
    tx, fx = full_clean
    m = (tx["timestamp"] >= "2022-09-04 00:00") & (tx["timestamp"] < "2022-09-06 06:00")
    w = tx[m]
    accts = np.sort(w["sender_id"].unique())
    chosen = set(np.random.default_rng(11).choice(accts, size=len(accts) * 3 // 100, replace=False))
    return w[w["sender_id"].isin(chosen)].reset_index(drop=True), fx
