"""
Full-data dry run of the feature pipeline - NO model training.

Clean -> assign chronological split -> fit FX on warm-up + train ->
compute leakage-safe features for every modelled row -> report timing,
memory and per-split feature summaries.
Writes outputs/metrics/feature_dry_run.json.

Run:  venv\\Scripts\\python.exe scripts\\feature_dry_run.py
"""
import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from services import feature_engineering as fe  # noqa: E402


def peak_mem_gb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 1e9, 2)


t = {}
t0 = time.perf_counter()
raw = fe.load_transactions(C.TRANSACTIONS_FILE, C.COLUMN_MAP)
t["load_s"] = time.perf_counter() - t0

t0 = time.perf_counter()
tx, clean_report = fe.clean_transactions(raw, reference_prefix="HI-Small")
del raw
tx["split"] = fe.assign_time_split(tx["timestamp"], C.SPLIT_WINDOWS)
t["clean_s"] = time.perf_counter() - t0
print("cleaning:", clean_report)
split_counts = tx.groupby("split")["label"].agg(rows="size", laundering="sum")
print(split_counts)

modelled = tx[tx["split"] != "excluded"].reset_index(drop=True)
fx = fe.fit_fx_rates(modelled[modelled["split"].isin(C.FX_FIT_SPLITS)])
print("FX currencies without rate:", fx["currencies_without_rate"])

t0 = time.perf_counter()
feats = fe.compute_features(modelled, fx)
t["features_s"] = time.perf_counter() - t0

t0 = time.perf_counter()
tail = fe.history_tail(modelled)
t["history_tail_s"] = time.perf_counter() - t0
history_info = {"rows": int(len(tail)),
                "from": str(pd.Timestamp(int(tail["minute"].min()) * 60, unit="s")),
                "to": str(pd.Timestamp(int(tail["minute"].max()) * 60, unit="s"))}

num = [f["name"] for f in fe.FEATURE_SCHEMA if f["kind"] != "categorical"]
feats["split"] = modelled["split"].to_numpy()
summary = feats.groupby("split")[num].median().T
print("\nper-split medians:\n", summary.round(3).to_string())
nan_count = int(feats[num].isna().sum().sum())
print("NaN feature values:", nan_count)
t = {k: round(v, 1) for k, v in t.items()}
print("timings (s):", t, "| peak working set (GB):", peak_mem_gb())
print("history tail for inference:", history_info)

out = {
    "cleaning": clean_report,
    "split_counts": {k: {"rows": int(r.rows), "laundering": int(r.laundering)}
                     for k, r in split_counts.iterrows()},
    "fx": fx, "feature_version": fe.FEATURE_VERSION,
    "features": fe.ALL_FEATURES, "nan_feature_values": nan_count,
    "per_split_medians": summary.round(6).to_dict(),
    "timings_s": t, "peak_working_set_gb": peak_mem_gb(),
    "history_window_minutes": fe.HISTORY_WINDOW_MINUTES,
    "history_tail": history_info,
}
(C.METRICS_DIR / "feature_dry_run.json").write_text(json.dumps(out, indent=2, default=str))
print("saved", C.METRICS_DIR / "feature_dry_run.json")
