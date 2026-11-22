"""
Read-only inspection of the raw IBM AML transaction file.

Answers the questions that must be settled BEFORE feature engineering:
schema, dtypes, missing/duplicates, timestamp range and ordering, daily
volume vs. label rate, account-key uniqueness, currencies and implied
exchange rates. Writes outputs/metrics/dataset_inspection.json.

Run:  venv\\Scripts\\python.exe scripts\\inspect_dataset.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

out = {}
pd.set_option("display.width", 160)


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


# ------------------------------------------------------------------ header
section("RAW HEADER")
with open(C.TRANSACTIONS_FILE, encoding="utf-8") as fh:
    raw_header = fh.readline().strip().split(",")
print(raw_header)
dupes = [c for c, n in Counter(raw_header).items() if n > 1]
print("duplicate header names:", dupes)
out["raw_header"] = raw_header
out["duplicate_header_names"] = dupes

# ------------------------------------------------------------------ load
section("LOAD")
cm = C.COLUMN_MAP
str_cols = [cm[k] for k in C.STRING_COLUMNS]
df = pd.read_csv(C.TRANSACTIONS_FILE, dtype={c: str for c in str_cols})
missing_map = [v for v in cm.values() if v not in df.columns]
assert not missing_map, f"COLUMN_MAP entries not in file: {missing_map}"
df = df.rename(columns={v: k for k, v in cm.items()})
print("shape:", df.shape)
print(df.dtypes.to_string())
print(df.head(5).to_string())
out["shape"] = list(df.shape)
out["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}

# ------------------------------------------------------------------ quality
section("MISSING / DUPLICATES")
na = df.isna().sum()
print(na.to_string())
n_dup = int(df.duplicated().sum())
n_dup_nolabel = int(df.drop(columns="label").duplicated().sum())
print("exact duplicate rows:", n_dup, "| duplicates ignoring label:", n_dup_nolabel)
dup_rows = df[df.duplicated(keep=False)]
print("laundering rows among exact duplicates:", int(dup_rows["label"].sum()))
out["missing"] = {k: int(v) for k, v in na.items()}
out["exact_duplicates"] = n_dup
out["duplicates_ignoring_label"] = n_dup_nolabel
out["laundering_rows_in_duplicate_groups"] = int(dup_rows["label"].sum())

# ------------------------------------------------------------------ timestamps
section("TIMESTAMPS")
ts = pd.to_datetime(df["timestamp"], format=C.TIMESTAMP_FORMAT, errors="coerce")
print("unparseable:", int(ts.isna().sum()))
print("min:", ts.min(), "| max:", ts.max())
non_mono = int((ts.diff() < pd.Timedelta(0)).sum())
print("rows where time goes backwards vs previous row:", non_mono)
print("seconds component always zero:", bool((ts.dt.second == 0).all()))
per_min = ts.value_counts()
print("distinct minutes:", per_min.size, "| max rows sharing one minute:", int(per_min.max()))
df["ts"] = ts
daily = df.groupby(ts.dt.date).agg(rows=("label", "size"), laundering=("label", "sum"))
daily["laundering_rate_%"] = (100 * daily["laundering"] / daily["rows"]).round(4)
print(daily.to_string())
out["timestamp"] = {
    "unparseable": int(ts.isna().sum()),
    "min": str(ts.min()), "max": str(ts.max()),
    "rows_out_of_order": non_mono,
    "minute_resolution": bool((ts.dt.second == 0).all()),
    "max_rows_same_minute": int(per_min.max()),
}
out["daily"] = {str(k): {kk: (float(vv) if isinstance(vv, float) else int(vv))
                         for kk, vv in r.items()} for k, r in daily.iterrows()}

# ------------------------------------------------------------------ labels
section("LABELS")
n_pos = int(df["label"].sum())
print(f"laundering: {n_pos} / {len(df)} = {100 * n_pos / len(df):.4f}%")
out["label_counts"] = {"laundering": n_pos, "normal": int(len(df) - n_pos)}

# ------------------------------------------------------------------ accounts
section("ACCOUNT KEYS")
for side in ("sender", "receiver"):
    b, a = df[f"{side}_bank"], df[f"{side}_account"]
    print(f"{side}: banks={b.nunique()} accounts={a.nunique()} "
          f"bank+account={(b + '|' + a).nunique()}")
acct_banks = pd.concat([
    df[["sender_account", "sender_bank"]].set_axis(["acc", "bank"], axis=1),
    df[["receiver_account", "receiver_bank"]].set_axis(["acc", "bank"], axis=1),
]).drop_duplicates()
multi = int((acct_banks.groupby("acc")["bank"].nunique() > 1).sum())
print("account ids appearing under >1 bank:", multi)
all_keys = pd.concat([df.sender_bank + "|" + df.sender_account,
                      df.receiver_bank + "|" + df.receiver_account])
print("distinct bank|account nodes:", all_keys.nunique())
bad_acct = int((~df.sender_account.str.fullmatch(r"[0-9A-F]+")).sum()
               + (~df.receiver_account.str.fullmatch(r"[0-9A-F]+")).sum())
bad_bank = int((~df.sender_bank.str.fullmatch(r"\d+")).sum()
               + (~df.receiver_bank.str.fullmatch(r"\d+")).sum())
print("non-hex account ids:", bad_acct, "| non-numeric bank codes:", bad_bank)
self_tx = df.sender_bank.eq(df.receiver_bank) & df.sender_account.eq(df.receiver_account)
print("self-transfers:", int(self_tx.sum()), "| laundering among them:", int(df.label[self_tx].sum()))
print(df[self_tx].payment_format.value_counts().to_string())
pairs = (df.sender_bank + "|" + df.sender_account + ">" + df.receiver_bank + "|" + df.receiver_account)
print("distinct directed pairs:", pairs.nunique())
out["accounts"] = {
    "account_ids_under_multiple_banks": multi,
    "distinct_nodes_bank_account": int(all_keys.nunique()),
    "non_hex_account_ids": bad_acct, "non_numeric_bank_codes": bad_bank,
    "self_transfers": int(self_tx.sum()),
    "laundering_self_transfers": int(df.label[self_tx].sum()),
    "distinct_directed_pairs": int(pairs.nunique()),
}

# ------------------------------------------------------------------ categories
section("PAYMENT FORMAT / CURRENCY")
pf = df.groupby("payment_format")["label"].agg(rows="size", laundering="sum")
pf["rate_%"] = (100 * pf.laundering / pf.rows).round(4)
print(pf.sort_values("rows", ascending=False).to_string())
print(df.payment_currency.value_counts().to_string())
out["payment_format"] = {k: {"rows": int(r.rows), "laundering": int(r.laundering)} for k, r in pf.iterrows()}
out["payment_currencies"] = df.payment_currency.value_counts().to_dict()
out["receiving_currencies"] = df.receiving_currency.value_counts().to_dict()

# ------------------------------------------------------------------ amounts
section("AMOUNTS")
for c in ("amount_paid", "amount_received"):
    s = df[c]
    print(f"{c}: dtype={s.dtype} min={s.min()} max={s.max()} "
          f"zero={int((s == 0).sum())} negative={int((s < 0).sum())} nan={int(s.isna().sum())}")
same_cur = df.payment_currency.eq(df.receiving_currency)
diff_amt_same_cur = int((same_cur & ~np.isclose(df.amount_paid, df.amount_received)).sum())
print("same currency but paid != received:", diff_amt_same_cur)
cross = ~same_cur
print("cross-currency rows:", int(cross.sum()), "| laundering:", int(df.label[cross].sum()))
out["amounts"] = {
    "amount_paid_min": float(df.amount_paid.min()), "amount_paid_max": float(df.amount_paid.max()),
    "zero_amount_paid": int((df.amount_paid == 0).sum()),
    "negative_amount_paid": int((df.amount_paid < 0).sum()),
    "same_currency_amount_mismatch": diff_amt_same_cur,
    "cross_currency_rows": int(cross.sum()),
    "cross_currency_laundering": int(df.label[cross].sum()),
}

# Implied exchange rate USD per unit of currency X, from cross-currency rows
# where one side is US Dollar.  rate = USD amount / X amount.
section("IMPLIED FX vs US DOLLAR (from cross-currency rows)")
x = df[cross & (df.amount_paid > 0) & (df.amount_received > 0)]
paid_usd = x[x.payment_currency == "US Dollar"]
recv_usd = x[x.receiving_currency == "US Dollar"]
r1 = (paid_usd.amount_paid / paid_usd.amount_received).groupby(paid_usd.receiving_currency)
r2 = (recv_usd.amount_received / recv_usd.amount_paid).groupby(recv_usd.payment_currency)
fx = pd.concat([r1.agg(["size", "median", "min", "max"]).add_prefix("usd_paid_"),
                r2.agg(["size", "median", "min", "max"]).add_prefix("usd_recv_")], axis=1)
fx["spread_%"] = (100 * (fx[["usd_paid_max", "usd_recv_max"]].max(axis=1)
                         / fx[["usd_paid_min", "usd_recv_min"]].min(axis=1) - 1)).round(3)
print(fx.round(6).to_string())
out["implied_fx_usd_per_unit"] = {
    k: {"n": int(np.nansum([r.usd_paid_size, r.usd_recv_size])),
        "median": float(np.nanmedian([r.usd_paid_median, r.usd_recv_median])),
        "spread_pct": float(r["spread_%"])}
    for k, r in fx.iterrows()}
cur_all = set(df.payment_currency) | set(df.receiving_currency)
out["currencies_without_usd_rate"] = sorted(cur_all - set(fx.index) - {"US Dollar"})
print("currencies with no direct USD cross rows:", out["currencies_without_usd_rate"])

# ------------------------------------------------------------------ patterns
section("PATTERNS FILE")
types, n_pat_rows = Counter(), 0
with open(C.PATTERNS_FILE, encoding="utf-8") as fh:
    for line in fh:
        m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)", line)
        if m:
            types[m.group(1)] += 1
        elif line[:2] == "20":
            n_pat_rows += 1
print(dict(types), "| transaction lines in patterns:", n_pat_rows,
      "| labelled laundering rows:", n_pos)
out["patterns"] = {"attempts_by_type": dict(types), "transaction_lines": n_pat_rows}

# ------------------------------------------------------------------ save
C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
target = C.METRICS_DIR / "dataset_inspection.json"
target.write_text(json.dumps(out, indent=2, default=str))
print("\nsaved", target)
