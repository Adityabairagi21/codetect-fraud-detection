"""
Generates CoDetect_Financial_Fraud_Detection.ipynb from the cell sources below.

Run:  venv\\Scripts\\python.exe scripts\\build_notebook.py
Then execute:
      venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace ^
          --ExecutePreprocessor.timeout=-1 CoDetect_Financial_Fraud_Detection.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# =====================================================================
md(r'''
# CoDetect – Financial Fraud Detection With Anomaly Feature Detection
### Stage 1: data, features, two Isolation Forests, evaluation and saved artifacts

**Research question.** Do relationship features from the transaction graph change how well an
unsupervised anomaly detector separates money-laundering transactions from normal ones,
compared with transaction features alone?

* **Model 1 – Transaction-only Isolation Forest.** Features of the transaction plus the sender's and receiver's recent activity.
* **Model 2 – Graph-enhanced Isolation Forest.** The same features plus graph-derived features (degrees, counterparties, activity, volume).

**Data.** IBM *Transactions for Anti-Money Laundering* (HI-Small). It is synthetic, comes from IBM's multi-agent
simulation, carries labels, and is licensed CDLA-Sharing-1.0.

**Ground rules followed throughout**
* The laundering label is used **only for evaluation** (and for highlighting in exploratory charts). It is never a model input.
* Every history feature of a transaction uses only transactions from the **previous 24 hours, strictly earlier minutes**.
* The split is chronological. Preprocessing, exchange rates, score normalisation and risk cut-points are fitted on training data only.
* Model settings are chosen on validation data. The test days are scored **once**, at the end.
* An anomaly score measures **how unusual** a transaction is. It is **not** a probability of fraud.

All feature, scoring and explanation logic is imported from `services/`, so Flask (Stage 2) runs exactly the same code.
''')

md(r'''
## 1. Project setup
Set `CODETECT_RUN_MODE=full` in the environment, or `RUN_MODE` in `config.py`, to train on every training row.
Quick mode trains each forest on a fixed random sample of 300,000 training rows and draws graphs from a 3-hour window.
Features are always computed on every row in both modes.
''')
code(r'''
import os, sys, platform, json, time, warnings
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "config.py").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import sklearn, joblib, networkx as nx, matplotlib, seaborn as sns, plotly
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from IPython.display import display, Markdown
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             roc_curve, precision_score, recall_score, f1_score, confusion_matrix)

import config as C
from services import feature_engineering as fe
from services import modeling as md
from services.fraud_inference import FraudInference, ARTIFACT_FILES, sha256

RUN_MODE = os.environ.get("CODETECT_RUN_MODE", C.RUN_MODE)
MODE = C.RUN_MODES[RUN_MODE]
SEED = C.RANDOM_SEED
rng = np.random.default_rng(SEED)
for d in (C.DATA_PROCESSED, C.MODELS_DIR, C.FIGURES_DIR, C.METRICS_DIR, C.GRAPHS_DIR, C.REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
NOTEBOOK_START = time.perf_counter()

if not C.TRANSACTIONS_FILE.exists():
    raise FileNotFoundError(
        f"{C.TRANSACTIONS_FILE} not found. Download HI-Small_Trans.csv as described in README section 2.")

versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "networkx": nx.__version__, "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__, "plotly": plotly.__version__}
print(json.dumps(versions, indent=2))
print("dataset file :", C.TRANSACTIONS_FILE.relative_to(ROOT),
      f"({C.TRANSACTIONS_FILE.stat().st_size / 1e6:.1f} MB)")
print("run mode     :", RUN_MODE, MODE)
print("random seed  :", SEED)
print("feature ver. :", fe.FEATURE_VERSION, "| history window:", fe.HISTORY_WINDOW_MINUTES, "minutes")
''')

code(r'''
# Chart style: recessive grid/axes, thin marks, fixed colour roles.
# Two-series charts use categorical slots 1-2 (blue, orange); risk levels use the reserved
# status colours and are always labelled with text, never colour alone.
PAL = {"s1": "#2a78d6", "s2": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
LABEL_COLORS = {"Normal": PAL["s1"], "Laundering": PAL["s2"]}
MODEL_COLORS = {"transaction_only": PAL["s1"], "graph_enhanced": PAL["s2"]}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2b2b"])

plt.rcParams.update({
    "figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"], "savefig.facecolor": PAL["surface"],
    "axes.edgecolor": PAL["axis"], "axes.labelcolor": PAL["ink2"], "axes.titlecolor": PAL["ink"],
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": PAL["muted"], "ytick.color": PAL["muted"],
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": PAL["grid"],
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2, "figure.dpi": 100})
FIGS = []

def save(fig, name):
    """Save to outputs/figures and show."""
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    FIGS.append(path.name)
    plt.show()

def thousands(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

def usd_log_ticks(ax, axis="x"):
    labels = {-6: "$0.000001", -3: "$0.001", 0: "$1", 3: "$1k", 6: "$1M", 9: "$1B", 12: "$1T"}
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mticker.FixedLocator(list(labels)))
    a.set_major_formatter(mticker.FuncFormatter(lambda v, _: labels.get(int(round(v)), "")))
''')

# =====================================================================
md(r'''
## 2. Dataset loading and inspection
The raw header repeats the column name `Account` (sender and receiver), so pandas renames the second one `Account.1`.
Bank codes have leading zeros, so bank and account columns are read as text.
The mapping from raw names to canonical names is configurable in `config.COLUMN_MAP`.
''')
code(r'''
with open(C.TRANSACTIONS_FILE, encoding="utf-8") as fh:
    raw_header = fh.readline().strip().split(",")
print("raw header:", raw_header)

t0 = time.perf_counter()
raw = fe.load_transactions(C.TRANSACTIONS_FILE, C.COLUMN_MAP)
print(f"loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns in {time.perf_counter() - t0:.1f}s")
display(pd.Series(C.COLUMN_MAP, name="raw column").rename_axis("canonical name").to_frame())
display(raw.dtypes.rename("dtype").to_frame().T)
display(raw.head(6))
''')
code(r'''
ts_probe = pd.to_datetime(raw["timestamp"], format=fe.TIMESTAMP_FORMAT, errors="coerce")
inspection = pd.Series({
    "rows": len(raw),
    "missing values (all columns)": int(raw.isna().sum().sum()),
    "exact duplicate rows": int(raw.duplicated().sum()),
    "unparseable timestamps": int(ts_probe.isna().sum()),
    "rows out of time order vs previous row": int((ts_probe.diff() < pd.Timedelta(0)).sum()),
    "max rows sharing one minute": int(ts_probe.value_counts().max()),
    "first timestamp": ts_probe.min(), "last timestamp": ts_probe.max(),
    "non-positive amounts": int(((raw["amount_paid"] <= 0) | (raw["amount_received"] <= 0)).sum()),
}, name="value").to_frame()
display(inspection)
display(raw.isna().sum().rename("missing").to_frame().T)
del ts_probe
''')

# =====================================================================
md(r'''
## 3. Data cleaning and preprocessing decisions
| Issue found | Decision |
|---|---|
| Rows are not in time order | Sort by minute. History features rely on time, not on file order |
| Minute-resolution ties (up to ~11k rows in one minute) | Rows in the same minute cannot see each other (strictly-earlier rule) |
| An account id is unique only within its bank | Account key = `bank|account` |
| 9 exact duplicate rows, none labelled laundering | Dropped. They cannot be told apart from repeated payments, and there are only 9 |
| 15 currencies, including Bitcoin | Converted to USD at **rates implied by the data's own cross-currency rows**, fitted on warm-up and training days only |
| Invalid timestamps, amounts or ids | Dropped and counted. Nothing is repaired or invented |
| Categorical attributes (payment format, currency) | Kept as text here. One-hot encoded inside the model preprocessing, fitted on training data only |

**Chronological split.**
* **Warm-up, Sep 1:** builds 24h history only. It is never trained on or scored.
* **Train:** Sep 2–6.
* **Validation:** Sep 7–8.
* **Test:** Sep 9–10.
* **Excluded:** Sep 11–18. This is a simulation tail of 1,108 rows, about 58% laundering. It would inflate any metric, so it is shown only in the exploratory analysis.
''')
code(r'''
t0 = time.perf_counter()
tx, clean_report = fe.clean_transactions(raw, reference_prefix="HI-Small")
del raw
tx["split"] = fe.assign_time_split(tx["timestamp"], C.SPLIT_WINDOWS)
print(f"cleaned in {time.perf_counter() - t0:.1f}s")
display(pd.Series(clean_report, name="count").to_frame().T)

fx = fe.fit_fx_rates(tx[tx["split"].isin(C.FX_FIT_SPLITS)])
assert not fx["currencies_without_rate"], fx["currencies_without_rate"]
tx["amount_usd"] = fe.to_usd(tx["amount_paid"], tx["payment_currency"], fx)
display(pd.DataFrame({"USD per unit": fx["rates"], "cross-currency observations": fx["n_observations"]})
        .sort_values("USD per unit").style.format({"USD per unit": "{:,.6f}", "cross-currency observations": "{:,.0f}"}))

SPLIT_ORDER = ["warmup", "train", "validation", "test", "excluded"]
split_tbl = (tx.groupby("split").agg(rows=("label", "size"), laundering=("label", "sum"),
                                     first=("timestamp", "min"), last=("timestamp", "max"))
               .reindex(SPLIT_ORDER))
split_tbl["laundering_rate_%"] = (100 * split_tbl["laundering"] / split_tbl["rows"]).round(4)
display(split_tbl)
''')
code(r'''
# Export the cleaned dataset (labels kept in their own column for evaluation).
clean_cols = ["transaction_reference", "timestamp", "sender_bank", "sender_account", "receiver_bank",
              "receiver_account", "sender_id", "receiver_id", "amount_paid", "payment_currency",
              "amount_received", "receiving_currency", "payment_format", "amount_usd", "split", "label"]
clean_path = C.DATA_PROCESSED / "cleaned_transactions.csv.gz"
t0 = time.perf_counter()
tx[clean_cols].to_csv(clean_path, index=False, compression={"method": "gzip", "compresslevel": 1})
print(f"saved {clean_path.relative_to(ROOT)} ({clean_path.stat().st_size / 1e6:.0f} MB) "
      f"in {time.perf_counter() - t0:.0f}s")
''')

# =====================================================================
md(r'''
## 4. Exploratory data analysis
These charts use all cleaned rows, including the excluded tail (it is marked where relevant).
Labels appear here only to describe the data.
''')
code(r'''
y_all = tx["label"].astype(int)
accounts_all = pd.concat([tx["sender_id"], tx["receiver_id"]])
overview = pd.Series({
    "transactions": f"{len(tx):,}",
    "distinct accounts (bank|account)": f"{accounts_all.nunique():,}",
    "distinct banks": f"{pd.concat([tx['sender_bank'], tx['receiver_bank']]).nunique():,}",
    "labelled laundering": f"{int(y_all.sum()):,}",
    "laundering rate": f"{100 * y_all.mean():.4f}%",
    "self-transfers": f"{int((tx['sender_id'] == tx['receiver_id']).sum()):,}",
    "cross-currency transactions": f"{int((tx['payment_currency'] != tx['receiving_currency']).sum()):,}",
    "period": f"{tx['timestamp'].min()} -> {tx['timestamp'].max()}",
}, name="value").to_frame()
display(overview)
''')
code(r'''
daily = tx.groupby(tx["timestamp"].dt.normalize()).agg(rows=("label", "size"), laundering=("label", "sum"))
daily["rate"] = 100 * daily["laundering"] / daily["rows"]
excluded_day = daily.index >= pd.Timestamp(C.SPLIT_WINDOWS["test"][1])

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
half = pd.Timedelta(hours=12)
axes[0].bar(daily.index, daily["rows"], width=0.75, color=PAL["s1"])
axes[0].set_title("Daily transactions, with the chronological split")
axes[0].set_ylabel("transactions")
axes[0].set_ylim(0, daily["rows"].max() * 1.18)
thousands(axes[0])
axes[1].plot(daily.index, daily["rate"], color=PAL["s2"], marker="o", markersize=6)
axes[1].set_yscale("log")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
axes[1].set_title("Share of each day's transactions labelled laundering (log scale)")
axes[1].set_ylabel("laundering rate")
tail_start = pd.Timestamp(C.SPLIT_WINDOWS["test"][1]) - half
for ax in axes:
    ax.axvspan(tail_start, daily.index.max() + half, color=PAL["grid"], alpha=0.6, lw=0)
axes[0].text(tail_start + (daily.index.max() + half - tail_start) / 2, daily["rows"].max() * 0.55,
             f"excluded simulation tail\n{int(daily.loc[excluded_day, 'rows'].sum()):,} transactions in 8 days\n"
             f"({int(daily.loc[excluded_day, 'laundering'].sum()):,} labelled laundering)",
             ha="center", va="center", fontsize=9, color=PAL["ink2"])
for name, (start, end) in C.SPLIT_WINDOWS.items():
    s, e = pd.Timestamp(start) - half, pd.Timestamp(end) - half
    for ax in axes:
        ax.axvline(s, color=PAL["muted"], lw=0.8, ls="--")
    axes[0].text(s + (e - s) / 2, daily["rows"].max() * 1.1, name, ha="center", va="center",
                 color=PAL["ink2"], fontsize=9)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.tight_layout()
save(fig, "eda_daily_trend")
''')
code(r'''
log_amt = np.log10(tx["amount_usd"])
bins = np.linspace(np.floor(log_amt.min()), np.ceil(log_amt.max()), 110)
fig, ax = plt.subplots(figsize=(11, 4.2))
for name, mask in (("Normal", y_all == 0), ("Laundering", y_all == 1)):
    ax.hist(log_amt[mask], bins=bins, density=True, histtype="step", lw=2,
            color=LABEL_COLORS[name], label=f"{name} (n = {int(mask.sum()):,})")
usd_log_ticks(ax)
ax.set_xlabel("amount in USD (log scale)")
ax.set_ylabel("density")
ax.set_title("Transaction amount distribution, normal vs labelled laundering")
ax.legend(loc="upper left")
save(fig, "eda_amount_distribution")
display(tx.groupby(y_all.map({0: "Normal", 1: "Laundering"}))["amount_usd"]
          .describe(percentiles=[.25, .5, .75, .99]).style.format("{:,.2f}"))
''')
code(r'''
pf = tx.groupby("payment_format").agg(rows=("label", "size"), laundering=("label", "sum")).sort_values("rows")
pf["rate"] = 100 * pf["laundering"] / pf["rows"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].barh(pf.index, pf["rows"], color=PAL["s1"], height=0.6)
axes[0].set_title("Transactions by payment format")
thousands(axes[0], "x")
for i, v in enumerate(pf["rows"]):
    axes[0].text(v, i, f" {v:,.0f}", va="center", fontsize=8, color=PAL["ink2"])
axes[1].barh(pf.index, pf["rate"], color=PAL["s2"], height=0.6)
axes[1].set_title("Share labelled laundering, by payment format")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}%"))
for i, (v, n) in enumerate(zip(pf["rate"], pf["laundering"])):
    axes[1].text(v, i, f" {v:.3f}% ({n:,})", va="center", fontsize=8, color=PAL["ink2"])
axes[0].set_xlim(0, pf["rows"].max() * 1.25)
axes[1].set_xlim(0, pf["rate"].max() * 1.45)
fig.tight_layout()
save(fig, "eda_payment_format")
display(pf.sort_values("rows", ascending=False).style.format({"rows": "{:,}", "laundering": "{:,}", "rate": "{:.4f}"}))
''')
code(r'''
sent = tx["sender_id"].value_counts()
recv = tx["receiver_id"].value_counts()
display(pd.DataFrame({"transactions sent per sender": sent.describe(percentiles=[.5, .9, .99]),
                      "transactions received per receiver": recv.describe(percentiles=[.5, .9, .99])}).round(2))

fig, ax = plt.subplots(figsize=(9, 4.4))
for name, s, col in (("senders: transactions sent", sent, PAL["s1"]),
                     ("receivers: transactions received", recv, PAL["s2"])):
    k = np.sort(s.to_numpy())
    ccdf = 1.0 - np.arange(len(k)) / len(k)
    ax.loglog(k, ccdf, color=col, lw=2, label=name)
ax.set_xlabel("transactions per account (log)")
ax.set_ylabel("share of accounts with at least this many (log)")
ax.set_title("Account activity is heavy-tailed: a few accounts are very busy")
ax.legend()
save(fig, "eda_account_activity_ccdf")
''')
code(r'''
p99 = tx["amount_usd"].quantile(0.99)
hv = tx["amount_usd"] >= p99
display(pd.DataFrame({
    "transactions": [int(hv.sum()), int((~hv).sum())],
    "laundering": [int(y_all[hv].sum()), int(y_all[~hv].sum())],
    "laundering_rate_%": [100 * y_all[hv].mean(), 100 * y_all[~hv].mean()]},
    index=[f"top 1% by amount (>= USD {p99:,.0f})", "remaining 99%"]).round(4))
print("Ten largest transactions (USD equivalent):")
display(tx.nlargest(10, "amount_usd")[["transaction_reference", "timestamp", "sender_id", "receiver_id",
                                        "amount_paid", "payment_currency", "amount_usd", "payment_format", "label"]])
''')
code(r'''
activity = (pd.concat([tx["sender_id"], tx["receiver_id"]]).value_counts().head(15)
              .rename("transactions (sent + received)"))
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(activity.index[::-1], activity.values[::-1], color=PAL["s1"], height=0.6)
thousands(ax, "x")
ax.set_title("Most active accounts (bank|account), whole period")
ax.set_xlabel("transactions sent + received")
save(fig, "eda_most_active_accounts")
''')

# =====================================================================
md(r'''
## 5. Restricting to the modelled period
From here on only the warm-up, train, validation and test days are used. Labels go into the separate array `y`.
''')
code(r'''
modelled = tx[tx["split"] != "excluded"].reset_index(drop=True)
del tx, y_all, accounts_all, log_amt
y = modelled["label"].astype(int).to_numpy()          # evaluation only
split = modelled["split"].to_numpy()
idx_train = np.flatnonzero(split == "train")
idx_val = np.flatnonzero(split == "validation")
idx_test = np.flatnonzero(split == "test")
print(f"modelled rows {len(modelled):,} | train {len(idx_train):,} | validation {len(idx_val):,} | test {len(idx_test):,}")
''')

# =====================================================================
md(r'''
## 6. Transaction feature engineering  &  8. Graph feature engineering
Both feature groups come from a single call to `services.feature_engineering.compute_features`.
History features use the trailing window **[m − 1440, m − 1] minutes**:
* Transactions in the same minute are invisible to each other.
* Future transactions are never used.
* The label is never read.

A fixed 24-hour window was chosen over running totals since Sep 1. Running totals grow every day, so validation and test
transactions would look abnormal simply because they come later (feature version 1 showed the median sender count going
16 → 35 → 39 across train/validation/test). The 24h window keeps the splits comparable (table below).

The graph features are defined on the directed transaction multigraph of the window:
* **in/out-degree** = distinct counterparties, excluding the account itself;
* **counterparties** = distinct partners in either direction;
* **transaction count** and **volume** = every edge in or out. A self-transfer counts once in each direction.
''')
code(r'''
t0 = time.perf_counter()
F = fe.compute_features(modelled, fx)
print(f"features for {len(F):,} transactions in {time.perf_counter() - t0:.1f}s; "
      f"missing values: {int(F.isna().sum().sum())}")
assert "label" not in F.columns and len(F) == len(modelled)
schema = pd.DataFrame(fe.FEATURE_SCHEMA)[["name", "group", "kind", "transform", "description"]]
schema["used by"] = np.where(schema["group"] == "graph", "Model 2", "Models 1 + 2")
display(schema)
display(F.head(5))
''')
code(r'''
# Leakage / consistency spot-check on the full data: score one hour of the test period
# against a separately built 24h history tail; it must equal the one-pass features.
probe_start = pd.Timestamp("2022-09-09 12:00")
pm = ((modelled["timestamp"] >= probe_start) & (modelled["timestamp"] < probe_start + pd.Timedelta(hours=1))).to_numpy()
hist = fe.history_tail(modelled[modelled["timestamp"] < probe_start])
F_probe = fe.compute_features(modelled[pm], fx, history=hist)
num_cols = [f["name"] for f in fe.FEATURE_SCHEMA if f["kind"] != "categorical"]
rel = ((F_probe[num_cols] - F.loc[pm, num_cols]).abs() / (F.loc[pm, num_cols].abs() + 1)).max().max()
assert (F_probe[["payment_format", "payment_currency"]].values == F.loc[pm, ["payment_format", "payment_currency"]].values).all()
print(f"{int(pm.sum()):,} probe transactions; history rows {len(hist):,}; max relative difference {rel:.2e}")
assert rel < 1e-9

drift = F[num_cols].assign(split=split).groupby("split").median().T[["warmup", "train", "validation", "test"]]
display(drift.style.format("{:,.2f}").set_caption("Median feature value per split (warm-up has no full 24h history yet)"))
''')

# =====================================================================
md(r'''
## 7. Financial transaction graph construction (NetworkX)
* Accounts are nodes. Each transaction is its own directed edge (`MultiDiGraph`), so repeated payments between
  the same accounts stay as separate edges with their own amount, time and reference.
* The full period has about 5 million edges, so the descriptive graph uses a bounded window (3 hours in quick
  mode, one day in full mode) instead of the whole network.
''')
code(r'''
GRAPH_WINDOW = (("2022-09-05 09:00", "2022-09-05 12:00") if RUN_MODE == "quick"
                else ("2022-09-05 00:00", "2022-09-06 00:00"))
gw = modelled[(modelled["timestamp"] >= GRAPH_WINDOW[0]) & (modelled["timestamp"] < GRAPH_WINDOW[1])]
t0 = time.perf_counter()
G = fe.build_transaction_graph(gw, fx)
D = nx.DiGraph(G)                                   # distinct directed relationships
D.remove_edges_from(list(nx.selfloop_edges(D)))
wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
graph_stats = pd.Series({
    "window": f"{GRAPH_WINDOW[0]} -> {GRAPH_WINDOW[1]}",
    "accounts (nodes)": f"{G.number_of_nodes():,}",
    "transactions (edges)": f"{G.number_of_edges():,}",
    "distinct sender->receiver relationships": f"{D.number_of_edges():,}",
    "edges on repeated relationships": f"{G.number_of_edges() - nx.DiGraph(G).number_of_edges():,}",
    "self-transfer edges": f"{nx.number_of_selfloops(G):,}",
    "weakly connected components": f"{len(wcc):,}",
    "largest component (accounts)": f"{wcc[0]:,} ({100 * wcc[0] / G.number_of_nodes():.1f}%)",
    "density (distinct relationships)": f"{nx.density(D):.2e}",
    "build time": f"{time.perf_counter() - t0:.1f}s",
}, name="value").to_frame()
display(graph_stats)

acct = pd.DataFrame({
    "in_degree": dict(D.in_degree()), "out_degree": dict(D.out_degree()),
    "transactions": dict(G.degree()),
    "volume_usd": pd.Series(dict(G.in_degree(weight="amount_usd"))) + pd.Series(dict(G.out_degree(weight="amount_usd"))),
}).fillna(0)
acct["unique_counterparties"] = [len(set(D.predecessors(n)) | set(D.successors(n))) for n in acct.index]
print("Most connected accounts in the window:")
display(acct.sort_values("unique_counterparties", ascending=False).head(10)
            .style.format({"volume_usd": "{:,.0f}"}))
pairs = pd.Series([(u, v) for u, v, _ in G.edges(keys=True) if u != v]).value_counts().head(10)
print("Most frequently repeated relationships (sender -> receiver: transactions):")
display(pairs.rename("transactions").to_frame())
''')
code(r'''
fig, ax = plt.subplots(figsize=(9, 4.2))
for name, col, c in (("out-degree (distinct receivers)", "out_degree", PAL["s1"]),
                     ("in-degree (distinct senders)", "in_degree", PAL["s2"])):
    vc = acct[col][acct[col] > 0].value_counts().sort_index()
    ax.loglog(vc.index, vc.values, marker="o", ms=4, lw=0, color=c, label=name)
ax.set_xlabel("degree (log)")
ax.set_ylabel("number of accounts (log)")
ax.set_title(f"Degree distribution in the graph window ({GRAPH_WINDOW[0][5:]} to {GRAPH_WINDOW[1][11:]})")
ax.legend()
save(fig, "graph_degree_distribution")
''')
md(r'''
**Representative subgraph.** This shows the neighbourhood of an account that sends labelled-laundering transactions inside the
window. Orange edges are labelled laundering. The labels are used for highlighting in this picture only.
''')
code(r'''
edge_label = dict(zip(gw["transaction_reference"], gw["label"]))
cands = gw.loc[gw["label"] == 1, "sender_id"].value_counts().index.tolist()
center = None
for cand in cands:
    E = nx.ego_graph(G, cand, radius=2, undirected=True)
    if 12 <= E.number_of_nodes() <= 70:
        center = cand
        break
if center is None:                                 # fall back to the 1-hop neighbourhood, capped
    center = cands[0] if cands else acct["transactions"].idxmax()
    nb = sorted(set(G.predecessors(center)) | set(G.successors(center)),
                key=lambda n: -G.degree(n))[:59]
    E = G.subgraph([center, *nb]).copy()
S = nx.DiGraph()
for u, v, d in E.edges(data=True):
    if u == v:
        continue
    w = S.get_edge_data(u, v, {"count": 0, "laundering": 0})
    S.add_edge(u, v, count=w["count"] + 1, laundering=w["laundering"] + edge_label.get(d["reference"], 0))
S.add_nodes_from(E.nodes())
pos = nx.spring_layout(S.to_undirected(), seed=SEED, k=0.9)

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
for side in ax.spines.values():
    side.set_visible(False)
norm_e = [(u, v) for u, v, d in S.edges(data=True) if d["laundering"] == 0]
laund_e = [(u, v) for u, v, d in S.edges(data=True) if d["laundering"] > 0]
widths = lambda es: [0.8 + 1.2 * np.log1p(S[u][v]["count"]) for u, v in es]
nx.draw_networkx_edges(S, pos, edgelist=norm_e, width=widths(norm_e), edge_color=PAL["axis"],
                       arrowsize=8, ax=ax, connectionstyle="arc3,rad=0.08")
nx.draw_networkx_edges(S, pos, edgelist=laund_e, width=widths(laund_e), edge_color=PAL["s2"],
                       arrowsize=10, ax=ax, connectionstyle="arc3,rad=0.08")
others = [n for n in S if n != center]
nx.draw_networkx_nodes(S, pos, nodelist=others, node_size=[40 + 25 * S.degree(n) for n in others],
                       node_color=PAL["s1"], edgecolors=PAL["surface"], linewidths=1.5, ax=ax)
nx.draw_networkx_nodes(S, pos, nodelist=[center], node_size=420, node_color=PAL["ink"],
                       edgecolors=PAL["surface"], linewidths=2, ax=ax)
ax.annotate(f"centre account\n{center}", pos[center], xytext=(12, 12), textcoords="offset points",
            fontsize=9, color=PAL["ink"])
ax.legend(handles=[Line2D([], [], color=PAL["axis"], lw=2, label="normal transactions"),
                   Line2D([], [], color=PAL["s2"], lw=2, label="labelled laundering"),
                   Line2D([], [], marker="o", lw=0, color=PAL["s1"], label="account (size = degree)")],
          loc="lower left")
ax.set_title(f"Transaction subgraph around a laundering-sending account "
             f"({S.number_of_nodes()} accounts, {E.number_of_edges()} transactions)")
save(fig, "graph_representative_subgraph")

# interactive version
ex, ey, lx, ly = [], [], [], []
for u, v in S.edges():
    xs, ys = (lx, ly) if S[u][v]["laundering"] else (ex, ey)
    xs += [pos[u][0], pos[v][0], None]; ys += [pos[u][1], pos[v][1], None]
fig_i = go.Figure()
fig_i.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=PAL["axis"], width=1.5),
                           name="normal", hoverinfo="skip"))
fig_i.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=PAL["s2"], width=2.5),
                           name="labelled laundering", hoverinfo="skip"))
nodes = list(S.nodes())
fig_i.add_trace(go.Scatter(
    x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name="account",
    marker=dict(size=[18 if n == center else 8 + 2 * S.degree(n) for n in nodes],
                color=[PAL["ink"] if n == center else PAL["s1"] for n in nodes],
                line=dict(color=PAL["surface"], width=1.5)),
    text=[f"{n}<br>in-degree {S.in_degree(n)}, out-degree {S.out_degree(n)}" for n in nodes],
    hoverinfo="text"))
fig_i.update_layout(title="CoDetect - transaction subgraph (hover for accounts)", template="plotly_white",
                    showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False),
                    plot_bgcolor=PAL["surface"], paper_bgcolor=PAL["surface"])
fig_i.write_html(C.GRAPHS_DIR / "representative_subgraph.html", include_plotlyjs="cdn")
print("interactive graph:", (C.GRAPHS_DIR / "representative_subgraph.html").relative_to(ROOT))
''')
md(r'''
**Bounded account-relationship similarity (illustration).** This is the Jaccard similarity of the counterparty sets of the
15 most connected accounts in the window. It is deliberately small; a dense all-account matrix is never built.
''')
code(r'''
top = acct.sort_values("unique_counterparties", ascending=False).head(15).index
nbrs = {a: set(D.predecessors(a)) | set(D.successors(a)) for a in top}
J = pd.DataFrame([[len(nbrs[a] & nbrs[b]) / max(len(nbrs[a] | nbrs[b]), 1) for b in top] for a in top],
                 index=[a[-9:] for a in top], columns=[a[-9:] for a in top])
fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(J, cmap=SEQ, vmin=0, vmax=1, square=True, linewidths=1, linecolor=PAL["surface"],
            cbar_kws={"label": "Jaccard similarity of counterparties", "shrink": 0.7}, ax=ax)
ax.set_title("Counterparty overlap among the 15 most connected accounts")
ax.set_xlabel("account (last 9 characters)"); ax.set_ylabel("")
save(fig, "graph_similarity_matrix")
''')

# =====================================================================
md(r'''
### 8.1 Graph features vs NetworkX
For a few query minutes, the 24h history graph is built with NetworkX and its features are recomputed directly.
They must match the vectorised features exactly (the unit tests check the same thing on real subsets).
''')
code(r'''
check_minutes = np.sort(rng.choice(modelled.loc[idx_val, "minute"].unique(), size=2, replace=False))
mism, checked = 0, 0
for m in check_minutes:
    Gm = fe.window_graph(modelled, fx, int(m))
    rows = np.flatnonzero(modelled["minute"].to_numpy() == m)[:15]
    for i in rows:
        ref = fe.graph_features_networkx(Gm, modelled.at[i, "sender_id"], modelled.at[i, "receiver_id"])
        for k, v in ref.items():
            checked += 1
            mism += not np.isclose(F.at[i, k], v, rtol=1e-9, atol=1e-6)
    del Gm
print(f"{checked} feature values compared against NetworkX at {len(check_minutes)} minutes: {mism} mismatches")
assert mism == 0
''')
code(r'''
sample_idx = rng.choice(idx_train, size=min(150_000, len(idx_train)), replace=False)
corr = F.iloc[sample_idx][num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=DIV, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            linecolor=PAL["surface"], cbar_kws={"label": "Spearman correlation", "shrink": 0.7}, ax=ax)
ax.set_title("Correlation between numerical features (training sample)")
save(fig, "features_correlation")
''')

# =====================================================================
md(r'''
## 9. Model 1: transaction-only Isolation Forest  &  10. Model 2: graph-enhanced Isolation Forest
* **Preprocessing** (fitted on training rows only): `log1p` + standardisation for skewed counts and amounts,
  and standardisation for the other numerics. Payment format, currency and the two flags are handled by one of the
  three candidate encodings below.
* **Training data:** training-period rows without labels. This is fully unsupervised, so the few laundering rows in the
  training days are not removed; a real deployment would not know which they are.
* **Search, the same for both models:** 3 encodings × 4 settings (`max_samples` × `max_features`, 200 trees). The
  combination with the best **validation PR-AUC** is kept, so each model ends with one configuration.

| Encoding | Payment format / currency | Cross-currency & self-transfer flags |
|---|---|---|
| `onehot` | one-hot columns | 0/1 |
| `frequency` | training frequency of the value | 0/1 |
| `numeric_only` | not used | not used |

> **Revision note (transparency).** The first version of this notebook used `onehot` only.
> Its validation results showed a structural weakness. Isolation Forest isolates *rare values* in low-cardinality
> columns first, so rare currencies, cross-currency payments and self-transfers filled the top of the ranking.
> Labelled laundering looks ordinary on exactly those columns: 0 cross-currency, almost no self-transfers, mostly ACH.
> The Critical level held 0 laundering transactions in both models.
>
> That first version **had already scored the test days once** before the run was stopped. Those test results were
> archived **unopened** in `outputs/archive/v1_onehot/`. The encoding search was then added, decided from validation
> evidence only. Both versions' test results are reported in Section 16.
''')
code(r'''
N_ESTIMATORS = 200
GRID = [dict(max_samples=256, max_features=1.0), dict(max_samples=256, max_features=0.6),
        dict(max_samples=2048, max_features=1.0), dict(max_samples=2048, max_features=0.6)]
n_fit = MODE["train_sample_rows"]
idx_fit = (np.sort(rng.choice(idx_train, size=n_fit, replace=False))
           if n_fit and n_fit < len(idx_train) else idx_train)
y_val, y_test = y[idx_val], y[idx_test]
print(f"fitting rows: {len(idx_fit):,} of {len(idx_train):,} training rows "
      f"({int(y[idx_fit].sum())} of them labelled laundering - not used)")

def train_model(name):
    feats = md.MODEL_FEATURES[name]
    F_fit, F_val = F.iloc[idx_fit][feats], F.iloc[idx_val][feats]
    rows, best = [], None
    for enc in md.ENCODINGS:
        pre = md.build_preprocessor(feats, enc).fit(F_fit)
        X_fit, X_val = pre.transform(F_fit), pre.transform(F_val)
        for i, cfg in enumerate(GRID):
            t0 = time.perf_counter()
            m = md.fit_isolation_forest(X_fit, n_estimators=N_ESTIMATORS, seed=SEED, **cfg)
            s = md.raw_anomaly_score(m, X_val)
            row = {"model": md.MODEL_LABELS[name], "encoding": enc, "config": i, **cfg,
                   "encoded_columns": X_fit.shape[1],
                   "val_pr_auc": average_precision_score(y_val, s), "val_roc_auc": roc_auc_score(y_val, s),
                   "seconds": round(time.perf_counter() - t0, 1)}
            rows.append(row)
            if best is None or row["val_pr_auc"] > best["row"]["val_pr_auc"]:
                best = {"row": row, "model": m, "pre": pre, "val_raw": s,
                        "fit_raw": md.raw_anomaly_score(m, X_fit)}
    r = best["row"]
    print(f"{md.MODEL_LABELS[name]}: selected encoding '{r['encoding']}', config {r['config']} "
          f"({r['encoded_columns']} encoded columns)")
    return {"pre": best["pre"], "model": best["model"], "encoding": r["encoding"],
            "input_features": md.input_features(feats, r["encoding"]),
            "config": {k: r[k] for k in ("max_samples", "max_features")},
            "val_raw": best["val_raw"], "fit_raw": best["fit_raw"],
            "grid": pd.DataFrame(rows), "encoded_features": list(best["pre"].get_feature_names_out())}

trained = {}
''')
code(r'''
trained["transaction_only"] = train_model("transaction_only")
display(trained["transaction_only"]["grid"].style.format({"val_pr_auc": "{:.4f}", "val_roc_auc": "{:.4f}"}))
''')
code(r'''
trained["graph_enhanced"] = train_model("graph_enhanced")
display(trained["graph_enhanced"]["grid"].style.format({"val_pr_auc": "{:.4f}", "val_roc_auc": "{:.4f}"}))
''')

# =====================================================================
md(r'''
## 11. Anomaly scoring
* **raw score** = −`score_samples`, so a higher value means more abnormal.
* **anomaly score (0–100)** = the percentile of the raw score among **training-period** raw scores. A score of 99 means
  "more unusual than 99% of training transactions". It ranks rarity and is **not** a fraud probability.

The normalisation quantiles are saved and reused unchanged at inference.
''')
code(r'''
normalizers = {m: md.fit_score_normalizer(t["fit_raw"]) for m, t in trained.items()}
for m, t in trained.items():
    t["val_score"] = md.normalize_scores(t["val_raw"], normalizers[m])
    t["fit_score"] = md.normalize_scores(t["fit_raw"], normalizers[m])
    assert np.all(np.diff(t["val_score"][np.argsort(t["val_raw"])]) >= 0), "normalisation must be monotone"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
bins = np.linspace(0, 100, 51)
for ax, (m, t) in zip(axes, trained.items()):
    for name, mask in (("Normal", y_val == 0), ("Laundering", y_val == 1)):
        ax.hist(t["val_score"][mask], bins=bins, density=True, histtype="step", lw=2,
                color=LABEL_COLORS[name], label=f"{name} (n = {int(mask.sum()):,})")
    ax.set_title(f"{md.MODEL_LABELS[m]}: validation scores")
    ax.set_xlabel("anomaly score (training percentile, 0-100)")
    ax.set_yscale("log")
axes[0].set_ylabel("density (log)")
axes[0].legend(loc="upper left")
fig.tight_layout()
save(fig, "scores_distribution_validation")
''')

# =====================================================================
md(r'''
## 12. Risk classification
The cut-points are percentiles of the training score distribution and work as an **alert budget**. They were not tuned on labels.

| Risk level | Anomaly score | Expected share of ordinary traffic |
|---|---|---|
| Low | < 90 | 90% |
| Medium | 90 – 97 | 7% |
| High | 97 – 99.5 | 2.5% |
| Critical | ≥ 99.5 | 0.5% |

**Suspicious** means High or Critical, about the top 3% most unusual transactions.
The table checks how these levels behave on validation, including the laundering rate per level.
''')
code(r'''
LEVELS = [n for n, _ in md.RISK_LEVELS]
risk_rows = []
for m, t in trained.items():
    for part, scores, labels in (("train (fitting rows)", t["fit_score"], y[idx_fit]),
                                 ("validation", t["val_score"], y_val)):
        lv = md.risk_level(scores)
        for L in LEVELS:
            mask = lv == L
            risk_rows.append({"model": md.MODEL_LABELS[m], "data": part, "risk level": L,
                              "transactions": int(mask.sum()), "share_%": 100 * mask.mean(),
                              "laundering": int(labels[mask].sum()),
                              "laundering_rate_%": 100 * labels[mask].mean() if mask.any() else np.nan})
risk_tbl = pd.DataFrame(risk_rows)
display(risk_tbl.style.format({"share_%": "{:.2f}", "laundering_rate_%": "{:.4f}", "transactions": "{:,}"}))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, m in zip(axes, trained):
    r = risk_tbl[(risk_tbl["model"] == md.MODEL_LABELS[m]) & (risk_tbl["data"] == "validation")]
    bars = ax.bar(r["risk level"], r["laundering_rate_%"], color=[RISK_COLORS[L] for L in r["risk level"]], width=0.6)
    for b, v, n in zip(bars, r["laundering_rate_%"], r["laundering"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}%\n({n:,})", ha="center", va="bottom",
                fontsize=8, color=PAL["ink2"])
    ax.set_title(f"{md.MODEL_LABELS[m]}: laundering rate by risk level")
    ax.set_yscale("log")
    ax.set_xlabel("risk level (validation)")
axes[0].set_ylabel("share labelled laundering (log)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
fig.tight_layout()
save(fig, "risk_levels_validation")
''')

# =====================================================================
md(r'''
## 13. Anomaly feature detection: which features are abnormal?
Each feature value of a flagged transaction is compared with the **training reference distribution**:
* **numeric:** tail probability = min(P(X ≤ x), P(X ≥ x));
* **categorical:** the category's frequency in training;
* **abnormality** = −log10(tail probability). A score of 3 means "a value this extreme occurs in about 1 in 1,000 training transactions".

The top three features per Suspicious transaction are reported with a plain-language sentence.
They are **abnormal indicators**. They describe what is unusual and do **not** prove why a transaction is or isn't laundering.
''')
code(r'''
feature_reference = md.fit_feature_reference(F.iloc[idx_fit], fe.ALL_FEATURES, seed=SEED)
''')

# =====================================================================
md(r'''
## 14. Model evaluation on the validation period
Both models are evaluated on **the same validation transactions**. Laundering is rare (about 0.1%), so PR-AUC is the
main measure: a random ranking scores roughly the prevalence. ROC-AUC is shown for completeness.
Precision, recall and F1 are reported at the Suspicious cut-point (score ≥ 97) and the Critical cut-point (score ≥ 99.5).
''')
code(r'''
HIGH_CUT, CRIT_CUT = dict(md.RISK_LEVELS)["High"], dict(md.RISK_LEVELS)["Critical"]

def at_cut(y_true, score, cut):
    pred = score >= cut
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "flagged": int(pred.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def evaluate(name, y_true, raw, score, part):
    row = {"model": md.MODEL_LABELS[name], "data": part, "transactions": len(y_true),
           "laundering": int(y_true.sum()), "prevalence": y_true.mean(),
           "pr_auc": average_precision_score(y_true, raw), "roc_auc": roc_auc_score(y_true, raw)}
    row.update({f"suspicious_{k}": v for k, v in at_cut(y_true, score, HIGH_CUT).items()})
    row.update({f"critical_{k}": v for k, v in at_cut(y_true, score, CRIT_CUT).items()})
    return row

def paired_bootstrap(y_true, raw_a, raw_b, n=100, seed=SEED):
    """95% interval of PR-AUC(b) - PR-AUC(a), resampling transactions with replacement."""
    r = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = r.integers(0, len(y_true), len(y_true))
        d.append(average_precision_score(y_true[i], raw_b[i]) - average_precision_score(y_true[i], raw_a[i]))
    return np.percentile(d, [2.5, 50, 97.5])

METRIC_FMT = {"prevalence": "{:.5f}", "pr_auc": "{:.4f}", "roc_auc": "{:.4f}",
              **{f"{p}_{k}": "{:.4f}" for p in ("suspicious", "critical") for k in ("precision", "recall", "f1")}}
val_metrics = pd.DataFrame([evaluate(m, y_val, t["val_raw"], t["val_score"], "validation") for m, t in trained.items()])
display(val_metrics.T)
''')
code(r'''
def curve_panels(y_true, raws, part, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, raw in raws.items():
        p, r, _ = precision_recall_curve(y_true, raw)
        axes[0].plot(r, p, color=MODEL_COLORS[m], lw=2,
                     label=f"{md.MODEL_LABELS[m]} (PR-AUC {average_precision_score(y_true, raw):.4f})")
        fpr, tpr, _ = roc_curve(y_true, raw)
        axes[1].plot(fpr, tpr, color=MODEL_COLORS[m], lw=2,
                     label=f"{md.MODEL_LABELS[m]} (ROC-AUC {roc_auc_score(y_true, raw):.4f})")
    axes[0].axhline(y_true.mean(), color=PAL["muted"], ls="--", lw=1, label=f"random ({y_true.mean():.4f})")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="recall", ylabel="precision (log)", title=f"Precision-recall, {part}")
    axes[1].plot([0, 1], [0, 1], color=PAL["muted"], ls="--", lw=1, label="random")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC, {part}")
    for ax in axes:
        ax.legend(loc="best")
    fig.tight_layout()
    save(fig, fname)

def confusion_panels(y_true, scores, part, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (m, s) in zip(axes, scores.items()):
        cm = confusion_matrix(y_true, s >= HIGH_CUT, labels=[0, 1])
        sns.heatmap(cm, annot=[[f"{v:,}" for v in row] for row in cm], fmt="", cmap=SEQ, cbar=False,
                    linewidths=2, linecolor=PAL["surface"], ax=ax,
                    xticklabels=["Normal", "Suspicious"], yticklabels=["Normal", "Laundering"],
                    norm=matplotlib.colors.LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
        ax.set_title(f"{md.MODEL_LABELS[m]} ({part})")
        ax.set_xlabel("predicted (score >= 97)")
        ax.set_ylabel("actual label")
    fig.tight_layout()
    save(fig, fname)

curve_panels(y_val, {m: t["val_raw"] for m, t in trained.items()}, "validation", "eval_curves_validation")
confusion_panels(y_val, {m: t["val_score"] for m, t in trained.items()}, "validation", "eval_confusion_validation")
''')
code(r'''
# Which features show up as top abnormal indicators? Explanations use only the inputs each model consumed.
def top_features_for(name, idx, scores):
    rows = idx[scores >= HIGH_CUT]
    tf = md.top_abnormal_features(F.iloc[rows], feature_reference, trained[name]["input_features"], k=3)
    tf["label"] = y[tf["row"].to_numpy()]
    return tf

val_top = {m: top_features_for(m, idx_val, t["val_score"]) for m, t in trained.items()}
tfg = val_top["graph_enhanced"]
share = (tfg.groupby(["feature_name", "label"])["row"].nunique()
            .unstack(fill_value=0)
            .div(tfg.groupby("label")["row"].nunique(), axis=1) * 100).rename(columns={0: "Normal", 1: "Laundering"})
share = share.reindex(share.max(axis=1).sort_values().index).tail(14)
fig, ax = plt.subplots(figsize=(10, 6))
yy = np.arange(len(share))
ax.barh(yy - 0.2, share.get("Normal", 0), height=0.38, color=LABEL_COLORS["Normal"], label="flagged, labelled normal")
ax.barh(yy + 0.2, share.get("Laundering", 0), height=0.38, color=LABEL_COLORS["Laundering"], label="flagged, labelled laundering")
ax.set_yticks(yy, [md.feature_label(f) for f in share.index])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("share of Suspicious transactions with this feature in their top 3")
ax.set_title("Most frequent abnormal indicators, graph-enhanced model (validation)")
ax.legend(loc="lower right")
save(fig, "abnormal_features_frequency")

samp = np.concatenate([rng.choice(np.flatnonzero(y_val == 0), 100_000, replace=False), np.flatnonzero(y_val == 1)])
ab, _ = md.feature_abnormality(F.iloc[idx_val[samp]], feature_reference, fe.ALL_FEATURES)
med = ab.groupby(y_val[samp]).mean().T.rename(columns={0: "Normal", 1: "Laundering"}).sort_values("Laundering")
fig, ax = plt.subplots(figsize=(10, 7))
yy = np.arange(len(med))
ax.hlines(yy, med.min(axis=1), med.max(axis=1), color=PAL["grid"], lw=2)
for name in ("Normal", "Laundering"):
    ax.plot(med[name], yy, "o", ms=8, color=LABEL_COLORS[name], label=name, mec=PAL["surface"], mew=1.5)
ax.set_yticks(yy, [f"{md.feature_label(f)}{'  [graph]' if f in fe.GRAPH_FEATURES else ''}" for f in med.index])
ax.set_xlabel("mean abnormality (-log10 tail probability vs training)")
ax.set_title("How abnormal each feature is, by actual label (validation)")
ax.legend(loc="lower right")
save(fig, "abnormal_features_by_label")
''')

# =====================================================================
md(r'''
## 15. Model comparison and selection (validation only)
The preferred approach is the model with the higher **validation PR-AUC**. A paired bootstrap of the validation
transactions gives a 95% interval for the difference.

Caveat: the bootstrap resamples individual transactions. Laundering transactions come in linked patterns, so this
interval is optimistic (too narrow).
''')
code(r'''
cmp = val_metrics.set_index("model")
diff_val = paired_bootstrap(y_val, trained["transaction_only"]["val_raw"], trained["graph_enhanced"]["val_raw"])
SELECTED = max(trained, key=lambda m: average_precision_score(y_val, trained[m]["val_raw"]))
other = "transaction_only" if SELECTED == "graph_enhanced" else "graph_enhanced"
selection = {
    "selected_model": SELECTED,
    "criterion": "highest validation PR-AUC",
    "validation_pr_auc": {m: float(average_precision_score(y_val, t["val_raw"])) for m, t in trained.items()},
    "pr_auc_difference_graph_minus_transaction_95ci": [float(v) for v in diff_val[[0, 2]]],
    "pr_auc_difference_graph_minus_transaction_median": float(diff_val[1]),
}
display(Markdown(
    f"**Selected: {md.MODEL_LABELS[SELECTED]}.** Validation PR-AUC "
    f"{selection['validation_pr_auc'][SELECTED]:.4f} vs {selection['validation_pr_auc'][other]:.4f} "
    f"({md.MODEL_LABELS[other]}). Graph-enhanced minus transaction-only: median {diff_val[1]:+.4f}, "
    f"95% bootstrap interval [{diff_val[0]:+.4f}, {diff_val[2]:+.4f}]. "
    + ("The interval excludes zero." if diff_val[0] > 0 or diff_val[2] < 0 else
       "The interval includes zero, so the two models are **not** clearly different on validation.")))
''')

# =====================================================================
md(r'''
## 16. Final evaluation on the test period (Sep 9–10)
The selection above is fixed. The test transactions are scored by both final model configurations, using the same transactions for each.

**Disclosure.** This is the **second** time the test days have been scored. The first, one-hot-only version (see the
revision note in Section 9) scored them once before it was revised. The revision was decided from validation results only,
and the first version's test numbers were archived without being viewed. Both are shown below. The first version's
test results are therefore not a selection target, and the second version's are not a fully untouched estimate.
''')
code(r'''
for m, t in trained.items():
    X_test = t["pre"].transform(F.iloc[idx_test][md.MODEL_FEATURES[m]])   # preprocessor drops unused columns
    t["test_raw"] = md.raw_anomaly_score(t["model"], X_test)
    t["test_score"] = md.normalize_scores(t["test_raw"], normalizers[m])
    del X_test
test_metrics = pd.DataFrame([evaluate(m, y_test, t["test_raw"], t["test_score"], "test") for m, t in trained.items()])
diff_test = paired_bootstrap(y_test, trained["transaction_only"]["test_raw"], trained["graph_enhanced"]["test_raw"])
comparison = pd.concat([val_metrics, test_metrics], ignore_index=True)
display(comparison.set_index(["data", "model"]).T)
display(Markdown(
    f"**Test (selected model, {md.MODEL_LABELS[SELECTED]}):** PR-AUC "
    f"{test_metrics.set_index('model').loc[md.MODEL_LABELS[SELECTED], 'pr_auc']:.4f} "
    f"(random = {y_test.mean():.5f}). Graph-enhanced minus transaction-only on test: median {diff_test[1]:+.4f}, "
    f"95% bootstrap interval [{diff_test[0]:+.4f}, {diff_test[2]:+.4f}]."))
''')
code(r'''
# Version history: first (one-hot only) vs current version, from the archived run.
V1_DIR = C.OUTPUTS_DIR / "archive" / "v1_onehot" / "metrics"
version_cmp = None
if (V1_DIR / "model_comparison.csv").exists():
    v1 = pd.read_csv(V1_DIR / "model_comparison.csv").assign(version="v1: one-hot only")
    v2 = comparison.assign(version="v2: encoding chosen on validation")
    keep = ["version", "data", "model", "pr_auc", "roc_auc", "suspicious_precision", "suspicious_recall",
            "critical_recall"]
    version_cmp = pd.concat([v1[keep], v2[keep]], ignore_index=True)
    display(version_cmp.style.format({c: "{:.4f}" for c in keep[3:]}).hide(axis="index"))
    version_cmp.to_csv(C.METRICS_DIR / "version_comparison.csv", index=False)
else:
    print("archived first-version results not found - version comparison skipped")
''')
code(r'''
curve_panels(y_test, {m: t["test_raw"] for m, t in trained.items()}, "test", "eval_curves_test")
confusion_panels(y_test, {m: t["test_score"] for m, t in trained.items()}, "test", "eval_confusion_test")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, metric, title in zip(axes, ["pr_auc", "suspicious_recall", "suspicious_precision"],
                             ["PR-AUC", "Recall at Suspicious (score >= 97)", "Precision at Suspicious (score >= 97)"]):
    xs = np.arange(2)
    for j, m in enumerate(trained):
        vals = comparison[comparison["model"] == md.MODEL_LABELS[m]].set_index("data").loc[["validation", "test"], metric]
        bars = ax.bar(xs + (j - 0.5) * 0.36, vals.values, width=0.34, color=MODEL_COLORS[m], label=md.MODEL_LABELS[m])
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=PAL["ink2"])
    ax.set_xticks(xs, ["validation", "test"])
    ax.set_title(title)
axes[0].legend(loc="upper left")
fig.suptitle("Transaction-only vs graph-enhanced Isolation Forest", x=0.01, ha="left", fontsize=13, fontweight="semibold")
fig.tight_layout()
save(fig, "model_comparison")
''')
code(r'''
test_top = top_features_for(SELECTED, idx_test, trained[SELECTED]["test_score"])
test_top.insert(0, "transaction_reference", modelled.loc[test_top["row"], "transaction_reference"].to_numpy())
print(f"Suspicious test transactions ({md.MODEL_LABELS[SELECTED]}): {test_top['row'].nunique():,}")
crit_rows = idx_test[trained[SELECTED]["test_score"] >= CRIT_CUT]
example_rows = list(crit_rows[y[crit_rows] == 1][:3]) + list(crit_rows[y[crit_rows] == 0][:2])
for r in example_rows:
    t = test_top[test_top["row"] == r]
    head = modelled.loc[r]
    display(Markdown(
        f"**{head['transaction_reference']}** · {head['timestamp']} · {head['sender_id']} → {head['receiver_id']} · "
        f"USD {F.at[r, 'amount_usd']:,.2f} · {head['payment_format']} · anomaly score "
        f"{trained[SELECTED]['test_score'][np.searchsorted(idx_test, r)]:.2f} · actual label: "
        f"{'laundering' if y[r] else 'normal'}"))
    display(t[["rank", "feature_name", "feature_value", "contribution_score", "description"]]
              .style.format({"contribution_score": "{:.2f}"}).hide(axis="index"))
''')

# =====================================================================
md(r'''
## 17. Model persistence
Saved to `models/`:
| File | Contents |
|---|---|
| `transaction_model.joblib` | Model 1 IsolationForest |
| `graph_model.joblib` | Model 2 IsolationForest |
| `preprocessing.joblib` | `{"transaction_only": ColumnTransformer, "graph_enhanced": ColumnTransformer}` |
| `scoring_reference.joblib` | FX rates, score-normalisation quantiles, training reference distributions for explanations |
| `history_tail.joblib` | Last 24h of transactions (no labels), the history context for batches that start after the data ends |
| `model_metadata.json` | Schema, feature lists, thresholds, settings, metrics, versions, dataset info and artifact checksums |

Results go to `outputs/metrics/` and `outputs/reports/`.
''')
code(r'''
joblib.dump(trained["transaction_only"]["model"], C.MODELS_DIR / ARTIFACT_FILES["transaction_model"])
joblib.dump(trained["graph_enhanced"]["model"], C.MODELS_DIR / ARTIFACT_FILES["graph_model"])
joblib.dump({m: t["pre"] for m, t in trained.items()}, C.MODELS_DIR / ARTIFACT_FILES["preprocessing"])
joblib.dump({"fx": fx, "normalizers": normalizers, "feature_reference": feature_reference},
            C.MODELS_DIR / ARTIFACT_FILES["scoring_reference"])
history_tail = fe.history_tail(modelled)
joblib.dump(history_tail, C.MODELS_DIR / ARTIFACT_FILES["history_tail"])

def to_py(x):
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return x

metadata = {
    "project": "CoDetect - Financial Fraud Detection With Anomaly Feature Detection",
    "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    "run_mode": RUN_MODE, "random_seed": SEED, "library_versions": versions,
    "dataset": {"name": C.DATASET_NAME, "file": C.TRANSACTIONS_FILE.name,
                "sha256": sha256(C.TRANSACTIONS_FILE),
                "source": "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
                "official_project": "https://github.com/IBM/AML-Data", "licence": "CDLA-Sharing-1.0",
                "column_map": C.COLUMN_MAP, "cleaning_report": clean_report,
                "split_windows": C.SPLIT_WINDOWS,
                "split_counts": split_tbl[["rows", "laundering"]].to_dict(orient="index")},
    "feature_version": fe.FEATURE_VERSION,
    "history_window_minutes": fe.HISTORY_WINDOW_MINUTES,
    "history_rule": "features of a transaction at minute m use only transactions in [m-1440, m-1]",
    "feature_schema": fe.FEATURE_SCHEMA,
    "model_features": md.MODEL_FEATURES,
    "encoding": {m: t["encoding"] for m, t in trained.items()},
    "input_features": {m: t["input_features"] for m, t in trained.items()},
    "encoded_features": {m: t["encoded_features"] for m, t in trained.items()},
    "required_input_columns": fe.REQUIRED_COLUMNS,
    "hyperparameters": {m: {"n_estimators": N_ESTIMATORS, **t["config"], "contamination": "auto",
                            "fit_rows": int(len(idx_fit))} for m, t in trained.items()},
    "hyperparameter_grid": {"encodings": list(md.ENCODINGS), "settings": GRID},
    "revision_history": [
        {"version": "v1", "change": "one-hot encoding only",
         "note": "test days scored once; results archived unopened in outputs/archive/v1_onehot before revision"},
        {"version": "v2", "change": "encoding (onehot / frequency / numeric_only) added to the validation search",
         "reason": "v1 validation: Isolation Forest isolated rare currencies, cross-currency and self-transfer "
                   "values first; labelled laundering is common-looking on those columns"}],
    "score_convention": {"raw": "-IsolationForest.score_samples (higher = more abnormal)",
                         "anomaly_score": "percentile (0-100) of raw score among training-period raw scores",
                         "note": "an anomaly score is a rarity rank, not a fraud probability"},
    "risk_levels": [{"level": n, "min_score": c} for n, c in md.RISK_LEVELS],
    "suspicious_levels": list(md.SUSPICIOUS_LEVELS),
    "abnormal_feature_method": "-log10 tail probability vs training reference sample; top 3 per suspicious transaction",
    "selected_model": SELECTED, "selection": selection,
    "combined_anomaly_score": "anomaly score of the selected model",
    "metrics": {"validation": val_metrics.to_dict(orient="records"),
                "test": test_metrics.to_dict(orient="records"),
                "test_pr_auc_difference_graph_minus_transaction_95ci": [float(diff_test[0]), float(diff_test[2])]},
    "history_tail": {"rows": int(len(history_tail)),
                     "from": str(pd.Timestamp(int(history_tail["minute"].min()) * 60, unit="s")),
                     "to": str(pd.Timestamp(int(history_tail["minute"].max()) * 60, unit="s"))},
}
metadata["artifact_sha256"] = {f: sha256(C.MODELS_DIR / f) for k, f in ARTIFACT_FILES.items() if k != "metadata"}
(C.MODELS_DIR / ARTIFACT_FILES["metadata"]).write_text(json.dumps(to_py(metadata), indent=2, default=str))

# results
pd.concat([trained[m]["grid"] for m in trained]).to_csv(C.METRICS_DIR / "hyperparameter_search.csv", index=False)
comparison.to_csv(C.METRICS_DIR / "model_comparison.csv", index=False)
risk_tbl.to_csv(C.METRICS_DIR / "risk_distribution.csv", index=False)
(C.METRICS_DIR / "metrics.json").write_text(json.dumps(to_py(metadata["metrics"] | {"selection": selection}), indent=2))
scored = []
for part, idx in (("validation", idx_val), ("test", idx_test)):
    key = "val" if part == "validation" else "test"
    scored.append(pd.DataFrame({
        "transaction_reference": modelled.loc[idx, "transaction_reference"].to_numpy(), "split": part,
        "actual_label": y[idx],
        **{f"{'transaction' if m == 'transaction_only' else 'graph'}_{kind}": trained[m][f"{key}_{kind2}"]
           for m in trained for kind, kind2 in (("raw_score", "raw"), ("anomaly_score", "score"))}}))
scored = pd.concat(scored, ignore_index=True)
for short in ("transaction", "graph"):
    scored[f"{short}_risk_level"] = md.risk_level(scored[f"{short}_anomaly_score"])
sel_short = "transaction" if SELECTED == "transaction_only" else "graph"
scored["risk_level"] = scored[f"{sel_short}_risk_level"]
scored["anomaly_status"] = md.anomaly_status(scored["risk_level"])
scored.to_csv(C.REPORTS_DIR / "scored_validation_test.csv.gz", index=False,
              compression={"method": "gzip", "compresslevel": 1})
test_top.drop(columns="row").to_csv(C.REPORTS_DIR / "test_suspicious_top_features.csv", index=False)
for f in sorted(C.MODELS_DIR.iterdir()):
    print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:8.2f} MB")
''')

# =====================================================================
md(r'''
## 18. Inference module and prediction-consistency check
`services/fraud_inference.FraudInference` reloads every artifact from disk, checks the checksums and feature version,
and scores **raw, CSV-shaped records** through the full path:
column mapping → cleaning → 24h-window features → both models → normalisation → risk level → explanations.

Here it scores one real hour from the test period (Sep 10, 12:00–13:00). The records are rebuilt in the raw CSV
format, and the 24 hours before them are passed as history. The results must match the notebook's in-memory
scores exactly.
''')
code(r'''
engine = FraudInference(C.MODELS_DIR)
b_mask = ((modelled["timestamp"] >= "2022-09-10 12:00") & (modelled["timestamp"] < "2022-09-10 13:00")).to_numpy()
batch_clean = modelled[b_mask]
raw_batch = batch_clean[list(C.COLUMN_MAP)].rename(columns=C.COLUMN_MAP)
raw_batch[C.COLUMN_MAP["timestamp"]] = batch_clean["timestamp"].dt.strftime(fe.TIMESTAMP_FORMAT)
raw_batch["transaction_reference"] = batch_clean["transaction_reference"].to_numpy()
history_ctx = modelled.loc[modelled["minute"] < int(batch_clean["minute"].min()), fe.HISTORY_COLUMNS]

t0 = time.perf_counter()
res = engine.score(raw_batch.reset_index(drop=True), history=history_ctx)
print(f"scored {len(res.results):,} raw records in {time.perf_counter() - t0:.1f}s; "
      f"history rows used: {res.history_info['history_rows_used']:,}")

nb = scored[scored["split"] == "test"].set_index("transaction_reference")
got = res.results.set_index("transaction_reference")
nb = nb.loc[got.index]
consistency = pd.Series({
    "transactions compared": len(got),
    "max |transaction raw score diff|": float(np.abs(got["transaction_raw_score"] - nb["transaction_raw_score"]).max()),
    "max |graph raw score diff|": float(np.abs(got["graph_raw_score"] - nb["graph_raw_score"]).max()),
    "max |anomaly score diff|": float(np.abs(got["combined_anomaly_score"] - nb[f"{sel_short}_anomaly_score"]).max()),
    "risk levels identical": bool((got["risk_level"].to_numpy() == nb["risk_level"].to_numpy()).all()),
    "labels passed through, not used": bool((got["actual_label"].to_numpy() == nb["actual_label"].to_numpy()).all()),
}, name="value").to_frame()
display(consistency)
assert consistency.loc["max |transaction raw score diff|", "value"] < 1e-9
assert consistency.loc["max |graph raw score diff|", "value"] < 1e-9
assert consistency.loc["risk levels identical", "value"]
display(res.results[res.results["anomaly_status"] == "Suspicious"].head(5))
display(res.explanations.head(6))

# A batch that does not start after the saved history must be given explicit history:
try:
    engine.score(raw_batch.head(20))
except fe.HistoryOverlapError as exc:
    print("expected validation message:", exc)
''')
md(r'''
### How Flask will call this module (Stage 2 contract)
```python
from services.fraud_inference import FraudInference, ArtifactError
engine = FraudInference("ml_artifacts")          # raises ArtifactError if anything is missing or incompatible
clean, report = engine.prepare(uploaded_df)      # SchemaError -> show "missing columns" message
problems = engine.validate(clean)                # e.g. unknown currency -> show message, do not score
result = engine.score(clean, already_clean=True, history=earlier_24h_from_db)
# result.results      -> Fraud_Analysis rows (transaction/graph/combined scores, risk_level, anomaly_status)
# result.explanations -> Anomaly_Features rows (feature_name, feature_value, contribution_score, description)
# High / Critical     -> Fraud_Alerts
```
''')

# =====================================================================
md(r'''
## 19. Summary
Everything below is printed from the computed results above.
''')
code(r'''
v = val_metrics.set_index("model"); t_ = test_metrics.set_index("model")
lines = [f"- Data: {len(modelled):,} modelled transactions (IBM AML HI-Small); train {len(idx_train):,}, "
         f"validation {len(idx_val):,}, test {len(idx_test):,}; laundering prevalence in test {y_test.mean():.4%}."]
for m in trained:
    L = md.MODEL_LABELS[m]
    lines.append(f"- {L}: validation PR-AUC {v.loc[L, 'pr_auc']:.4f} / ROC-AUC {v.loc[L, 'roc_auc']:.4f}; "
                 f"test PR-AUC {t_.loc[L, 'pr_auc']:.4f} / ROC-AUC {t_.loc[L, 'roc_auc']:.4f}; "
                 f"test recall at Suspicious {t_.loc[L, 'suspicious_recall']:.3f} with precision "
                 f"{t_.loc[L, 'suspicious_precision']:.4f} ({t_.loc[L, 'suspicious_flagged']:,} flagged).")
lines.append("- Encodings selected on validation: " + ", ".join(
    f"{md.MODEL_LABELS[m]} = `{t['encoding']}`" for m, t in trained.items())
    + ". The test days were scored once earlier by the one-hot-only first version (see Sections 9 and 16).")
lines.append(f"- Selected on validation: {md.MODEL_LABELS[SELECTED]}. Graph minus transaction PR-AUC, 95% bootstrap "
             f"interval: validation [{diff_val[0]:+.4f}, {diff_val[2]:+.4f}], test [{diff_test[0]:+.4f}, {diff_test[2]:+.4f}].")
lines.append(f"- Reloaded inference module reproduced the notebook scores on {len(got):,} raw records "
             f"(max raw-score difference {max(consistency.loc['max |transaction raw score diff|', 'value'], consistency.loc['max |graph raw score diff|', 'value']):.1e}).")
lines.append(f"- Figures saved: {len(FIGS)} in outputs/figures; total notebook runtime "
             f"{(time.perf_counter() - NOTEBOOK_START) / 60:.1f} min ({RUN_MODE} mode).")
display(Markdown("\n".join(lines)))
''')
md(r'''
**Reading the results responsibly**
* Isolation Forest is unsupervised. It finds *unusual* transactions, and most unusual transactions are not laundering,
  so precision at any practical alert volume is low. Low precision is the expected outcome for this method on 0.1% prevalence.
* Whether graph features help is the measured difference above, with its uncertainty. It is not an assumption.
* The bootstrap intervals resample individual transactions and ignore the linked structure of laundering patterns, so they are optimistic.
* The data is synthetic. Patterns such as "81% of laundering is ACH" come from the simulator and may not hold in real banking data.
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (CoDetect venv)", "language": "python"}
out = ROOT / "CoDetect_Financial_Fraud_Detection.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
