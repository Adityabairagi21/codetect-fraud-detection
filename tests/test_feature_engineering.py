"""
Leakage and correctness tests for services/feature_engineering.py.

Two kinds of input:
* an 8-row hand-written UNIT-TEST FIXTURE (not data, never used for results)
  whose expected feature values were worked out by hand;
* real IBM HI-Small rows for property and cross-check tests.

History rule under test: a transaction at minute m sees only transactions in
[m-1440, m-1].
"""
import numpy as np
import pandas as pd
import pytest

from services import feature_engineering as fe

FX_TOY = {"base": "US Dollar", "rates": {"US Dollar": 1.0, "Euro": 1.2}}


def toy_raw():
    rows = [  # timestamp, sender acct, receiver acct, amount, currency
        ("2022/09/01 10:00", "A", "B", 100.0, "US Dollar"),   # t0
        ("2022/09/01 10:00", "A", "C", 50.0, "US Dollar"),    # t1 same minute as t0
        ("2022/09/01 10:01", "A", "B", 200.0, "US Dollar"),   # t2
        ("2022/09/01 10:02", "B", "A", 10.0, "US Dollar"),    # t3
        ("2022/09/01 10:03", "A", "A", 5.0, "US Dollar"),     # t4 self-transfer
        ("2022/09/01 10:04", "A", "B", 100.0, "Euro"),        # t5 = 120 USD
        ("2022/09/02 10:04", "A", "B", 30.0, "US Dollar"),    # t6 exactly 1440 min after t5
        ("2022/09/02 10:05", "A", "B", 40.0, "US Dollar"),    # t7 t5 has just expired
    ]
    return pd.DataFrame({
        "Timestamp": [r[0] for r in rows], "From Bank": "010", "Account": [r[1] for r in rows],
        "To Bank": "010", "Account.1": [r[2] for r in rows],
        "Amount Received": [r[3] for r in rows], "Receiving Currency": [r[4] for r in rows],
        "Amount Paid": [r[3] for r in rows], "Payment Currency": [r[4] for r in rows],
        "Payment Format": "ACH", "Is Laundering": [0, 0, 1, 0, 0, 1, 0, 0],
    })


@pytest.fixture
def toy():
    tx, _ = fe.clean_transactions(fe.standardize_columns(toy_raw()), reference_prefix="T")
    return tx


def feats_by_ref(tx, fx, history=None):
    f = fe.compute_features(tx, fx, history)
    f.index = tx["transaction_reference"].to_numpy()
    return f


# ------------------------------------------------------------------ hand-checked values
def test_toy_features_match_hand_calculation(toy):
    f = feats_by_ref(toy, FX_TOY)
    t = lambda i: f.loc[f"T-{i + 1:08d}"]  # noqa: E731

    # t0 / t1: first minute - no history, and they must not see each other
    for i in (0, 1):
        assert t(i)[["sender_out_count_24h", "pair_count_24h", "sender_txn_count_24h",
                     "receiver_in_count_24h", "sender_out_degree_24h"]].sum() == 0
    # t2 A->B @10:01 sees t0, t1
    assert t(2).sender_out_count_24h == 2
    assert t(2).sender_out_degree_24h == 2 and t(2).sender_in_degree_24h == 0
    assert t(2).sender_counterparties_24h == 2
    assert t(2).sender_volume_24h == pytest.approx(150)
    assert t(2).receiver_in_count_24h == 1 and t(2).receiver_in_degree_24h == 1
    assert t(2).pair_count_24h == 1
    mean = (np.log1p(100) + np.log1p(50)) / 2
    assert t(2).sender_amount_deviation_24h == pytest.approx(np.log1p(200) - mean)
    assert t(2).sender_amount_std_24h == pytest.approx(np.std([np.log1p(100), np.log1p(50)]))
    # t3 B->A @10:02
    assert t(3).sender_out_count_24h == 0 and t(3).sender_amount_deviation_24h == 0
    assert t(3).sender_txn_count_24h == 2 and t(3).sender_volume_24h == pytest.approx(300)
    assert t(3).receiver_out_degree_24h == 2 and t(3).receiver_txn_count_24h == 3
    assert t(3).receiver_volume_24h == pytest.approx(350)
    assert t(3).pair_count_24h == 0
    # t4 A->A self-transfer @10:03
    assert t(4).is_self_transfer == 1
    assert t(4).sender_out_count_24h == 3 and t(4).sender_in_degree_24h == 1
    assert t(4).sender_out_degree_24h == 2 and t(4).sender_counterparties_24h == 2
    assert t(4).sender_txn_count_24h == 4 and t(4).sender_volume_24h == pytest.approx(360)
    # t5 A->B @10:04 in Euro; self-loop excluded from degrees, counted in counts/volume
    assert t(5).amount_usd == pytest.approx(120)
    assert t(5).sender_out_count_24h == 4
    assert t(5).sender_out_degree_24h == 2 and t(5).sender_in_degree_24h == 1
    assert t(5).sender_txn_count_24h == 6 and t(5).sender_volume_24h == pytest.approx(370)
    assert t(5).pair_count_24h == 2
    # t6 @Sep2 10:04: window [Sep1 10:04, Sep2 10:03] holds only t5 (boundary inclusive)
    assert t(6).sender_out_count_24h == 1 and t(6).pair_count_24h == 1
    assert t(6).sender_out_degree_24h == 1 and t(6).sender_in_degree_24h == 0
    assert t(6).sender_counterparties_24h == 1 and t(6).sender_txn_count_24h == 1
    assert t(6).sender_volume_24h == pytest.approx(120)
    assert t(6).receiver_in_count_24h == 1 and t(6).receiver_in_degree_24h == 1
    assert t(6).sender_amount_deviation_24h == pytest.approx(np.log1p(30) - np.log1p(120))
    # t7 @Sep2 10:05: t5 has expired, only t6 remains
    assert t(7).sender_out_count_24h == 1 and t(7).pair_count_24h == 1
    assert t(7).sender_volume_24h == pytest.approx(30)
    assert t(7).sender_out_degree_24h == 1 and t(7).sender_counterparties_24h == 1


def test_degree_reappears_after_gap_longer_than_window():
    raw = toy_raw().iloc[[0, 6, 7]].copy()           # A->B Sep1 10:00, Sep2 10:04, Sep2 10:05
    tx, _ = fe.clean_transactions(fe.standardize_columns(raw), reference_prefix="G")
    f = feats_by_ref(tx, FX_TOY)
    assert f["sender_out_degree_24h"].tolist() == [0, 0, 1]  # Sep1 10:00 expired by Sep2 10:04


# ------------------------------------------------------------------ cleaning
def test_cleaning_keeps_leading_zeros_and_builds_account_key(toy):
    assert set(toy["sender_id"]) == {"010|A", "010|B"}


def test_cleaning_drops_invalid_and_duplicate_rows():
    raw = toy_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)       # exact duplicate
    raw.loc[1, "Amount Paid"] = -5                                  # invalid amount
    raw.loc[2, "Timestamp"] = "not a date"                          # invalid timestamp
    tx, rep = fe.clean_transactions(fe.standardize_columns(raw))
    assert rep["dropped_exact_duplicates"] == 1
    assert rep["invalid_amount"] == 1 and rep["invalid_timestamp"] == 1
    assert rep["output_rows"] == len(raw) - 3 == len(tx)


def test_missing_required_column_raises():
    with pytest.raises(fe.SchemaError, match="payment_format"):
        fe.standardize_columns(toy_raw().drop(columns="Payment Format"))


def test_unknown_currency_raises(toy):
    with pytest.raises(fe.FXError):
        fe.compute_features(toy, {"base": "US Dollar", "rates": {"US Dollar": 1.0}})


def test_fx_rates_recover_simulator_rates(full_clean):
    _, fx = full_clean
    assert fx["rates"]["US Dollar"] == 1.0
    assert fx["rates"]["Euro"] == pytest.approx(1.171783, rel=1e-4)
    assert fx["currencies_without_rate"] == []


# ------------------------------------------------------------------ leakage
def test_label_is_never_a_feature_and_never_read(real_sample):
    tx, fx = real_sample
    assert "label" not in fe.ALL_FEATURES and "label" not in fe.HISTORY_COLUMNS
    a = fe.compute_features(tx, fx)
    b = fe.compute_features(tx.assign(label=1 - tx["label"]), fx)
    c = fe.compute_features(tx.drop(columns="label"), fx)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)


def test_future_transactions_do_not_change_past_features(real_sample):
    tx, fx = real_sample
    cut = int(tx["minute"].quantile(0.5))
    full = feats_by_ref(tx, fx)
    only_past = feats_by_ref(tx[tx["minute"] <= cut], fx)
    pd.testing.assert_frame_equal(full.loc[only_past.index], only_past)


def test_same_minute_transactions_are_invisible(real_sample):
    tx, fx = real_sample
    busiest = tx["minute"].value_counts().idxmax()
    target = tx[tx["minute"] == busiest].iloc[0]["transaction_reference"]
    keep = (tx["minute"] != busiest) | (tx["transaction_reference"] == target)
    a = feats_by_ref(tx, fx).loc[target]
    b = feats_by_ref(tx[keep], fx).loc[target]
    pd.testing.assert_series_equal(a, b)


def test_transactions_older_than_window_are_ignored(real_multiday_subset):
    tx, fx = real_multiday_subset
    last_day = tx["minute"] >= int(tx["minute"].max()) - 6 * 60
    target = tx[last_day]
    cutoff = int(target["minute"].min()) - fe.HISTORY_WINDOW_MINUTES
    a = feats_by_ref(tx, fx).loc[target["transaction_reference"]]
    b = feats_by_ref(tx[tx["minute"] >= cutoff], fx).loc[target["transaction_reference"]]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_row_order_does_not_matter(real_sample):
    tx, fx = real_sample
    a = feats_by_ref(tx, fx)
    b = feats_by_ref(tx.sample(frac=1.0, random_state=0), fx).loc[a.index]
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ training / inference parity
def test_history_tail_reproduces_one_pass_features(real_multiday_subset):
    """Batch scored against a saved 24h history tail == one pass over all data."""
    tx, fx = real_multiday_subset
    cut = int(tx["minute"].quantile(0.7))
    hist, batch = tx[tx["minute"] <= cut], tx[tx["minute"] > cut]
    one_pass = feats_by_ref(tx, fx).loc[batch["transaction_reference"]]
    tail = fe.history_tail(hist)
    assert set(tail.columns) == set(fe.HISTORY_COLUMNS)
    assert int(tail["minute"].min()) > cut - fe.HISTORY_WINDOW_MINUTES
    incremental = feats_by_ref(batch, fx, history=tail)
    pd.testing.assert_frame_equal(one_pass, incremental, check_exact=False, rtol=1e-9)


def test_overlapping_history_is_rejected(real_sample):
    tx, fx = real_sample
    with pytest.raises(fe.HistoryOverlapError):
        fe.compute_features(tx.tail(10), fx, history=fe.history_tail(tx))


# ------------------------------------------------------------------ NetworkX cross-check
def test_windowed_features_match_networkx(real_multiday_subset):
    tx, fx = real_multiday_subset
    f = fe.compute_features(tx, fx)
    rng = np.random.default_rng(7)
    # query minutes on the second and third day, where the window has expired rows
    later = tx["minute"].unique()
    later = later[later >= later.min() + fe.HISTORY_WINDOW_MINUTES + 60]
    checked = 0
    for m in np.sort(rng.choice(later, size=15, replace=False)):
        G = fe.window_graph(tx, fx, int(m))
        rows = tx.index[tx["minute"] == m]
        active = [i for i in rows if tx.at[i, "sender_id"] in G or tx.at[i, "receiver_id"] in G]
        for i in (active or list(rows))[:4]:
            ref = fe.graph_features_networkx(G, tx.at[i, "sender_id"], tx.at[i, "receiver_id"])
            for k, v in ref.items():
                assert f.at[i, k] == pytest.approx(v, rel=1e-9), (k, i, m)
            checked += 1
    assert checked >= 40


def test_time_split_windows():
    ts = pd.Series(pd.to_datetime(["2022-09-01 23:59", "2022-09-02 00:00", "2022-09-10 23:59",
                                   "2022-09-11 00:00"]))
    w = {"warmup": ("2022-09-01", "2022-09-02"), "train": ("2022-09-02", "2022-09-07"),
         "test": ("2022-09-09", "2022-09-11")}
    assert fe.assign_time_split(ts, w).tolist() == ["warmup", "train", "test", "excluded"]
