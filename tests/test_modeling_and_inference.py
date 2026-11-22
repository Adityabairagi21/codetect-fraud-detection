"""
Tests for services/modeling.py (scores, risk levels, explanations) and
services/fraud_inference.py (artifact loading and end-to-end inference).

Artifact tests need the notebook to have been executed (models/ populated);
they are skipped otherwise.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services import feature_engineering as fe
from services import modeling as md
from services.fraud_inference import ARTIFACT_FILES, ArtifactError, FraudInference

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HAVE_ARTIFACTS = all((MODELS / f).exists() for f in ARTIFACT_FILES.values())
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="run the notebook first")


# ------------------------------------------------------------------ scoring
def test_isolation_forest_scores_outliers_higher(real_sample):
    """Score direction: an extreme amount must score as more abnormal than a typical one."""
    tx, fx = real_sample
    F = fe.compute_features(tx, fx)
    feats = md.MODEL_FEATURES["transaction_only"]
    pre = md.build_preprocessor(feats).fit(F[feats])
    model = md.fit_isolation_forest(pre.transform(F[feats]), n_estimators=100, seed=0)
    typical = F[feats].iloc[[0]].copy()
    typical[["amount_usd"]] = F["amount_usd"].median()
    extreme = typical.copy()
    extreme[["amount_usd"]] = F["amount_usd"].max() * 1000
    s_typ, s_ext = md.raw_anomaly_score(model, pre.transform(pd.concat([typical, extreme])))
    assert s_ext > s_typ


def test_normalization_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    train = rng.normal(size=5000)
    norm = md.fit_score_normalizer(train)
    x = np.sort(rng.normal(scale=3, size=2000))
    s = md.normalize_scores(x, norm)
    assert np.all(np.diff(s) >= 0)
    assert s.min() >= 0 and s.max() <= 100
    assert md.normalize_scores(np.array([np.median(train)]), norm)[0] == pytest.approx(50, abs=1)


def test_risk_level_boundaries():
    s = np.array([0, 89.999, 90, 96.99, 97, 99.49, 99.5, 100])
    assert md.risk_level(s).tolist() == ["Low", "Low", "Medium", "Medium", "High", "High",
                                         "Critical", "Critical"]
    assert md.anomaly_status(md.risk_level(s)).tolist() == ["Normal"] * 4 + ["Suspicious"] * 4


def test_risk_levels_follow_training_percentiles():
    rng = np.random.default_rng(1)
    train = rng.normal(size=200_000)
    norm = md.fit_score_normalizer(train)
    shares = pd.Series(md.risk_level(md.normalize_scores(train, norm))).value_counts(normalize=True)
    assert shares["Critical"] == pytest.approx(0.005, abs=0.001)
    assert shares["High"] + shares["Critical"] == pytest.approx(0.03, abs=0.002)


def test_frequency_encoder_uses_training_frequencies_only():
    enc = md.FrequencyEncoder().fit(pd.DataFrame({"payment_format": ["ACH", "ACH", "ACH", "Cash"]}))
    out = enc.transform(pd.DataFrame({"payment_format": ["ACH", "Cash", "Carrier Pigeon"]}))
    np.testing.assert_allclose(out[:, 0], [0.75, 0.25, 0.0])


@pytest.mark.parametrize("encoding", md.ENCODINGS)
def test_each_encoding_uses_the_declared_inputs(real_sample, encoding):
    tx, fx = real_sample
    F = fe.compute_features(tx.head(5000), fx)
    feats = md.MODEL_FEATURES["graph_enhanced"]
    pre = md.build_preprocessor(feats, encoding).fit(F[feats])
    X = pre.transform(F[feats])
    used = md.input_features(feats, encoding)
    if encoding == "numeric_only":
        assert not {"payment_format", "payment_currency", "is_cross_currency", "is_self_transfer"} & set(used)
        assert X.shape[1] == len(used)
    elif encoding == "frequency":
        assert X.shape[1] == len(feats)
    assert np.isfinite(X).all()


# ------------------------------------------------------------------ abnormal features
def test_abnormality_ranks_the_extreme_feature_first(real_sample):
    tx, fx = real_sample
    F = fe.compute_features(tx, fx)
    ref = md.fit_feature_reference(F, fe.ALL_FEATURES)
    row = F.iloc[[0]].copy()
    row["sender_out_degree_24h"] = 10_000
    top = md.top_abnormal_features(row, ref, fe.ALL_FEATURES, k=3)
    assert len(top) == 3 and top.iloc[0]["feature_name"] == "sender_out_degree_24h"
    assert top.iloc[0]["direction"] == "high"
    assert "unusually high" in top.iloc[0]["description"]
    assert top["contribution_score"].is_monotonic_decreasing


def test_unseen_category_is_maximally_rare(real_sample):
    tx, fx = real_sample
    F = fe.compute_features(tx, fx)
    ref = md.fit_feature_reference(F, ["payment_format"])
    row = F.iloc[[0]].copy()
    row["payment_format"] = "Carrier Pigeon"
    ab, dr = md.feature_abnormality(row, ref, ["payment_format"])
    assert ab.iloc[0, 0] == pytest.approx(np.log10(len(F)))
    assert dr.iloc[0, 0] == "rare"


# ------------------------------------------------------------------ artifacts / inference
def test_missing_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ArtifactError, match="Missing model artifacts"):
        FraudInference(tmp_path)


@needs_artifacts
def test_artifacts_load_and_match_code_schema():
    eng = FraudInference(MODELS)
    assert eng.selected_model in md.MODEL_FEATURES
    meta = eng.metadata
    assert meta["feature_version"] == fe.FEATURE_VERSION
    assert meta["model_features"] == md.MODEL_FEATURES
    assert [r["min_score"] for r in meta["risk_levels"]] == [c for _, c in md.RISK_LEVELS]


@needs_artifacts
def test_tampered_artifact_is_rejected(tmp_path):
    for f in ARTIFACT_FILES.values():
        shutil.copy(MODELS / f, tmp_path / f)
    with open(tmp_path / ARTIFACT_FILES["graph_model"], "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ArtifactError, match="checksum"):
        FraudInference(tmp_path)


@needs_artifacts
def test_incompatible_feature_version_is_rejected(tmp_path):
    for f in ARTIFACT_FILES.values():
        shutil.copy(MODELS / f, tmp_path / f)
    meta = json.loads((tmp_path / "model_metadata.json").read_text())
    meta["feature_version"] = "0.0.1"
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="feature version"):
        FraudInference(tmp_path)


@needs_artifacts
def test_inference_on_raw_records_matches_saved_notebook_scores(full_clean):
    """Raw CSV-shaped records -> FraudInference == scores the notebook saved for those rows."""
    tx, _ = full_clean
    eng = FraudInference(MODELS)
    scored = pd.read_csv(ROOT / "outputs" / "reports" / "scored_validation_test.csv.gz")
    start = pd.Timestamp("2022-09-08 03:00")
    batch = tx[(tx["timestamp"] >= start) & (tx["timestamp"] < start + pd.Timedelta(minutes=20))]
    raw = batch[list(fe.IBM_AML_COLUMN_MAP)].rename(columns=fe.IBM_AML_COLUMN_MAP)
    raw["Timestamp"] = batch["timestamp"].dt.strftime(fe.TIMESTAMP_FORMAT)
    raw["transaction_reference"] = batch["transaction_reference"].to_numpy()
    history = tx.loc[tx["minute"] < int(batch["minute"].min()), fe.HISTORY_COLUMNS]
    res = eng.score(raw.reset_index(drop=True), history=history)
    got = res.results.set_index("transaction_reference")
    exp = scored.set_index("transaction_reference").loc[got.index]
    np.testing.assert_allclose(got["transaction_raw_score"], exp["transaction_raw_score"], rtol=1e-9)
    np.testing.assert_allclose(got["graph_raw_score"], exp["graph_raw_score"], rtol=1e-9)
    assert (got["risk_level"].to_numpy() == exp["risk_level"].to_numpy()).all()
    sus = got.index[got["anomaly_status"] == "Suspicious"]
    if len(sus):
        assert set(res.explanations["transaction_reference"]) == set(sus)
        assert (res.explanations.groupby("transaction_reference").size() == 3).all()


@needs_artifacts
def test_inference_rejects_bad_input(real_sample):
    eng = FraudInference(MODELS)
    tx, _ = real_sample
    raw = tx.head(50)[list(fe.IBM_AML_COLUMN_MAP)].rename(columns=fe.IBM_AML_COLUMN_MAP)
    raw["Timestamp"] = tx.head(50)["timestamp"].dt.strftime(fe.TIMESTAMP_FORMAT)
    with pytest.raises(fe.SchemaError):
        eng.score(raw.drop(columns="Payment Format"))
    bad = raw.copy()
    bad["Payment Currency"] = "Doubloon"
    with pytest.raises(ValueError, match="exchange rate"):
        eng.score(bad, history=tx.iloc[:0][fe.HISTORY_COLUMNS])
    with pytest.raises(fe.HistoryOverlapError):
        eng.score(raw)       # Sep 5 batch vs saved Sep 10 history tail


@needs_artifacts
def test_label_column_does_not_affect_inference(real_sample):
    eng = FraudInference(MODELS)
    tx, _ = real_sample
    b = tx.head(300)
    raw = b[list(fe.IBM_AML_COLUMN_MAP)].rename(columns=fe.IBM_AML_COLUMN_MAP)
    raw["Timestamp"] = b["timestamp"].dt.strftime(fe.TIMESTAMP_FORMAT)
    empty_hist = tx.iloc[:0][fe.HISTORY_COLUMNS]
    a = eng.score(raw, history=empty_hist, explain="none").results
    flipped = raw.assign(**{"Is Laundering": 1 - raw["Is Laundering"]})
    c = eng.score(flipped, history=empty_hist, explain="none").results
    nolab = eng.score(raw.drop(columns="Is Laundering"), history=empty_hist, explain="none").results
    for other in (c, nolab):
        np.testing.assert_array_equal(a["graph_raw_score"], other["graph_raw_score"])
        np.testing.assert_array_equal(a["transaction_raw_score"], other["transaction_raw_score"])
