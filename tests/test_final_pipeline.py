from __future__ import annotations

import pandas as pd
import numpy as np

from evaluation.evaluate_windows import evaluate_windows
from generator.pmu_generator import generate_scenario
from neural.feature_extractor import extract_window_features
from neural.timing_features import compute_timing_features
from neural.train import _collect_scenario_windows
from wls import WeightedLeastSquares
from chi_squared_test import ChiSquaredTest
from fusion.decision_fusion import DecisionFusion
from dsse.pmu_localization import localize_dominant_pmu


def test_generator_reproducibility(tmp_path):
    out1 = tmp_path / "rep1.csv"
    out2 = tmp_path / "rep2.csv"
    generate_scenario("normal", out1, pmu=1, seed=7)
    generate_scenario("normal", out2, pmu=1, seed=7)
    assert out1.read_bytes() == out2.read_bytes()


def test_fault_scenarios_generate(tmp_path):
    out = tmp_path / "normal.csv"
    generate_scenario("normal", out, pmu=1, seed=11)
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) > 100


def test_timings_are_finite_and_distinct():
    sync = pd.read_csv("data/scenarios/PMU2_sync_r01.csv")
    drift = pd.read_csv("data/scenarios/PMU2_clock_drift_r01.csv")

    def fault_slice(df, fault_col, start_pad=0.1, end_pad=0.2):
        active = df[fault_col].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        assert active.any(), f"No active samples found for {fault_col}"
        fault_start = float(df.loc[active, "Time (s)"].min())
        fault_end = float(df.loc[active, "Time (s)"].max())
        lo = max(float(df["Time (s)"].min()), fault_start - start_pad)
        hi = min(float(df["Time (s)"].max()), fault_end + end_pad)
        return df[(df["Time (s)"] >= lo) & (df["Time (s)"] <= hi)].copy()

    sync_slice = fault_slice(sync, "PMU2 Sync Fault Active")
    drift_slice = fault_slice(drift, "PMU2 Clock Drift Fault")

    assert not sync_slice.empty
    assert not drift_slice.empty

    sync_feat = compute_timing_features(sync_slice)
    drift_feat = compute_timing_features(drift_slice)

    for key in ["offset", "short_term_slope", "long_term_slope", "variance", "step_change", "persistence", "delta_phi_mean"]:
        assert np.isfinite(sync_feat[key])
        assert np.isfinite(drift_feat[key])

    assert sync_feat["short_term_slope"] > 0.0
    assert drift_feat["long_term_slope"] > sync_feat["long_term_slope"]


def test_feature_vector_retains_pmupair_specific_signal():
    df = pd.DataFrame({
        "Time (s)": np.linspace(0.0, 1.0, 128),
        "PMU1 Voltage Phase": np.linspace(0.0, 10.0, 128),
        "PMU2 Voltage Phase": np.linspace(2.0, 12.0, 128),
        "PMU3 Voltage Phase": np.linspace(4.0, 14.0, 128),
        "PMU1 Sync Fault Active": [False] * 128,
        "PMU2 Sync Fault Active": [False] * 128,
        "PMU3 Sync Fault Active": [False] * 128,
        "PMU1 Clock Drift Fault": [False] * 128,
        "PMU2 Clock Drift Fault": [False] * 128,
        "PMU3 Clock Drift Fault": [False] * 128,
        "PMU1 Bad Data": [False] * 128,
        "PMU2 Bad Data": [False] * 128,
        "PMU3 Bad Data": [False] * 128,
    })
    feat = extract_window_features(df)
    assert feat.shape[0] >= 20
    assert np.all(np.isfinite(feat))


def test_training_labels_follow_window_fault_state(tmp_path):
    rows = []
    for i in range(40):
        t = i * 0.1
        row = {
            "Time (s)": t,
            "PMU1 Voltage Phase": 0.0 + t,
            "PMU2 Voltage Phase": 2.0 + t,
            "PMU3 Voltage Phase": 4.0 + t,
            "PMU1 Sync Fault Active": False,
            "PMU2 Sync Fault Active": False,
            "PMU3 Sync Fault Active": False,
            "PMU1 Clock Drift Fault": False,
            "PMU2 Clock Drift Fault": False,
            "PMU3 Clock Drift Fault": False,
            "PMU1 Bad Data": False,
            "PMU2 Bad Data": False,
            "PMU3 Bad Data": False,
        }
        rows.append(row)
    rows[15]["PMU2 Sync Fault Active"] = True
    rows[16]["PMU2 Sync Fault Active"] = True
    rows[17]["PMU2 Sync Fault Active"] = True
    path = tmp_path / "PMU2_sync_probe.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    samples = _collect_scenario_windows(path, window_size=10, stride=5, max_windows_per_file=400)
    labels = [sample["fault_type"] for sample in samples]
    assert "NORMAL" in labels
    assert "SYNC" in labels
    assert len(set(labels)) >= 2


def test_detection_latency_uses_time_seconds(tmp_path):
    rows = []
    for i in range(30):
        t = i * 0.1
        rows.append({
            "Time (s)": t,
            "PMU1 Voltage Phase": 0.0 + t,
            "PMU2 Voltage Phase": 2.0 + t,
            "PMU3 Voltage Phase": 4.0 + t,
            "PMU1 Sync Fault Active": False,
            "PMU2 Sync Fault Active": False,
            "PMU3 Sync Fault Active": False,
            "PMU1 Clock Drift Fault": False,
            "PMU2 Clock Drift Fault": False,
            "PMU3 Clock Drift Fault": False,
            "PMU1 Bad Data": False,
            "PMU2 Bad Data": False,
            "PMU3 Bad Data": False,
        })
    rows[10]["PMU2 Sync Fault Active"] = True
    rows[11]["PMU2 Sync Fault Active"] = True
    rows[12]["PMU2 Sync Fault Active"] = True
    path = tmp_path / "latency_case.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    class DummyModel:
        def predict(self, features):
            return np.array(["SYNC"], dtype=object)

    result = evaluate_windows(str(path), model=DummyModel(), window_size=10, stride=5)
    assert 0.0 <= result["detection_latency"] <= 1.0
    assert np.isfinite(result["detection_latency"])


def test_wls_and_chi_square_smoke():
    from measurement_model import measurement_model
    from network_model import NUM_BUSES

    x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    z = measurement_model(x)
    solver = WeightedLeastSquares()
    x_hat, residual, W, active_indices, chi2_result = solver.solve(z, x, bad_data_weight=0.1)
    assert residual.shape[0] == z.shape[0]
    assert "p_value" in chi2_result
    assert 0.0 <= chi2_result["p_value"] <= 1.0


def test_pmu_localization_and_fusion():
    residual = np.array([0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    dominant, energies, share = localize_dominant_pmu(residual)
    assert dominant in {1, 2, 3}
    assert share >= 0.0

    fusion = DecisionFusion()
    decision = fusion.fuse({"fault_type": "BAD_DATA", "faulty_pmu": dominant, "fault_confidence": 0.9, "pmu_confidence": 0.9}, {"anomaly": True, "chi_square": 30.0, "p_value": 0.01, "dominant_pmu": dominant, "residual_share": share})
    assert decision["decision"] in {"HIGH", "MEDIUM", "LOW", "STATISTICAL", "NORMAL"}


def test_post_isolation_re_estimation_smoke():
    from measurement_model import measurement_model
    x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    z = measurement_model(x)
    solver = WeightedLeastSquares()
    x_hat, residual, W, active_idx, chi2 = solver.solve(z, x, bad_data_index=0, bad_data_weight=0.05)
    assert np.all(np.isfinite(x_hat))
