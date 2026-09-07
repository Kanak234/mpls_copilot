import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from netcopilot.sim.topology import build_default_topology, Role, SiteType, Topology
from netcopilot.sim.generator import FaultType, FaultScenario, TelemetryGenerator, build_demo_generator
from netcopilot.telemetry.export import export_csv
from netcopilot.model.features import build_features, feature_columns
from netcopilot.model.train import train_and_eval, time_split
from netcopilot.copilot.inference import Predictor, Alert, SIGNAL_MEANING, SIGNAL_TO_HYPOTHESIS


def test_topology_structure():
    topo = build_default_topology()
    assert len(topo.nodes) == 9
    assert len(topo.links) == 11
    
    ce_nodes = topo.sites()
    assert len(ce_nodes) == 4
    assert {n.name for n in ce_nodes} == {"branch1", "branch2", "hub", "datacenter"}
    
    tunnels = [l for l in topo.links if l.is_tunnel]
    assert len(tunnels) == 3


def test_fault_scenario_phases():
    sc = FaultScenario(
        link_key=("branch1", "pe1"),
        fault_type=FaultType.CONGESTION,
        start_step=10,
        ramp_steps=5,
        peak_steps=5,
        severity=1.0,
    )
    phase, intensity = sc.phase(5)
    assert phase == "clear"
    assert intensity == 0.0

    phase, intensity = sc.phase(12)
    assert phase == "precursor"
    assert 0.0 < intensity < 1.0

    phase, intensity = sc.phase(16)
    assert phase == "impact"
    assert intensity == 1.0

    phase, intensity = sc.phase(25)
    assert phase == "recovered"
    assert intensity == 0.0


def test_telemetry_generation_and_export():
    gen = build_demo_generator()
    steps = 20

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_csv = f.name
    try:
        summary = export_csv(gen, steps, tmp_csv)
        assert summary["rows"] == steps * len(gen.topo.links)
        assert summary["steps"] == steps
        assert os.path.exists(tmp_csv)

        df = pd.read_csv(tmp_csv)
        assert len(df) == steps * len(gen.topo.links)
        assert "step" in df.columns
        assert "link" in df.columns
        assert "utilization_pct" in df.columns
        assert "packet_loss_pct" in df.columns
        assert "fault_phase" in df.columns
        assert (df["utilization_pct"] >= 0).all()
        assert (df["packet_loss_pct"] >= 0).all()
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def test_features_and_training_pipeline():
    gen = build_demo_generator()
    steps = 100

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "telemetry.csv")
        model_path = os.path.join(tmp_dir, "model.joblib")
        export_csv(gen, steps, csv_path)

        df = pd.read_csv(csv_path)
        feats = build_features(df)
        cols = feature_columns(feats)
        assert len(cols) > 0
        assert "will_impact" in feats.columns

        train, test, cutoff = time_split(feats, train_frac=0.7)
        assert len(train) > 0
        assert len(test) > 0
        assert train["step"].max() <= cutoff

        report = train_and_eval(csv_path=csv_path, model_out=model_path)
        assert os.path.exists(model_path)
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
        assert "lead_time" in report
        assert report["train_rows"] == len(train)


def test_copilot_predictor_inference():
    gen = build_demo_generator()
    steps = 100

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "telemetry.csv")
        model_path = os.path.join(tmp_dir, "model.joblib")
        export_csv(gen, steps, csv_path)
        train_and_eval(csv_path=csv_path, model_out=model_path)

        predictor = Predictor(model_path=model_path, topo=gen.topo)
        raw_df = pd.read_csv(csv_path)

        alerts = predictor.latest_alerts(raw_df, risk_threshold=0.3)
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, Alert)
            assert alert.severity in ("low", "medium", "high")
            assert 0.0 <= alert.risk_score <= 1.0
            assert isinstance(alert.to_dict(), dict)
