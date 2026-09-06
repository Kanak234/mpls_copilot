"""
generator.py
------------
Turns a Topology into a stream of realistic telemetry samples over time.

For every link, at every time-step, we emit a sample with:
    utilization_pct  - how full the link is (0-100%)
    latency_ms       - round-trip-ish latency
    jitter_ms        - variation in latency
    packet_loss_pct  - fraction of packets dropped
    errors           - interface error counter (cumulative)

WHY THIS MATTERS:
The whole challenge is "predict failures BEFORE they happen". A model can only
learn that if the training data contains the *precursor pattern* - the slow drift
that happens before a breakdown. So our generator does two things:

  1. NORMAL baseline  : small random noise around healthy values (the "calm")
  2. FAULT injection  : a fault doesn't appear instantly. It RAMPS UP over a
                        window (the "precursor"), peaks (the "impact"), then
                        optionally recovers. The ramp is what the AI learns to
                        catch early.

Each sample is labelled with the ground-truth fault state, so later we can
train and score the predictive model honestly.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Iterator

from netcopilot.sim.topology import Topology, build_default_topology


class FaultType(str, Enum):
    NONE = "none"
    CONGESTION = "congestion"          # link fills up, latency + loss climb
    ROUTE_FLAP = "route_flap"          # BGP/OSPF instability, latency spikes erratically
    TUNNEL_DEGRADE = "tunnel_degrade"  # IPSec tunnel: loss + jitter creep up
    UNDERLAY_FAIL = "underlay_fail"    # physical link slowly dies


@dataclass
class FaultScenario:
    """Describes one fault injected onto one link over a time window."""
    link_key: tuple          # which link (canonical (a,b) tuple)
    fault_type: FaultType
    start_step: int          # when the precursor begins
    ramp_steps: int          # how long the slow build-up lasts (the lead-time window)
    peak_steps: int          # how long it stays at full impact
    severity: float = 1.0    # 0..1 multiplier on how bad it gets

    def phase(self, step: int) -> tuple[str, float]:
        """
        Returns (phase_name, intensity 0..1) for a given time-step.
        intensity ramps 0 -> 1 during the precursor, holds at 1 during impact.
        """
        if step < self.start_step:
            return ("clear", 0.0)
        ramp_end = self.start_step + self.ramp_steps
        peak_end = ramp_end + self.peak_steps
        if step < ramp_end:
            # linear ramp - the precursor the model must catch
            frac = (step - self.start_step) / max(1, self.ramp_steps)
            return ("precursor", frac * self.severity)
        if step < peak_end:
            return ("impact", self.severity)
        return ("recovered", 0.0)


@dataclass
class Sample:
    """One telemetry reading for one link at one time-step."""
    step: int
    link: str                 # "a-b"
    is_tunnel: bool
    utilization_pct: float
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    errors: int
    # ground truth (used for training/scoring, NOT given to the model as a feature)
    fault_type: str
    fault_phase: str          # clear / precursor / impact / recovered


class TelemetryGenerator:
    def __init__(self, topo: Topology, seed: int = 42):
        self.topo = topo
        self.rng = random.Random(seed)
        self.scenarios: List[FaultScenario] = []
        self._error_counts: Dict[str, int] = {}

    def add_scenario(self, scenario: FaultScenario) -> None:
        self.scenarios.append(scenario)

    def _baseline(self, link) -> Dict[str, float]:
        """Healthy values with light random noise."""
        util = self.rng.uniform(20, 45)
        latency = link.base_latency_ms * self.rng.uniform(0.9, 1.2)
        jitter = self.rng.uniform(0.3, 1.5)
        loss = max(0.0, self.rng.gauss(0.05, 0.05))
        return {"util": util, "latency": latency, "jitter": jitter, "loss": loss}

    def _apply_fault(self, base: Dict[str, float], ftype: FaultType,
                     intensity: float) -> Dict[str, float]:
        """Distort the baseline according to fault type and intensity (0..1)."""
        v = dict(base)
        i = intensity
        if ftype == FaultType.CONGESTION:
            v["util"] = min(100.0, base["util"] + 55 * i)
            v["latency"] = base["latency"] * (1 + 2.5 * i)
            v["loss"] = base["loss"] + 4.0 * i
            v["jitter"] = base["jitter"] * (1 + 2 * i)
        elif ftype == FaultType.ROUTE_FLAP:
            spike = self.rng.choice([0, 0, 1]) * i
            v["latency"] = base["latency"] * (1 + 4 * spike)
            v["jitter"] = base["jitter"] * (1 + 8 * i)
            v["loss"] = base["loss"] + 2.0 * spike
        elif ftype == FaultType.TUNNEL_DEGRADE:
            v["loss"] = base["loss"] + 6.0 * i
            v["jitter"] = base["jitter"] * (1 + 5 * i)
            v["latency"] = base["latency"] * (1 + 1.5 * i)
        elif ftype == FaultType.UNDERLAY_FAIL:
            v["loss"] = base["loss"] + 15.0 * i
            v["latency"] = base["latency"] * (1 + 3 * i)
            v["util"] = max(0.0, base["util"] - 30 * i)
        return v

    def _active_fault(self, link_key, step) -> tuple[FaultType, str, float]:
        """Find which (if any) scenario is active on this link at this step."""
        for sc in self.scenarios:
            if sc.link_key == link_key:
                phase, intensity = sc.phase(step)
                if phase in ("precursor", "impact"):
                    return (sc.fault_type, phase, intensity)
        return (FaultType.NONE, "clear", 0.0)

    def stream(self, n_steps: int) -> Iterator[Sample]:
        """Yield every link's sample for every step, 0..n_steps-1."""
        for step in range(n_steps):
            for link in self.topo.links:
                key = link.key()
                label = f"{link.a}-{link.b}"
                base = self._baseline(link)
                ftype, phase, intensity = self._active_fault(key, step)
                vals = self._apply_fault(base, ftype, intensity) if intensity > 0 else base

                inc = int(vals["loss"] * self.rng.uniform(0, 3))
                self._error_counts[label] = self._error_counts.get(label, 0) + inc

                yield Sample(
                    step=step,
                    link=label,
                    is_tunnel=link.is_tunnel,
                    utilization_pct=round(vals["util"], 2),
                    latency_ms=round(vals["latency"], 2),
                    jitter_ms=round(vals["jitter"], 2),
                    packet_loss_pct=round(max(0.0, vals["loss"]), 3),
                    errors=self._error_counts[label],
                    fault_type=ftype.value,
                    fault_phase=phase,
                )


def build_demo_generator() -> TelemetryGenerator:
    """
    A ready-to-run generator with MANY injected faults spread across the
    whole timeline and across different links. This matters for honest
    evaluation: we need several complete fault-cycles in BOTH the training
    window (early steps) and the test window (later steps), so the measured
    lead-time is statistically meaningful and not based on a single fault.

    Faults are deliberately varied:
      - all four fault types appear
      - they sit on underlay links AND tunnels
      - some overlap in time (realistic - real NOCs juggle concurrent issues)
    """
    topo = build_default_topology()
    gen = TelemetryGenerator(topo, seed=7)

    scenarios = [
        # (link_a, link_b, type, start, ramp, peak, severity)
        ("hub", "pe1", FaultType.CONGESTION, 40, 22, 15, 0.9),
        ("branch1", "datacenter", FaultType.TUNNEL_DEGRADE, 70, 25, 18, 1.0),
        ("p1", "p2", FaultType.ROUTE_FLAP, 110, 18, 20, 0.85),
        ("branch2", "pe2", FaultType.UNDERLAY_FAIL, 150, 20, 15, 0.95),
        ("pe1", "p1", FaultType.CONGESTION, 190, 24, 16, 0.8),
        ("branch2", "datacenter", FaultType.TUNNEL_DEGRADE, 230, 26, 18, 0.9),
        ("pe3", "p2", FaultType.ROUTE_FLAP, 275, 16, 22, 0.85),
        ("hub", "datacenter", FaultType.TUNNEL_DEGRADE, 315, 22, 16, 1.0),
        ("datacenter", "pe3", FaultType.CONGESTION, 355, 20, 15, 0.85),
        ("branch1", "pe1", FaultType.UNDERLAY_FAIL, 395, 22, 16, 0.9),
    ]
    for a, b, ftype, start, ramp, peak, sev in scenarios:
        gen.add_scenario(FaultScenario(
            link_key=tuple(sorted((a, b))),
            fault_type=ftype,
            start_step=start, ramp_steps=ramp, peak_steps=peak, severity=sev,
        ))
    return gen


if __name__ == "__main__":
    gen = build_demo_generator()
    print(gen.topo.summary())
    print(f"Injected scenarios: {len(gen.scenarios)}\n")

    target = "hub-pe1"
    print(f"Watching link '{target}' as congestion builds (steps 35-70):")
    print(f"{'step':>4} {'util%':>7} {'lat ms':>7} {'loss%':>7} {'phase':>10}")
    for s in gen.stream(75):
        if s.link == target and 35 <= s.step <= 70 and s.step % 5 == 0:
            print(f"{s.step:>4} {s.utilization_pct:>7} {s.latency_ms:>7} "
                  f"{s.packet_loss_pct:>7} {s.fault_phase:>10}")
