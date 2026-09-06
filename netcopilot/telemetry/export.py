"""
export.py
---------
Runs a TelemetryGenerator and writes the full stream to a CSV file inside
the air-gapped data directory. This CSV becomes the training/validation set
for the predictive engine in Phase 3.

Everything stays local - no network, no cloud. That is the whole point of an
air-gapped design: the data never leaves the box.
"""

from __future__ import annotations
import csv
from dataclasses import fields
from pathlib import Path

from netcopilot.sim.generator import TelemetryGenerator, build_demo_generator, Sample


def export_csv(gen: TelemetryGenerator, n_steps: int, out_path: str | Path) -> dict:
    """
    Stream telemetry to a CSV file. Returns a small summary dict
    (rows written, fault counts) for logging.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [f.name for f in fields(Sample)]
    rows = 0
    phase_counts: dict[str, int] = {}

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for s in gen.stream(n_steps):
            writer.writerow(s.__dict__)
            rows += 1
            phase_counts[s.fault_phase] = phase_counts.get(s.fault_phase, 0) + 1

    return {
        "rows": rows,
        "steps": n_steps,
        "file": str(out_path),
        "phase_counts": phase_counts,
    }


if __name__ == "__main__":
    gen = build_demo_generator()
    n_steps = 200
    summary = export_csv(gen, n_steps, "data/telemetry.csv")

    print("Phase 1 telemetry export complete.")
    print(f"  file        : {summary['file']}")
    print(f"  time-steps  : {summary['steps']}")
    print(f"  total rows  : {summary['rows']}")
    print("  per-phase   :")
    for phase, count in sorted(summary["phase_counts"].items()):
        print(f"      {phase:>10} : {count}")
