"""
run_phase1.py
-------------
One-command entry point for Phase 1. Run this and it will:
  1. Build the network topology
  2. Inject the demo fault scenarios
  3. Generate telemetry over time
  4. Write everything to data/telemetry.csv

Usage:
    python3 run_phase1.py
    python3 run_phase1.py --steps 500 --out data/big.csv
"""

from __future__ import annotations
import argparse

from netcopilot.sim.generator import build_demo_generator
from netcopilot.telemetry.export import export_csv


def main():
    parser = argparse.ArgumentParser(description="Phase 1 - telemetry generation")
    parser.add_argument("--steps", type=int, default=200,
                        help="number of time-steps to simulate (default 200)")
    parser.add_argument("--out", type=str, default="data/telemetry.csv",
                        help="output CSV path")
    args = parser.parse_args()

    gen = build_demo_generator()
    print(gen.topo.summary())
    print(f"Injected fault scenarios: {len(gen.scenarios)}")
    print(f"Simulating {args.steps} time-steps...\n")

    summary = export_csv(gen, args.steps, args.out)

    print("Done. Air-gapped dataset written:")
    print(f"  file       : {summary['file']}")
    print(f"  rows       : {summary['rows']}")
    print("  phase mix  :")
    for phase, count in sorted(summary["phase_counts"].items()):
        print(f"      {phase:>10} : {count}")
    print("\nNext: Phase 3 will train a predictive model on this CSV.")


if __name__ == "__main__":
    main()
