# Air-Gapped Predictive Copilot for Secure MPLS Operations
### BAH 2026 — Problem Statement 13

**Team:** Kanak Prabhakar (Lead), Sachin Yadav, Roshan Kumar
**Institute:** AISECT University, Hazaribagh

---

## What this is

An autonomous, **air-gapped** AI assistant for a Network Operations Center (NOC).
It watches an MPLS / SD-WAN network, **predicts failures before they happen**,
and explains them in plain language — all without any connection to the internet
or cloud.

This repository is the **Phase 1 + 2 foundation**: the simulated network and the
telemetry it produces. The predictive model (Phase 3), offline LLM copilot
(Phase 4), and dashboard (Phase 5) build on top of this data.

---

## The 5-layer design

```
Layer 1  Simulated MPLS / SD-WAN network   <-- THIS PHASE
Layer 2  Telemetry pipeline (CSV export)   <-- THIS PHASE
Layer 3  Predictive fault engine (AI)          (next)
Layer 4  Offline LLM copilot (RAG)             (next)
Layer 5  NOC operator dashboard                (next)
```

Everything runs inside an **air-gapped boundary** — no outbound network calls,
ever. That is the core security requirement of the problem statement.

---

## Folder structure

```
mpls_copilot/
├── run_phase1.py              # one-command entry point
├── README.md                  # this file
├── requirements.txt
├── netcopilot/
│   ├── sim/
│   │   ├── topology.py        # defines the network (nodes, links, roles)
│   │   └── generator.py       # produces telemetry + injects faults
│   └── telemetry/
│       └── export.py          # writes telemetry to CSV
└── data/
    └── telemetry.csv          # generated dataset (after you run it)
```

---

## How to run

```bash
# from inside the mpls_copilot/ folder
python3 run_phase1.py

# or generate a bigger dataset
python3 run_phase1.py --steps 500 --out data/big.csv
```

You only need `numpy` and `pandas` for later phases — Phase 1 itself uses the
Python standard library only, so it runs anywhere.

```bash
pip install -r requirements.txt
```

---

## How to read the code (start here)

Read the files in this order — each one has detailed comments:

1. **`netcopilot/sim/topology.py`**
   The network map. Defines CE / PE / P device roles, the sites
   (branch, hub, datacenter), physical links, and IPSec overlay tunnels.
   Run it directly to see the network: `python3 -m netcopilot.sim.topology`

2. **`netcopilot/sim/generator.py`**
   The heart of Phase 1. Produces realistic telemetry (utilization, latency,
   jitter, packet loss, errors) for every link at every time-step. Crucially,
   when a fault is injected it **ramps up slowly** — this slow build-up (the
   "precursor") is exactly what the AI will learn to catch early.
   Run it directly to watch a fault build: `python3 -m netcopilot.sim.generator`

3. **`netcopilot/telemetry/export.py`**
   Streams all the telemetry to a CSV file inside `data/`. This CSV is the
   training set for the predictive model.

---

## The key idea (for the pitch / finale)

Conventional NOC tools are **reactive**: they alert only after a threshold is
breached — i.e. after users are already suffering.

Our generator deliberately creates a **precursor window**: every fault drifts
upward gradually before it becomes critical. A threshold alarm fires late
(at the "impact" phase). Our model is trained to detect the drift during the
"precursor" phase — giving the NOC **lead time** to act before service breaks.

That lead time is the whole value of the system, and it is what the
Technical Merit score (35%) rewards.

---

## What's next

- **Phase 3:** train an LSTM / gradient-boosting model on `telemetry.csv` to
  flag precursor patterns and estimate time-to-impact.
- **Phase 4:** bundle a small quantized LLM + RAG over runbooks for plain-language
  explanations, fully offline.
- **Phase 5:** a local web dashboard answering: what fails next, why, and what to do.
