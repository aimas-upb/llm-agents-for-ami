# BEIA automation benchmark

Six standing rules BEIA suggested for its lab, served by HASP with
`homelab.yaml`. Each one is checked twice:

- **static**: the plan the agents propose is scored against the case's
  `ground_truth` facts. Nothing runs.
- **behaviour**: the plan runs against the BEIA virtual lab, on its sped-up
  simulated clock.

| Case | Request (English) | Kind |
|---|---|---|
| A1 | If garage PM2.5 stays above 25 µg/m³ for 5 min, Blue Air to full speed; back to Auto once below 12 for 10 min | maintenance, held spans |
| A2 | Whenever SolarEdge exports and garage PM2.5 > 25, Blue Air to full speed | triggered, recurring |
| A3 | Alert me whenever the SolarEdge meter shows more than 2 kW imported or exported | monitor, alert |
| A4 | Whenever the garage AC turns on, close the garage window | triggered, recurring |
| A5 | Keep garage humidity 60–65 %: dehumidifier on above 65, off below 60 | maintenance |
| A6 | Whenever it gets warmer than 26 °C in the garage, turn on the AC at 21 °C | triggered, recurring |

A3 uses 2 kW, not 3. At the simulated site's late-autumn noon, the PV array
cannot export 3 kW.

## Files

- `cases/A*.json`:
  - `query`;
  - `ground_truth`, the facts a right plan has;
  - `initial_state`;
  - `stimuli`, simulator settings at simulated times, each followed by the
    assertions it must cause.
- `ground_truth.py`: the static scorer (reads, acts, alerts, forbidden, shape).
- `reference_plans.py`: a hand-written right plan per case. The unit test
  `tests/unit/test_beia_ground_truth.py` checks that each one passes its own
  case, and that planted mistakes fail on the right fact.
- `run_cases.py`: the runner, built on the lab308e harness. It writes
  `results/beia-<timestamp>.jsonl`.
- `score_plans.py`: re-scores a results file without the agents or an LLM.

## Running

Static only: about 2–4 LLM calls per case, a couple of minutes in all. HASP and
the agents' stack are needed; the simulator is not.

```bash
cd ami_agents/environment/integration/HomeAssistant && source prepare-adapter-env.sh && cd -
export AREAS="peroni_22,garaj,telemetrie,invertoare,solaredge"
export SEMANTIC_CONFIG="$PWD/ami_agents/environment/integration/HomeAssistant/homelab.yaml"
conda run --no-capture-output -n ami-agents python tests/BEIA/run_cases.py --static-only --manage-hasp
```

Full run: the same LLM calls, plus about 5–10 real minutes of simulated time.
The BEIA virtual lab (`../BEIA-virtual/beia_virtual_lab.py`) must be running in
its version that publishes `sensor.sim_time`. HASP serves that clock at `/_clock`
(`CLOCK_ENTITY=sensor.sim_time`, set by the runner), and the agents read it.

```bash
conda run --no-capture-output -n ami-agents python tests/BEIA/run_cases.py --manage-hasp
```

Re-score saved plans after changing a ground truth:

```bash
conda run -n ami-agents python tests/BEIA/score_plans.py tests/BEIA/results/beia-<stamp>.jsonl
```

A case whose plan fails the static check skips the behavioural one (reported
`skipped (static failed)`); `--behaviour-anyway` runs it regardless.
