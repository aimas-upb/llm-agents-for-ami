# HomeBench environment

HomeBench (17,366 smart-home requests over 100 homes, 12 rooms and up to 15 device types
each) served the way SimuHome is served: as HMAS Thing Descriptions, so the UA, ISA and
EnvExplorer agents run on it unchanged. Data layer: `engine.py`, `tests/homebench/convert.py`.

## Two processes, as for SimuHome

```
engine  simulator.py   FastAPI, API under /api, {status, data, error} envelope,
                       wraps engine.HomeBenchEngine           (SimuHome: ../SimuHome simulator)
HBTD    hbtd.py        Thing Descriptions + WebSub hub, stateless projection of the
                       engine's home, SHTD's CLI and routes   (SimuHome: shtd.py)
```

```
python ami_agents/environment/integration/HomeBench/simulator.py --port 8099 \
    --home tests/homebench/data/homes/home75.json
python ami_agents/environment/integration/HomeBench/hbtd.py \
    --sim http://127.0.0.1:8099/api --home home75 --port 8097
```

`verify_phase2.py` starts such a pair on free ports and checks it end to end (see below).

**`worker.py`** is what the bench starts (`tests/simuhome/bench/suites/homebench.py`):
`HomeBenchWorker`, `SimuHomeWorker`'s API. `start()` runs the engine; `load(name)` ends the
previous episode (HBTD stopped, so its subscriptions go), resets the engine to the episode's
home and restarts HBTD bound to `home<id>`. The home id repeats across the many episodes of
one home. `python .../HomeBench/worker.py` loads two episodes in a row as a smoke test.

**`run_engine.py`** runs one home by hand, on fixed ports (the counterpart of SimuHome's
`run_inspector.py`): `python .../HomeBench/run_engine.py --home 42 --sim-port 8099
--hbtd-port 8097`, then point a console runner at `YGGDRASIL_URL=http://127.0.0.1:8097`.

| File | Role |
|---|---|
| `worker.py` | `HomeBenchWorker`: an engine and an HBTD per worker, the episode lifecycle |
| `run_engine.py` | one home loaded, on fixed ports, for debugging |
| `mappings.py` | loads `tests/homebench/mappings/*.yaml`; the lookups the TD builder and HBTD use; the gold round trip |
| `td_builder.py` | `HomeBenchTD(SimuHomeTD)`: `artifact()` and the attribute handling are replaced, everything else is inherited |
| `simulator.py` | the engine as an HTTP app |
| `sim_client.py` | `HomeBenchClient(SimuHomeClient)`: same method names, HomeBench actuation calls |
| `hbtd.py` | the TD server |
| `verify_phase2.py` | the P2 gate |

## What a device looks like

Ids: room id = HomeBench key (`ding_room` stays `ding_room`, titled "Dining Room");
device id = `<room>_<device>` (`bathroom_light`). The vacuum robot has no room in HomeBench:
it is `household_vacuum_robot` in a synthetic `household` workspace (`homeont:BuildingSpace`).

Each attribute is one property affordance; each settable attribute is also one action of
the same `td:name`, typed with a command class, as SHTD does for Matter attributes. The
planner resolves goals by the command class, so `onOff` stands for `turn_on` and `turn_off`:

| Device state | Action | Command class | Value |
|---|---|---|---|
| on/off devices | `onOff` | `homeont:SetOnOffCommand` | boolean (`true` = `turn_on()`) |
| curtain, blinds, garage door | `openClose` | `homeont:SetOpenCloseCommand` | `open` / `closed` |
| media player | `playback` | `homeont:SetOperationalStateCommand` | `playing` / `paused` / `stopped` |
| trash | `pack` | `homeont:PackCommand` | none (`pack()`) |
| vacuum `state` | none (read only) | | |

Numeric and enum attributes (`brightness`, `temperature`, `mode`, `fanSpeed`, `swing`,
`speed`, `intensity`, `interval`, `degree`, `volume`, `cleaningArea`, `color`) use
`saref:SetAbsoluteLevelCommand`, `homeont:SetModeCommand` or `homeont:SetColorCommand`;
their range or options are the device's own (`constraints` in the home file). `battery` and
`tank` are read only. `set_song`, `set_artist` and `set_style` never occur in gold and are
not served.

**The gold string is a function of `(device, action, value)`** through the `command_map`
table: `bathroom_light` + `brightness` + 20 is `bathroom.light.set_brightness(20)`;
`household_vacuum_robot` + `mode` + `auto` is `vacuum_robot.set_mode(auto)`. The forward
direction (action to engine command) is parsed out of the same table, so the two cannot
drift.

## Errors

A missing or wrongly typed value is **HTTP 400**: `onOff` needs a JSON boolean, a number an
integer, an enum a string, `color` three integers, `pack` no value. SHTD reads a missing
value on an on/off action as `false` and silently powers the device off; HBTD does not. A
well-typed value the device refuses (out of range, not one of its options) is **422**, as
in SHTD. Unknown device or action: 404.

## Change notification

A WebSub subscription to an artifact, a room or the home (`POST /hub/`, intent verified)
gets one notification per changed attribute, with SHTD's fields plus `propertyUri` (the
property's form URL) and `value`, which the EnvExplorer's event loop needs to update its
state (SHTD's payload has neither, and the update is skipped silently).

## No physics, no TD-SOSA effects

HomeBench has no clock and no room variables: `GET /_shtd/time` returns the home's fixed
`base_time`, state changes only when a command is applied, and a command has no side
effect on any other attribute (the engine is gold-consistent, not physically correct: it
applies `set_*` to a device that is off, because 6,472 gold calls do exactly that).
Therefore:

* no `homeont:Environment` resource and no room properties are served;
* no ambient sensor exists (the `observable_properties`, `ambient_sensors` and
  `actuation_effects` tables are empty on purpose);
* **no action carries a `tdsosa:` effect.** Such a node claims "this action changes that
  variable", a causal model HomeBench does not have; faking one would feed the planner a
  fact the benchmark never states.

## Verify

```
python ami_agents/environment/integration/HomeBench/verify_phase2.py     # ~1.5 min, no LLM
```

Crawls one home with `YggdrasilIntegration` (the EnvExplorer's code path) and checks that
every device is found with its state, every action round-trips to its gold string (live,
through the TD forms, and over all 18,832 gold calls of the dataset), a missing value is
400, the WebSub payload carries `propertyUri` and `value`, and the documents have the shape
above. It needs `python tests/homebench/convert.py --homebench ../HomeBench` run once
(`tests/homebench/data/`, git-ignored).

## Tables

`tests/homebench/mappings/*.yaml` are generated by `python tests/homebench/convert.py
--mappings-only --homebench ../HomeBench` (reads only the homes file). A row with
`status: approved` is kept verbatim on regeneration. The generator fails when the data has
a device type, attribute or room it has no specification for.
