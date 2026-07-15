# iQFoil Race — ML Simulation

**1,000 boards, each steered by its own tiny neural network, learn to race a full iQFoil
course — start sequence, upwind beats, gate choice, and all — through pure evolution.
Nobody teaches them to sail. Tacking emerges on its own.**

Part of **The 35 to 24 Project** — a windsurfer's push from 35th to 24th, with data.

![status](https://img.shields.io/badge/status-active-brightgreen) ![deps](https://img.shields.io/badge/dependencies-zero-blue)

---

## The simulation (`ml_demo.html`)

Open the file in any browser. No build, no server, no dependencies — everything
(physics, neural networks, evolution, rendering) is hand-coded vanilla JS in one file.

### The course

Real iQFoil race format, top-down view, wind from the top:

```
                    ● MARK 1
                   /|\
                  / | \        ← laylines (tacking cone)
                 /  |  \
        GATE L ●    |    ● GATE R      ← downwind gate, boats pick a side
                    |
      ○───────── START LINE ─────────○
   MARK 3 ●         |
            FINISH  ┊  ← crossed on a beam reach
```

1. **Prestart** — boats spawn scattered behind the line and may move freely, but the
   line is closed until the gun (the real 3-minute sequence, compressed to 18s).
2. **Start** — cross the line between the buoys.
3. **Beat** upwind to Mark 1 — impossible to sail straight (no-go cone), so tacking is required.
4. **Run** downwind to the leeward gate — each boat freely chooses the left or right buoy.
5. **Second beat** back up to Mark 1.
6. **Reach** to Mark 3 (downwind-left of the start), then a **half-wind sprint** to the finish.

### The boats

Each board obeys a real sailing polar: dead upwind is a ±35° no-go cone, best speed at
90–110° true wind angle, slow dead downwind. Speed is a pure function of wind angle —
steering is the only control.

Each board is steered by its own **7 → 8 → 1 neural network** (73 weights):

| Inputs | Output |
|---|---|
| bearing to next mark (sin/cos), wind angle (sin/cos), distance, countdown, leg number | rudder: turn left/right |

### The learning

No gradients, no backprop — this is an **evolution strategy**:

1. Race all 1,000 policies simultaneously.
2. Rank by course progress (legs completed, distance to next mark, finish time).
3. Keep the top 50 verbatim, breed the next 950 from the top 100 with Gaussian mutation.
4. Repeat.

What emerges, unprogrammed:

- **Tacking** upwind and angling downwind (fastest surviving policy under the polar constraints)
- **Line positioning** during the prestart
- **Fleet herding** at the gate — the population usually converges on one side within
  ~30 generations, just like a real fleet finding the favored gate

Measured on the current settings: first finishers around generation 10; by generation 100
roughly 750/1000 boats complete the course and the best race time drops from ~70s to ~51s.

### Controls

| Control | What it does |
|---|---|
| Start evolution | run/pause the simulation |
| Skip 1 gen | compute a full generation instantly, no animation |
| Speed 1×–40× | simulation steps per frame |
| Reset fleet | new random seed — a different evolution every run |

Charts track best/mean fitness and best race time per generation; the history table
logs every generation including the gate split.

---

## The session tracker (`session_tracker.py`)

CLI tool that logs real training sessions to `sessions.json` — the data that will
eventually feed a real (PyTorch) setup-prediction model.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy

.venv/bin/python session_tracker.py add --date 2026-07-14 --wind 18 --board 95 \
    --sail 8.0 --location "Kaunas Lagoon" --minutes 90 --type training
.venv/bin/python session_tracker.py list
.venv/bin/python session_tracker.py filter --min-wind 15 --max-wind 25 --type race
.venv/bin/python session_tracker.py stats
```

`stats` uses NumPy for averages, monthly counts, and best-result conditions.

---

## Roadmap

- [x] Session tracker CLI (dicts → class → JSON persistence → NumPy stats → argparse)
- [x] Browser ML demo: supervised fit of sail size vs wind (v1)
- [x] Browser ML demo: evolutionary fleet racing on a full course (current)
- [ ] Proper mark roundings (port-side, crossing-direction checks)
- [ ] Wind shifts and gusts during a race
- [ ] **M8**: PyTorch model predicting equipment setup from real logged sessions

## Why evolution and not backprop?

Steering a race is a control problem: the "right answer" per timestep is unknown, only
the final outcome is scored — so there is no label to backpropagate against. Evolution
strategies sidestep that by searching weight-space directly. The PyTorch equivalents of
this demo would be REINFORCE or ES with batched policy tensors; that comparison is the
point of the exercise.
