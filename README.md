# Zenmux Docs

Model Weight [**HuggingFace**](https://huggingface.co/agenzaai/super-zenmux) 

A quantitative research pipeline that ingests MetaTrader 5 candles for Gold
(XAU/USD), engineers multi-timeframe technical features without lookahead bias,
and trains LightGBM / XGBoost directional classifiers under **purged
walk-forward validation** on Kaggle `GPU T4 x2`.

> Educational and research use only. Nothing here is financial, investment or
> trading advice. Leveraged instruments carry substantial risk of loss.

---

## Pipeline

```text
MetaTrader 5 terminal (XAUUSD, M1 → D1)
        │
        ▼
src/mt5_extract.py     resumable chunked pull, atomic writes, validated merge
        │
        ▼
src/features.py        multi-timeframe features (causal), ATR-deadband labels
        │              HTF context merged on bar-CLOSE time via merge_asof
        ▼
src/validation.py      expanding walk-forward folds, training tail purged
        │
        ▼
src/train.py           LightGBM + XGBoost, device probed per backend
        │
        ▼
src/evaluate.py        ROC/PR-AUC, MCC, cost-aware backtest, TreeSHAP
src/plots.py           diagnostic figures from the OOF predictions
```

---

## Quick start

```bash
python -m pip install -r requirements.txt

# 1. Pull candles from MetaTrader 5 (Windows only - needs MT5 installed + logged in)
python -m src.mt5_extract --timeframes M1 M5 M15 M30 H1 H4 D1 --start 2010-01-01

# 2. Inspect what arrived
python tools/check_raw.py

# 3. Build the feature matrix
python -m src.dataset --rebuild

# 4. Fast correctness check (~20 s, tiny folds)
python -m src.pipeline --smoke --device cpu

# 5. Full study
python -m src.pipeline --device cpu --shap --out artifacts/local_cpu
```

### On Kaggle (T4 x2)

```bash
# Stage + publish the dataset (candles + a source snapshot)
python tools/stage_kaggle_dataset.py
kaggle datasets version -p kaggle/dataset_payload --dir-mode tar -m "update"

# Build, validate and push the GPU notebook
python tools/build_kaggle_notebook.py
python tools/check_notebook.py
python tools/kaggle_cli.py kernels push -p kaggle

# Follow / collect results
python tools/kaggle_cli.py kernels status menmengleap/xauusd-direction-gpu
python tools/kaggle_cli.py kernels output menmengleap/xauusd-direction-gpu -p kaggle_out
python tools/parse_kaggle_log.py
```

---

## Design decisions

### Lookahead discipline

Two rules, both enforced in code:

1. **Every base-timeframe feature is causal** — trailing windows only
   (`ewm`, `rolling`, `diff`, `shift` with non-negative lags).
2. **Higher-timeframe context is joined on bar-close time.** An HTF bar stamped
   `t` covers `[t, t + duration)` and is only usable once it has closed, so the
   merge key is `t + duration` with `direction="backward"`. At base time `T` the
   newest selectable HTF bar is therefore the last one that closed at or before
   `T` — a 10:30 M5 bar can never see the 10:00–10:59 H1 close.

Gaps are preserved rather than forward-filled, so volatility and momentum
windows measure real elapsed time instead of inventing weekend bars.

### Why the training tail is purged

All features are backward-looking, so the only way a training row can "see" the
test period is through its **label**: the forward return of the last
`LABEL_HORIZON_BARS` training rows is computed from prices inside the test
window. `PURGE_BARS = LABEL_HORIZON_BARS` removes exactly those rows, and
`validation.check_no_overlap` asserts the invariant before a single model is
fitted — a violated split aborts the run rather than producing a pretty number.

### Why each backend probes its own device

The `lightgbm` wheels on PyPI are frequently compiled **without** the CUDA tree
learner (`CUDA Tree Learner was not enabled in this build`), and the OpenCL
build may be missing too. `train.select_device` fits a throwaway model on each
candidate (`cuda` → `gpu` → `cpu`) and keeps the first that works, so a wheel
without GPU support degrades to CPU instead of crashing the run.

Neither library benefits much from data-parallel histogram training at this
dataset size, so on ≥ 2 GPUs the backends run as **separate processes**, each
pinned with `CUDA_VISIBLE_DEVICES`. Both T4s are used, memory stays isolated,
and one backend failing cannot lose the other's results.

### Backtest honesty

Each OOF row already carries a `horizon`-bar forward return, so summing them
bar-by-bar would count one price move `horizon` times and inflate Sharpe by
roughly `sqrt(horizon)`. `evaluate.backtest` therefore strides through the OOF
series in steps of `horizon`, producing genuinely non-overlapping trades, and
charges round-trip costs only on position changes.

The threshold sweep is reported **as a diagnostic only** — selecting the best
threshold on the same out-of-sample data is itself selection bias.

---

## Data reality check

The connected terminal is logged into **MetaQuotes-Demo**, which serves roughly
100 000 bars per timeframe and caps M1 history at about three months:

| Timeframe | Bars    | Coverage from       |
| --------- | ------- | ------------------- |
| M1        |  97 446 | 2026-06-18 (3.5 mo) |
| M5        |  98 800 | 2025-04-28 (17 mo)  |
| M15       |  99 616 | 2022-07-01 (4.3 yr) |
| M30       |  99 612 | 2018-04-05 (8.5 yr) |
| H1        |  97 924 | 2009-12-31 (16 yr)  |
| H4        |  25 804 | 2010-01-04          |
| D1        |   4 318 | 2010-01-04          |

Because M1 depth is only ~3.5 months, the **M5 base timeframe** (98 800 bars,
~17 months) is used for the modelling matrix, with M15/M30/H1/H4/D1 merged as
lagged context. That is the deepest intraday option this server offers. Point
`MT5_TERMINAL_PATH` at a real broker terminal to get multi-year M1 history and
switch the base with `XAU_BASE_TF=M1`.

Quality checks (`tools/check_raw.py`) confirm zero duplicates, zero NaNs, zero
OHLC violations and a strictly increasing index across all seven timeframes.

---

## Results

Trained on Kaggle `GPU T4 x2` (two Tesla T4s; LightGBM on OpenCL GPU, XGBoost
on CUDA, one process per device). 92 545 samples × 276 features, five purged
walk-forward folds.

| Model   | Device | ROC-AUC      | PR-AUC       | Dir. accuracy | MCC       | Fold best iters | Train time |
| ------- | ------ | ------------ | ------------ | ------------- | --------- | --------------- | ---------- |
| LightGBM | gpu    | 0.5147 ±0.026 | 0.5200 ±0.044 | 0.5041        | 0.0112    | 42              | 49 s       |
| XGBoost  | cuda   | **0.5328** ±0.013 | 0.5325 ±0.046 | **0.5111** | **0.0531** | 120             | 17 s       |

Per-fold ROC-AUC (chronological, out-of-sample):

| Fold | Test window                | LightGBM | XGBoost |
| ---- | -------------------------- | -------- | ------- |
| 0    | 2025-12-05 → 2026-01-22    | 0.5000   | 0.5430  |
| 1    | 2026-01-22 → 2026-03-09    | 0.5656   | 0.5495  |
| 2    | 2026-03-09 → 2026-04-23    | 0.4982   | 0.5291  |
| 3    | 2026-04-23 → 2026-06-09    | 0.5020   | 0.5309  |
| 4    | 2026-06-09 → 2026-07-24    | 0.5077   | 0.5113  |

### Read this honestly

**A ROC-AUC of ~0.51–0.53 means there is no reliable, tradable edge in this
setup.** Four of the ten folds sit at or below 0.51, which is what random looks
like. The cost-aware long-only backtest is negative for both models
(Sharpe −0.39 and −1.25, profit factor 0.98 and 0.93) because the gross edge
(`mean_fwd_when_long_bps` ≈ +0.4 to +0.8 bps) is smaller than the 6 bps round
trip.

This is the expected outcome and the pipeline reports it as such instead of
tuning until the number looks respectable. Published claims of 54%+ accuracy on
gold direction usually come from random K-fold splits leaking the future, from
a single lucky window, or from tuning the threshold on the test set.

The value delivered here is the **methodology** — the purged walk-forward
protocol, the lookahead-safe multi-timeframe merge, the device probing and the
non-overlapping cost-aware backtest — not an exploitable signal.

### Diagnostic figures

All figures are computed from out-of-fold predictions only
(`artifacts/<backend>/plots/`):

| Figure | What it shows |
| --- | --- |
| `roc_curve.png` | ROC vs the chance diagonal — the curve hugs the diagonal |
| `precision_recall.png` | PR vs the 50.2% positive-rate baseline |
| `calibration.png` | Reliability diagram + prediction histogram |
| `cumulative_returns.png` | Strategy vs buy-and-hold on one trade grid |
| `drawdown.png` | Underwater curve, max-DD trough marked |
| `threshold_perf.png` | Sharpe / win rate / exposure across thresholds |
| `benchmark.png` | Head-to-head: return, Sharpe, drawdown, win rate |
| `return_attribution.png` | Gross signal vs transaction costs |

The two most informative:

**Calibration** — predicted probabilities span 0.30–0.75 but observed frequency
sits flat at ~0.50 across the whole range. The model is confidently wrong: it
has learned a ranking with no magnitude information. This is exactly why a
0.5 cut-off is not a meaningful trade decision here.

**Threshold sweep** — Sharpe is negative at *every* threshold from 0.40 to
0.70, and signal precision never separates from the 50.2% base rate. There is no
operating point worth trading.

**Return attribution** — gross signal +1 274 bps, costs −2 985 bps, net −1 711
bps. The (weak) gross edge is real but costs roughly twice as much to harvest at
3 bps per side, because the signal toggles ~1 000 times across 3 334 trades.
That, not the model, is the binding constraint.

Full numbers: `artifacts/reports/summary.md`, per-backend
`artifacts/<backend>/reports/report.md`, importances in
`gain_importance_*.csv` and `shap_*.csv`.

Key configuration knobs live in `src/config.py` and can be overridden by
environment variable: `XAU_BASE_TF`, `XAU_LABEL_HORIZON`, `XAU_LABEL_DEADBAND`,
`XAU_N_SPLITS`, `XAU_TRAIN_SIZE`, `XAU_TEST_SIZE`, `XAU_COST_BPS`.

---

#### Why there are two prediction routes

These are **tabular GBDT classifiers, not language models** - they consume 276
engineered features and emit a probability. Accepting features directly would
force every client to reimplement the feature pipeline and would silently allow
column-order mistakes, the classic way a tree model "works" while producing
nonsense. `/predict/candles` therefore rebuilds features server-side with the
exact same code used at training, and `/predict` exists for callers that already
have the feature matrix (it rejects missing *and* unexpected columns).

#### Feature parity

Features are computed over a bounded trailing window (`AGENZA_FEATURE_WINDOW`,
default 6000 bars) rather than the whole history, keeping requests at ~1-3 s on
a single core. The window must be *large*, not small: `ewm(span=200,
adjust=False)` is an infinite-impulse-response filter, so at 1000 bars the
residual initial-condition weight is 6.7e-3, which showed up as a 2.7% error on
`ema200_slope`; at 6000 bars it is 8.7e-14. `tools/test_serve.py` asserts the
predictions are **bit-identical** to a full-history computation.

#### Deploying / updating

```bash
$env:UPCLOUD_TOKEN = "ucat_..."; $env:AGENZA_API_KEY = "sk-..."
python tools/deploy_upcloud.py --ip 94.237.46.108            # update in place
python tools/deploy_upcloud.py --zone nl-ams1 --plan PREMIUM-2xCPU-4GB --hostname agenza-infer
```

The nginx route is installed separately by `deploy/enable_nginx_route.sh`, which
backs up the site config, syntax-checks, and only reloads on success.

#### UpCloud API notes

The published create-server documentation does not match the live API. From
UpCloud's own Go client (`upcloud-go-api/upcloud/request/server.go`) and probing:

| Field | Reality |
| --- | --- |
| body | must be wrapped: `{"server": {...}}` |
| `login_user` | a **nested object** (`username` / `ssh_keys` / `create_password`), not a string |
| `ssh_keys` | only valid here, inside `login_user`; sending it at top level reports "unknown attribute" |
| `storage_devices` | wrapped `{"storage_device": [...]}`, each disk needs `action` + `storage` + `title` |
| `plan` | required, and selects CPU/RAM; `core_number` / `memory_amount` are rejected |
| `password` | read-only (the one-time password returned *in* the response) |

Two account-level limits are worth knowing: `public_ipv4` is capped (trial: 2),
and on a trial account the firewall is immutable (`TRIAL_FIREWALL`), so a newly
created server drops all inbound traffic and cannot serve a public endpoint.

---

### Tests

```bash
python tools/test_import_without_mt5.py   # package imports without MetaTrader5
python tools/test_pipeline_offline.py     # full pipeline, MetaTrader5 poisoned
python tools/check_notebook.py            # generated notebook is sound
python tools/check_raw.py                 # OHLCV integrity report
$env:HF_TOKEN="hf_..."; python tools/verify_hf_models.py   # published models still load + score
```

`test_pipeline_offline.py` is the important one: it stubs out `MetaTrader5`,
synthesises candles and runs the real pipeline in a subprocess, reproducing the
Kaggle (Linux) environment so a missing MT5 dependency can never reach the GPU
run.

`verify_hf_models.py` closes the loop — it downloads the published boosters from
the Hub, rebuilds the feature matrix locally and asserts the saved feature
contract matches column-for-column, because a silent column-order mismatch is
the classic way a tree model "works" while producing nonsense.

---

## Notes on platform quirks

* **Kaggle JSON must be BOM-free.** PowerShell's `Set-Content -Encoding UTF8`
  and many editors write a BOM, which the CLI rejects with
  `Unexpected UTF-8 BOM`. All generated files are written with
  `encoding="utf-8"` and asserted BOM-free.
* **`kaggle datasets create` silently skips sub-folders.** Without
  `--dir-mode tar|zip` only top-level files upload — the notebook then fails at
  runtime with a missing dataset. Always pass `--dir-mode tar`.
* **Kaggle retires `security.OAuthService/IntrospectToken`,** which
  `kaggle==2.2.4` still calls, so token-only machines get a spurious
  "Authentication required". `tools/kaggle_cli.py` patches that call and steps
  aside entirely when a legacy key is configured.
* **`src` must be importable without MetaTrader5.** The wheel is Windows-only,
  so `mt5_extract` now defers the dependency to `require_mt5()` at call time.
