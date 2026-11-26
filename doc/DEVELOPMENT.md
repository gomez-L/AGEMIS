# AGEMIS: developer guide

How to configure, run and extend the AGEMIS pipeline and user interface. For an overview of the platform, see the [README](../README.md).

AGEMIS uses FalkorDB for graph persistence and Streamlit for the user interface.

## Prerequisites

- Python 3.11+ and Docker (for FalkorDB)
- A LLM endpoint.
  ```bash
  ollama pull gemma-4
  ```

## Configuration

All configuration is read from the environment-file (`.env`):

| Variable | Default                     | Notes                                                            |
| --- |-----------------------------|------------------------------------------------------------------|
| `BASE_URL` | `http://localhost:11434/v1` | Inside Docker usually sth like `http://host.docker.internal:11434/v1` |
| `API_KEY` | –                           | Any non-empty value for local Ollama (`not-needed-local`).       |
| `MODEL_NAME` | `gemma-4`               | Model tag as the endpoint knows it.                              |
| `FALKORDB_HOST` | `127.0.0.1`                 | `falkordb` inside Compose;                                       |
| `FALKORDB_PORT` | `6379`                      |                                                                  |
| `FALKORDB_PASSWORD` | –                           | `agemis` for the dcker-compose setup                             |

```bash
cp .env.example .env
```

## Run with Docker Compose

The Compose stack starts FalkorDB, the Streamlit review interface, and provides a pipeline runner:

```bash
docker compose up --build -d
```

Open:

- AGEMIS user interface (Modules, Horizon, Scenarios, Contrastive scenarios): http://localhost:8501
- FalkorDB browser: http://localhost:3000

Run pipeline stages in the same container environment:

```bash
# Phase A
docker compose run --rm pipeline

# Run pipeline with docker
docker compose run --rm pipeline --graph
```

Application outputs are written directly out. FalkorDB data and the
Hugging Face model cache live in named volumes so they survive container restarts.

## Run locally

Install dependencies:
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Start FalkorDB:
```bash
docker run -p 6379:6379 -p 3000:3000 -e REDIS_ARGS="--requirepass agemis" -it --rm falkordb/falkordb:latest
```

`FALKORDB_PASSWORD=agemis` to use te password-protected local containere.

Run the pipeline:
```bash
./.venv/bin/python3 main.py
```

Start the user interface:
```bash
./.venv/bin/streamlit run phase_ui.py
```

The **Horizon** tab contains the horizon graph, a short overview and the human-in-the-loop chat. The **Scenarios** tab
reads Phase C results from `main_state.json`, with narration, module stress scores,
an event trace. Run `./.venv/bin/python main.py --graph` to
generate scenarios. Graph edits take effect in scenarios after rerunning the pipeline.

Scenarios are written directly for central-bank analysts: a short, plain-language
story connects an emerging development to financial services and explains what
could make the situation spread or remain contained. Each includes up to three
monitoring indicators, the key missing fact, and a practical follow-up question.
There is no separate financial-stability briefing or model-score panel in the main
reading view. Evidence, assumptions and model mechanics sit in collapsed details.

The existing narration call produces the whole scenario, with no extra calls or
dependencies. It uses bounded evidence from affected modules (up to 12 modules and
two signals each). References are checked in the supporting analysis, rather than
inserted into every sentence. Unobserved financial links must remain conditional;
these checks establish traceability, not empirical validity. If the model fails,
the trace is saved with a clear notice instead of a mechanical pseudo-scenario.
Older saved text remains available in the details; regenerate to use the new format.

To refresh scenarios without rerunning signal processing, graph mapping or policy
evaluation, use **Regenerate scenarios only** in the sidebar, or run:

```bash
./.venv/bin/python main.py --scenarios
```

This uses the saved modules and horizon graph in `main_state.json`, preserves the
other pipeline state, and clears the previous evaluation because it refers to the
old scenarios. A current saved graph is required. Edits made only in FalkorDB are
not included until they are reflected in the saved horizon map.

## Setup a local SSH tunnel (port forwarding)


```bash
ssh -N -L 11434:localhost:11434 user@gpu-host
```

locally:

```bash
BASE_URL=http://localhost:11434/v1 API_KEY=not-needed-local MODEL_NAME=gemma-4 ./.venv/bin/python3 main.py
```


## Pipeline stages and flags

Input is `signals.json` in the repo root which iss a JSON list of signal objects. State is then persisted to `main_state.json`, which records the processed
signal titles, the modules and the horizon graph.

```bash
# Run pipeline phase A
python3 main.py

# Phase A again on all signals
python3 main.py --force

# Phase B
python3 main.py --graph
```

With no new signals and no flag, `main.py` reports the current state and exits without
calling the LLM. Use `--force` to re-run Phase A over all `signals.json`

### Phase A v2 experiments

Run `python main.py --force-v2`. Every invocation creates a new folder under
`out/phase_a_v2/`; use `--v2-run-dir out/phase_a_v2/baseline` to name it explicitly
(the destination must not exist). Runs save input signals, the taxonomy, extraction
checkpoints with `driver` and `effect`, normalized `embeddings.npy`, cluster
assignments, synthesis verdicts including rejections/outliers, reattachment offers,
final module memberships, weak signals, metrics and the final result.

Re-cluster the exact same extraction and embeddings with a different threshold:

```sh
PHASE_A_V2_DISTANCE=0.40 python main.py --force-v2 \
  --v2-cache-run out/phase_a_v2/baseline \
  --v2-run-dir out/phase_a_v2/distance-040
```

Cache reuse checks the input, taxonomy, extraction prompt/schema, implication and
vector hashes, and embedder identity. It makes no extraction or embedding calls
when both cached stages are present. A completed extraction can also be reused
if embedding failed. Extraction failures remain recorded in the cache; start a
fresh run to retry them. Synthesis and reattachment run again for each experiment.

Python callers can pass `PhaseAConfig(...)` to `run_phase_a_v2`; omitted config
uses environment values read at run time. `config.json` records all settings,
model names, bound LLM parameters, prompt/schema hashes and extraction provenance.
The existing distance, linkage, minimum signals, maximum share and cohesion
variables remain supported. Additional variables are `PHASE_A_V2_SPLIT_FACTOR`,
`PHASE_A_V2_MIN_SPLIT_DISTANCE`, `PHASE_A_V2_MAX_STEEPS` and
`PHASE_A_V2_REATTACH_CANDIDATES` (default `3`; `2` offers two modules; `0` disables
reattachment for a baseline).

Implications are embedded as statement plus `driver -> effect` and clustered with
average linkage at distance `0.40` by default, so clusters follow the mechanism
rather than shared vocabulary. Embeddings cached from the older statement-only
text are recomputed.

After synthesis a consolidation pass offers module pairs to the LLM when their
centroid similarity, plus `0.05` per shared signal, reaches
`PHASE_A_V2_MERGE_SIMILARITY` (default `0.7`; `0` disables it). Modules judged to
be one subsystem are re-synthesized as one module whose `sub_themes` lists the
modules it absorbed; every check is saved in `consolidation.json`.

Weak implications are offered in cosine-nearest order to up to three modules.
An LLM must accept the shared subsystem and transmission path; the module is then
re-synthesized. The attachment is declined only if synthesis flags the candidate
itself; existing members it flags return to the weak pool and are logged as
`evicted_ids`. Failed/declined offers leave the implication weak.

Extraction scores each implication's `novelty` and `impact` (1-10). Weak signals
are returned ranked by `emerging_score` (novelty x impact x source trust), and
those with both scores at 6 or more carry `tier: "emerging"`. Phase B receives
the 20 highest-ranked statements.

The return value adds `failed_extractions` as the fourth result list alongside
`modules`, `weak_signals` and `discarded`, plus `metrics` and `run_dir`. Failures
include the input signal, its run-local ID and the error. Main state preserves
these failures, the run path and metrics for downstream use.

The printed and saved metrics use **all input signals**, including failed and
discarded extractions, as the coverage denominator. A signal with multiple
implications may support a module and still have a weak implication;
`weak_only_signal_share` distinguishes signals with no module evidence. Module
size counts distinct input signals, including separate inputs with identical
titles. Rejection/failure rates use synthesis attempts; outlier rate uses all
implications offered to initial synthesis. Reattachment counts are separate.
STEEPS spread counts broad categories over retained implications, rather than
only the truncated display labels. Cross-domain spread is a diagnostic for
process-based grouping, not proof of it. Stability reports adjusted Rand index
on pre-synthesis cluster partitions (including weak clusters) at distance ±0.05,
holding other settings fixed and using the same vectors. Cache-based runs also
report ARI against the source run and record its configuration; it does not measure LLM
synthesis or reattachment stability.

`phase_a/debug_ui.py` (`render_debug`) inspects these saved runs without
calling the LLM; it is not shown in the main interface and can be added back as a tab for development. It refreshes every three seconds to show extraction checkpoints
while a run is in progress. Select a signal to compare its raw content with its
structural implications, driver, effect, financial stability channel and STEEPS
labels, then follow the diagram into clusters and module/weak-signal outcomes.
Status filters expose discarded, failed and pending inputs; expandable sections
show synthesis/reattachment decisions, metrics, configuration and prompts, with
an extraction JSON download.


## Module summaries

Each module opens with a two-sentence, plain-language summary written for central bankers
(`phase_a/describe.py`). Summaries are written automatically after Phase A and only rewritten when a
module's content changes. To add or refresh them for the saved state:

```bash
./.venv/bin/python scripts/describe_modules.py          # missing or outdated summaries only
./.venv/bin/python scripts/describe_modules.py --force  # rewrite all
```

## Hosted deployment (Azure Container Apps)

`Dockerfile.azure` builds a slim image of the user interface only (no local embedding models; the
pipeline controls are hidden with `AGEMIS_HIDE_PIPELINE=1`). At start-up `scripts/seed_falkordb.py`
loads the saved horizon graph from `main_state.json` into a FalkorDB sidecar container, so the hosted app
shows the pre-computed modules, horizon and scenarios. `deploy_azure.sh` builds and pushes a new image
version and switches the container app to it.

Setting `APP_PASSWORD` enables a simple shared-password screen (`auth_gate.py`); without it the app
opens directly, as in local use. Optional: `CHAT_TIMEOUT_SECONDS` (default 60) and
`CHAT_REASONING_EFFORT` (default `low`) tune the human-in-the-loop chat.
