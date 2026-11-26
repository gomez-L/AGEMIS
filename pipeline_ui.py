
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SIGNALS_PATH = BASE_DIR / "signals.json"
STATE_PATH = BASE_DIR / "main_state.json"
RUN_DIR = BASE_DIR / "out"
LOG_PATH = RUN_DIR / "pipeline.log"
RUN_INFO_PATH = RUN_DIR / "pipeline_run.json"
EXIT_PATH = RUN_DIR / "pipeline.exit"
ABORT_PATH = RUN_DIR / "pipeline.aborted"


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _pipeline_status():
    signals = _load_json(SIGNALS_PATH, [])
    state = _load_json(STATE_PATH, {})
    processed = set(state.get("processed_events", []))

    return {
        "signals": len(signals),
        "new_signals": sum(s.get("title") not in processed for s in signals),
        "modules": len(state.get("modules") or []),
        "has_graph": bool((state.get("horizon_map") or {}).get("knowledge_graph_edges")) and not state.get("stale"),
        "scenarios": len(state.get("scenarios") or []),
        "emerging": (
            len(state.get("proto_modules") or [])
            + sum(w.get("tier") == "emerging" for w in state.get("weak_signals") or [] if isinstance(w, dict))
        ),
    }


def _pid_alive(pid):
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            return False
    except ChildProcessError:
        pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    # The PID may have been reassigned since the run ended.
    try:
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "main.py" in cmd


def _abort_run(pid, grace=5.0):
    """Terminate the run's process group and then kill it.."""
    ABORT_PATH.write_text(str(int(time.time())))
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    deadline = time.time() + grace
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _current_run():
    info = _load_json(RUN_INFO_PATH, None)
    if not info:
        return None

    info["running"] = _pid_alive(info["pid"])

    try:
        info["exit_code"] = int(EXIT_PATH.read_text().strip())
    except (OSError, ValueError):
        info["exit_code"] = None

    info["aborted"] = ABORT_PATH.exists()
    return info


def _start_run(args, label):
    RUN_DIR.mkdir(exist_ok=True)
    EXIT_PATH.unlink(missing_ok=True)
    ABORT_PATH.unlink(missing_ok=True)

    command = shlex.join([sys.executable, "-u", str(BASE_DIR / "main.py"), *args])
    shell_cmd = f"{command}; echo $? > {shlex.quote(str(EXIT_PATH))}"

    with open(LOG_PATH, "w") as log:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", shell_cmd],
            cwd=BASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )

    RUN_INFO_PATH.write_text(json.dumps({"pid": proc.pid, "label": label}))


def _read_log(max_chars=6000):
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


@st.fragment(run_every=2)
def _live_run_view():
    run = _current_run()
    if run is None:
        return

    if run["running"]:
        st.info(f"⏳ {run['label']} is running...")
        if st.button("⛔ Abort run", key="abort_run", type="primary", use_container_width=True,
                     help="Stops the running pipeline process. Steps already saved to disk are kept."):
            _abort_run(run["pid"])
            st.session_state.pipeline_watching = False
            st.rerun(scope="app")
        st.code(_read_log() or "(waiting for output)", language=None)
    elif st.session_state.get("pipeline_watching"):
        st.session_state.pipeline_watching = False
        st.rerun(scope="app")


def render_pipeline_controls():
    sb = st.sidebar
    sb.header("Pipeline")

    status = _pipeline_status()
    sb.markdown(
        f"- Signals: **{status['signals']}** ({status['new_signals']} new)\n"
        f"- Modules: **{status['modules']}**\n"
        f"- Horizon graph: **{'ready' if status['has_graph'] else 'not generated'}**\n"
        f"- Scenarios: **{status['scenarios']}**"
    )

    run = _current_run()
    running = bool(run and run["running"])

    force = sb.checkbox(
        "Reprocess all signals",
        disabled=running,
        help="v1: decomposes every signal in signals.json, not just new ones. "
        "v2: re-extracts implications from every signal with the LLM instead of reusing the "
        "extractions saved by the last v2 run. v2 always clusters and synthesizes over all signals.",
    )
    no_new = status["new_signals"] == 0 and not force

    if sb.button(
        "Run Phase A: v1 · Direct decomposition",
        disabled=running or no_new,
        use_container_width=True,
        help="Decomposes signals directly into modules. Processes new signals unless "
        "'Reprocess all signals' is selected.",
    ):
        _start_run(["--force"] if force else [], "Phase A v1 · Direct decomposition")
        st.session_state.pipeline_watching = True
        st.rerun()

    if no_new:
        sb.caption("No new signals for v1. Click 'Reprocess all signals' to rerun it.")

    if sb.button(
        "Run Phase A: v2 · Implication clustering",
        disabled=running or status["signals"] == 0,
        use_container_width=True,
        help="Phase A V2: extract implications, cluster them by similarity, then synthesize one "
        "module per cluster. Extraction runs only for new signals (reusing the last v2 run, even an "
        "aborted one) unless 'Reprocess all signals' is selected. Clustering always covers all signals.",
    ):
        _start_run(["--force-v2"] if force else ["--v2"],
                   "Phase A v2 · Implication clustering" + (" (re-extract all)" if force else ""))
        st.session_state.pipeline_watching = True
        st.rerun()

    sb.caption(
        "After Phase A you can review and edit the modules in "
        "main_state.json before building the graph."
    )

    emerging = sb.checkbox(
        "Add emerging signals as graph nodes",
        value=True,
        disabled=running or status["emerging"] == 0,
        help="Adds the top-ranked emerging developments (proto-modules and single emerging weak "
        "signals) to the horizon graph as provisional, low-certainty nodes, so scenarios can start from them. The module list is not changed. Applies when the graph is built.",
    )
    if status["emerging"] == 0:
        sb.caption("No emerging signals saved. Run Phase A v2 to produce them.")

    if sb.button(
        "Run Phase B–D · Graph & scenarios",
        disabled=running or status["modules"] == 0,
        use_container_width=True,
    ):
        with_emerging = emerging and status["emerging"] > 0
        _start_run(["--graph", "--emerging-nodes"] if with_emerging else ["--graph"],
                   "Phase B–D · with emerging nodes" if with_emerging else "Phase B–D")
        st.session_state.pipeline_watching = True
        st.rerun()

    if sb.button(
        "Regenerate scenarios only",
        disabled=running or not status["has_graph"] or status["modules"] == 0,
        use_container_width=True,
        help="Run Phase C using saved modules and the saved horizon graph. Skips signal processing, graph mapping and policy evaluation. Clears the previous evaluation.",
    ):
        _start_run(["--scenarios"], "Phase C · Scenarios only")
        st.session_state.pipeline_watching = True
        st.rerun()
    if not status["has_graph"]:
        sb.caption("Build the horizon graph first to regenerate scenarios on their own.")

    with sb:
        if running:
            st.session_state.pipeline_watching = True
            _live_run_view()
        elif run:
            exit_code = run["exit_code"]
            ok = exit_code == 0

            if run["aborted"]:
                result = "aborted"
            elif exit_code is None:
                result = "stopped"
            elif ok:
                result = "finished"
            else:
                result = f"failed (exit {exit_code})"

            with st.expander(
                f"Last run: {run['label']} {result}",
                expanded=not ok,
            ):
                st.code(_read_log() or "(no output)", language=None)
