"""Read-only inspections for Phase A v2 gates. This includes runs in progress."""
import json
from collections import Counter
from pathlib import Path
from textwrap import fill

import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
RUN_ROOT = BASE_DIR / "out" / "phase_a_v2"


def _read(path, default, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default  # Later stages are often not there yet
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return default


def discover_runs(root, state_path):
    paths = {p.parent.resolve() for p in Path(root).glob("*/config.json")}
    state = _read(Path(state_path), {}, [])
    saved = state.get("phase_a_run_dir")
    if saved:
        path = Path(saved)
        path = path if path.is_absolute() else BASE_DIR / path
        if (path / "config.json").is_file():
            paths.add(path.resolve())
    return sorted(paths, key=lambda p: (p / "config.json").stat().st_mtime, reverse=True)


def load_debug_run(folder):
    errors = []
    defaults = {"config": {}, "events": [], "extraction": {}, "clusters": {},
                "modules": [], "memberships": [], "weak_signals": [], "synthesis": [],
                "reattachment": [], "metrics": {}, "prompts": {}}
    data = {name: _read(Path(folder) / f"{name}.json", default, errors)
            for name, default in defaults.items()}
    data["errors"] = errors
    return data


def signal_rows(data):
    extraction = data["extraction"]
    counts = Counter(i["signal_id"] for i in extraction.get("implications", []))
    failed = {f["signal_id"]: f.get("error", "Extraction failed")
              for f in extraction.get("failed_extractions", [])}
    discarded = {d["signal_id"] for d in extraction.get("discarded", [])}
    rows = []
    for idx, event in enumerate(data["events"]):
        status = "Failed" if idx in failed else "Extracted" if counts[idx] else "Discarded" if idx in discarded else "Pending"
        rows.append({"Signal ID": idx, "Signal": event.get("title") or "Untitled signal",
                     "Status": status, "Implications": counts[idx], "Error": failed.get(idx, "")})
    return rows


def _destinations(data):
    modules = {idx: module.get("name", f"Module {idx + 1}") for idx, module in enumerate(data["modules"])}
    membership = {imp: modules.get(idx, f"Module {idx + 1}")
                  for idx, group in enumerate(data["memberships"]) for imp in group}
    weak = {i["id"] for i in data["weak_signals"]}
    return membership, weak


def signal_flow(items, data, title):
    """Builds a small provenance graph."""
    membership, weak = _destinations(data)
    clusters = {i: c for c in data["clusters"].get("clusters", []) for i in c["implication_ids"]}
    lines = ['digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Arial"];']

    def node(key, label, color="#eef2ff"):
        lines.append(f'{json.dumps(key)} [label={json.dumps(fill(label, 48))}, fillcolor={json.dumps(color)}];')

    node("signal", title, "#e0f2fe")
    for item in items:
        key = f"imp{item['id']}"
        node(key, item["statement"])
        lines.append(f'"signal" -> "{key}" [label="extract"];')
        cluster = clusters.get(item["id"])
        previous = key
        if cluster:
            previous = f"cluster{cluster['cluster_id']}"
            node(previous, f"Cluster {cluster['cluster_id']}\n{cluster['signal_count']} signals · cohesion {cluster['cohesion']:.2f}")
            lines.append(f'"{key}" -> "{previous}" [label="embed + cluster"];')
        if item["id"] in membership:
            destination, color = membership[item["id"]], "#dcfce7"
        elif item["id"] in weak:
            destination, color = "Remains a weak implication", "#fef3c7"
        else:
            destination, color = "Awaiting later stages", "#f1f5f9"
        node(f"dest{item['id']}", destination, color)
        lines.append(f'"{previous}" -> "dest{item["id"]}";')
    return "\n".join(lines + ["}"])


def _render_signal(data, row):
    idx = row["Signal ID"]
    items = [i for i in data["extraction"].get("implications", []) if i["signal_id"] == idx]
    source, extracted = st.columns([1, 2])
    with source:
        st.markdown("**Original signal**")
        st.write(row["Signal"])
        st.write(data["events"][idx].get("content") or "No content supplied.")
        st.caption(f"Signal {idx} · {row['Status']}")
        with st.expander("Full input"):
            st.json(data["events"][idx])
    with extracted:
        st.markdown("**Extracted structural implications**")
        if row["Status"] == "Failed":
            st.error(row["Error"])
        elif row["Status"] == "Discarded":
            st.info("LLM marked this signal irrelevant or returnd no implications.")
        elif not items:
            st.info("No extraction checkpoint for this signal yet existing.")
        membership, weak = _destinations(data)
        for item in items:
            st.markdown(f"**Implication {item['id']}:** {item['statement']}")
            st.markdown(f"**Driver:** {item.get('driver', '—')}")
            st.markdown(f"**Effect:** {item.get('effect', '—')}")
            st.markdown(f"**Financial stability channel:** {item.get('mechanism', '—')}")
            st.write("STEEPS: " + "; ".join(item.get("domains", [])))
            st.caption("Destination: " + membership.get(item["id"], "Weak implication" if item["id"] in weak else "Awaiting later stages"))
    if items:
        st.graphviz_chart(signal_flow(items, data, row["Signal"]), width="stretch")
        ids = {item["id"] for item in items}
        decisions = {"synthesis": [v for v in data["synthesis"] if ids.intersection(v["implication_ids"])],
                     "reattachment": [v for v in data["reattachment"] if v["implication_id"] in ids]}
        with st.expander("Synthesis and reattachment decisions for this signal"):
            st.json(decisions)


@st.fragment(run_every=3)
def render_debug(run_root=RUN_ROOT, state_path=BASE_DIR / "main_state.json"):
    st.subheader("Phase A v2 · Debug")
    st.markdown(
        "**Extraction turns a news signal into 1–3 general structural developments.** "
        "The LLM records the driver, effect, financial stability channel and STEEPS domains. "
        "Only the implication statemnts are embedded & clustered. Synthesis then decides which clusters form modules. `EXTRACT 2/42` means the second input signal, not the second module."
    )
    paths = discover_runs(run_root, state_path)
    if not paths:
        st.info("No saved v2 runs yet. Use ‘Run Phase A: v2 · Implication clustering’ in the sidebar to generate extraction checkpoints.")
        return
    selected = st.selectbox("Saved v2 run", paths, format_func=lambda p: p.name, key="debug_run")
    data = load_debug_run(selected)
    metadata = data["config"]
    st.caption(f"{selected} · Status: {metadata.get('status', 'unknown')} · Refreshes every 3 seconds")
    for error in data["errors"]:
        st.warning(error)
    if metadata.get("error"):
        st.error(metadata["error"])
    rows = signal_rows(data)
    counts = Counter(r["Status"] for r in rows)
    implications = data["extraction"].get("implications", [])
    for column, label, value in zip(st.columns(6),
                                    ["Input signals", "Processed", "Implications", "Discarded", "Failed", "Modules"],
                                    [len(rows), data["extraction"].get("processed_count", 0), len(implications),
                                     counts["Discarded"], counts["Failed"], len(data["modules"])]):
        column.metric(label, value)
    if rows:
        st.progress(min(data["extraction"].get("processed_count", 0) / len(rows), 1), text="Extraction progress")
    st.caption("Signal → structural implications → statement embeddings → clusters → synthesized modules / weak implications")
    st.markdown("#### Inspect a signal")
    query = st.text_input("Search signals and implications", key="debug_search").strip().casefold()
    status = st.selectbox("Extraction status", ["All", "Extracted", "Discarded", "Failed", "Pending"], key="debug_status")
    searchable = {row["Signal ID"]: json.dumps(data["events"][row["Signal ID"]], ensure_ascii=False).casefold() for row in rows}
    for item in implications:
        searchable[item["signal_id"]] = searchable.get(item["signal_id"], "") + json.dumps(item, ensure_ascii=False).casefold()
    filtered = [r for r in rows if (status == "All" or r["Status"] == status)
                and (not query or query in searchable[r["Signal ID"]])]
    if filtered:
        st.dataframe(filtered, hide_index=True, width="stretch")
        by_id = {r["Signal ID"]: r for r in filtered}
        idx = st.selectbox("Signal to inspect", list(by_id),
                           format_func=lambda n: f"{n} · {by_id[n]['Signal']} ({by_id[n]['Status']})",
                           key=f"debug_signal_{selected.name}")
        _render_signal(data, by_id[idx])
    else:
        st.info("No signals match these filters.")
    with st.expander("Clusters and run metrics"):
        clusters = data["clusters"].get("clusters", [])
        if clusters:
            st.dataframe(clusters, hide_index=True, width="stretch")
            st.bar_chart({"Implications per cluster": {str(c["cluster_id"]): len(c["implication_ids"]) for c in clusters}})
        st.json(data["metrics"])
    with st.expander("Configuration, prompts and raw extraction"):
        st.json(metadata)
        st.json(data["prompts"])
        st.download_button("Download extraction JSON", json.dumps(data["extraction"], ensure_ascii=False, indent=2),
                           file_name=f"{selected.name}_extraction.json", mime="application/json")
        st.json(data["extraction"])
