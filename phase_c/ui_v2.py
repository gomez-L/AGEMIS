"""Contrastive scenarios tab (shown in the app as 'Contrastive scenarios')."""

import json
import os
from pathlib import Path

import streamlit as st

from phase_c.generator_v2 import inputs_hash
from phase_c.ui import _card_html, _esc, _open_scenario, render_scenario  # noqa: F401

KIND = {"cross_domain_chain": "Cross-domain chain", "convergence": "Convergence of two developments"}


def _path_html(scenario, in_card=True):
    """The cross-domain journey: domains crossed, the module-by-module path(s), size and emerging start."""
    surprise = scenario.get("surprise") or {}
    if not surprise:
        return ""
    kind = KIND.get((scenario.get("seed") or {}).get("motif_type"), "Scenario")
    parts = ['<div class="agemis-scn-sub">Domains it crosses</div><div class="agemis-scn-path">'
             + '<span class="agemis-scn-arrow">→</span>'.join(
                 f'<span class="agemis-scn-domain">{_esc(d)}</span>' for d in surprise.get("domains") or [])
             + "</div>"]
    paths = surprise.get("paths") or []
    parts.append('<div class="agemis-scn-sub">How it travels</div>')
    for number, path in enumerate(paths, 1):
        if len(paths) > 1:
            parts.append(f'<div class="agemis-scn-path-label">Development {number}</div>')
        chips = [f'<span class="agemis-scn-chip{" agemis-scn-chip-start" if i == 0 else ""}">{_esc(name)}</span>'
                 for i, name in enumerate(path)]
        parts.append('<div class="agemis-scn-path">' + '<span class="agemis-scn-arrow">→</span>'.join(chips) + "</div>")
    stats = [f"<b>{_esc(kind)}</b>"]
    if surprise.get("hops") is not None:
        stats.append(f"<b>{surprise['hops']}</b> steps")
    if surprise.get("domain_crossings") is not None:
        stats.append(f"<b>{surprise['domain_crossings']}</b> domain crossings")
    parts.append('<div class="agemis-scn-stats">' + "".join(f'<span class="agemis-scn-stat">{x}</span>' for x in stats)
                 + "</div>")
    if surprise.get("emerging"):
        parts.append('<div class="agemis-scn-emerging">🌱 Starts from an emerging development with little evidence '
                     "so far: " + _esc(", ".join(surprise["emerging"])) + "</div>")
    if not in_card:
        parts.append('<div style="height:0.8rem"></div>')
    return "".join(parts)


def render_scenarios_v2():
    from pipeline_ui import _current_run, _start_run

    st.markdown('<p class="agemis-scn-note-line"><b>Under development.</b> These scenarios are chosen for how far '
                "and across how many domains a development travels, to surface cross-domain surprises, rather than "
                "for the most fragile starting point. The scenarios in the Scenarios tab are not changed.</p>",
                unsafe_allow_html=True)
    state_path = Path(__file__).resolve().parents[1] / "main_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    has_graph = bool((state.get("horizon_map") or {}).get("knowledge_graph_edges")) and not state.get("stale")
    scenarios = state.get("scenarios_v2") or []

    if not has_graph:
        st.info("Build the horizon graph first to generate these scenarios.")
    elif not isinstance(scenarios, list) or any(not isinstance(s, dict) for s in scenarios):
        st.error("Saved contrastive scenarios must be a list of scenario objects.")
    elif not scenarios:
        st.info("No contrastive scenarios saved yet.")
    else:
        if (state.get("scenarios_v2_info") or {}).get("inputs_hash") != inputs_hash(state):
            st.warning("The modules or horizon graph changed after these scenarios were generated. "
                       "Regenerate to update them.")
        with st.container(key="agemis_scn_grid_v2"):
            for index, scenario in enumerate(scenarios):
                with st.container(key=f"agemis_scn_card_v2_{index}"):
                    st.markdown(_card_html(index, scenario, label=f"Contrastive scenario {index + 1}",
                                           spread_html=_path_html(scenario)), unsafe_allow_html=True)
                    if st.button(f"Open contrastive scenario {index + 1}", key=f"agemis_scn_v2_open_{index}"):
                        st.session_state.pop(f"scn_v2_{index}_panel", None)
                        _open_scenario(index, scenario, key_prefix="scn_v2", file_prefix="contrastive_scenario",
                                       header_html=_path_html(scenario, in_card=False))

        ranking = (state.get("scenarios_v2_info") or {}).get("top_candidates") or []
        if ranking:
            with st.expander(f"Why these paths: top {len(ranking)} candidates and their score factors"):
                st.caption("score = hops × domain crossings × outside-in × emerging × familiarity × √end stress")
                st.dataframe([{"rank": r["rank"], "selected": r["selected_as"] or "", "kind": r["motif_type"],
                               "path": " / ".join(" → ".join(p) for p in r["paths"]), "score": r["score"],
                               **r["factors"]} for r in ranking], hide_index=True)

    # Actions at the end. Generating is a pipeline run, so it is hidden where the pipeline is (Azure).
    with st.container(horizontal=True, gap="small"):
        if os.getenv("AGEMIS_HIDE_PIPELINE") != "1":
            running = bool((_current_run() or {}).get("running"))
            if st.button("Generate contrastive scenarios", key="generate_scenarios_v2", type="primary",
                         disabled=running or not has_graph or not state.get("modules"),
                         help="Uses the saved modules and horizon graph. Makes two scenario-writing calls."):
                _start_run(["--scenarios-v2"], "Phase C v2 · Cross-domain scenarios")
                st.session_state.pipeline_watching = True
                st.rerun()
        st.button("Reload saved scenarios", key="reload_scenarios_v2")
