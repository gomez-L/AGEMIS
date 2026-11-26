"""Central-bank scenarios first; model mechanics available on demand."""

import json
from pathlib import Path

import streamlit as st


def render_scenario(scenario, key="scenario", show_title=True):
    ready = scenario.get("presentation_version") == 2 and scenario.get("narrated_by") == "llm"
    if ready:
        if show_title:
            st.markdown(f"### {scenario.get('title', 'Scenario')}")
        if scenario.get("counterfactual_assumption"):
            st.markdown(f'<p class="agemis-scn-whatif">{_esc(scenario["counterfactual_assumption"])}</p>',
                        unsafe_allow_html=True)
        st.markdown(scenario.get("narration", ""))
        watch = [item for item in scenario.get("watch_for") or [] if isinstance(item, dict)]
        if watch:
            cards = "".join(f'<div class="agemis-scn-watch-item"><b>{_esc(item.get("indicator", ""))}</b>'
                            f'{_esc(item.get("interpretation", ""))}</div>' for item in watch)
            st.markdown(f'<div class="agemis-scn-label">What to watch</div><div class="agemis-scn-watch">{cards}</div>',
                        unsafe_allow_html=True)
        notes = [(label, scenario.get(key)) for label, key in
                 [("What we still need to know", "key_uncertainty"), ("Question to follow up", "analyst_question")]
                 if scenario.get(key)]
        if notes:
            st.markdown('<div class="agemis-scn-notes">' + "".join(
                f'<div class="agemis-scn-note"><div class="agemis-scn-label">{label}</div>{_esc(text)}</div>'
                for label, text in notes) + "</div>", unsafe_allow_html=True)
    elif scenario.get("narrated_by") == "symbolic":
        st.warning("Scenario writing was unavailable. Check the model connection and the last-run log, then select ‘Regenerate scenarios only’. The evidence and model results are saved below.")
    else:
        st.info("This saved scenario uses the older format. Select ‘Regenerate scenarios only’ for a readable scenario focused on central-bank concerns. The original text is saved below.")

    failure = scenario.get("narration_failure")
    if failure:
        with st.expander("Why this scenario was not written"):
            st.code(failure.get("error", ""), language=None)
            if failure.get("rejected_output") is not None:
                st.caption("Draft returned by the model and rejected:")
                st.json(failure["rejected_output"], expanded=False)

    # Two clickable boxes; the chosen one opens its content below (click again to close).
    panel_key = f"{key}_panel"
    current = st.session_state.get(panel_key)
    st.markdown('<div class="agemis-scn-label" style="margin-top:0.8rem">Look behind the scenario</div>',
                unsafe_allow_html=True)
    with st.container(horizontal=True, gap="small", key=f"agemis_scn_boxes_{key}"):
        for name, icon in [("Evidence & assumptions", ":material/fact_check:"), ("Model details", ":material/hub:")]:
            st.button(name, icon=icon, key=f"{key}_{name}", width="stretch",
                      type="primary" if current == name else "secondary",
                      on_click=_toggle_panel, args=(panel_key, name))
    if current == "Evidence & assumptions":
        _render_evidence(scenario, ready)
    elif current == "Model details":
        _render_model_details(scenario)


def _toggle_panel(panel_key, name):
    # Runs before the rerun (of the pop-up only, when inside one), so the pop-up stays open.
    st.session_state[panel_key] = None if st.session_state.get(panel_key) == name else name


def _esc(text):
    import html
    return html.escape(str(text))


def _render_evidence(scenario, ready):
    with st.container(border=True):
        steps = scenario.get("supporting_analysis") or []
        if steps:
            st.dataframe([{"Claim": step["mechanism"], "Basis": step["basis"],
                           "References": ", ".join(step.get("references") or [])}
                          for step in steps], hide_index=True, width="stretch")
        repairs = scenario.get("reference_repairs") or []
        if repairs:
            st.caption(f"{len(repairs)} claim(s) cited references that were not supplied. Those references "
                       "were removed, and claims left without support are shown as assumptions.")
        evidence = (scenario.get("financial_evidence") or {}).get("evidence") or []
        if evidence:
            st.markdown("**Source signals**")
            st.dataframe(evidence, hide_index=True, width="stretch")
        for key, label in [("trigger_signal", "Starting signal"), ("novelty_element", "What changes")]:
            if scenario.get(key):
                st.markdown(f"**{label}**")
                st.write(scenario[key])
        if not ready:
            st.markdown("**Previously saved text**")
            st.write(scenario.get("narration", "No written scenario saved."))
            if scenario.get("financial_stability"):
                st.json(scenario["financial_stability"], expanded=False)
        if not steps and not evidence:
            st.caption("No structured supporting evidence was saved for this scenario.")



def _render_model_details(scenario):
    with st.container(border=True):
        st.caption("These scores describe graph propagation, not financial losses or forecast probabilities. Steps do not represent elapsed time.")
        stresses = scenario.get("stresses") or {}
        rounds = scenario.get("activation_round") or {}
        events = scenario.get("events") or []
        st.write(f"{len(stresses)} affected modules · {len(events)} calculated relationships")
        if scenario.get("shock"):
            st.json(scenario["shock"], expanded=False)
        if stresses:
            rows = [{"Module": name, "Stress": value, "First step": rounds.get(name)}
                    for name, value in sorted(stresses.items(), key=lambda item: -item[1])]
            st.bar_chart(rows, x="Module", y="Stress", horizontal=True)
            st.dataframe(rows, hide_index=True, width="stretch")
        if events:
            st.dataframe(events, hide_index=True, width="stretch")
        if scenario.get("dynamic_weights_utilization"):
            st.write(scenario["dynamic_weights_utilization"])
        if scenario.get("seed"):
            st.json(scenario["seed"], expanded=False)


SCENARIO_CSS = """
<style>
[class*="st-key-agemis_scn_grid"] { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; align-items: stretch !important; }
.st-key-agemis_scn_grid_v2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
[class*="st-key-agemis_scn_grid"] > div { width: 100% !important; min-width: 0; display: flex !important; }
[class*="st-key-agemis_scn_card_"] { position: relative; align-self: stretch !important; height: auto !important; border: 2px solid #1e3a8a; border-radius: 12px;
  background: #ffffff; padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  transition: background 0.15s, box-shadow 0.15s, transform 0.15s; }
[class*="st-key-agemis_scn_card_"]:hover { background: #eff6ff; box-shadow: 0 6px 16px rgba(30, 58, 138, 0.15);
  transform: translateY(-2px); }
/* The whole card is the click target: an invisible button covers it. */
[class*="st-key-agemis_scn_card_"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
  position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important; z-index: 2; margin: 0; }
[class*="st-key-agemis_scn_card_"] [data-testid="stButton"], [class*="st-key-agemis_scn_card_"] [data-testid="stButton"] button {
  width: 100% !important; height: 100% !important; max-width: none !important; opacity: 0; cursor: pointer; }
.agemis-scn-num { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #1d4ed8; }
.agemis-scn-title { font-size: 1.15rem; font-weight: 700; line-height: 1.3; color: #17233c; margin: 0.25rem 0 0.6rem 0; }
.agemis-scn-q { font-size: 0.93rem; line-height: 1.45; color: #334155; font-style: italic; }
[class*="st-key-agemis_scn_card_"] { min-height: 34rem; padding: 1.2rem 1.3rem !important; }
.agemis-scn-q { background: #f8fafc; border-left: 3px solid #1e3a8a; border-radius: 6px; padding: 0.55rem 0.75rem; }
.agemis-scn-lead { font-size: 0.93rem; line-height: 1.55; color: #1f2937; margin-top: 0.8rem; }
.agemis-scn-sub { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: #64748b; margin: 0.9rem 0 0.35rem 0; }
.agemis-scn-chain { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem; }
.agemis-scn-chip { font-size: 0.8rem; line-height: 1.3; padding: 0.2rem 0.55rem; border-radius: 999px;
  background: #eff6ff; color: #1e3a8a; border: 1px solid #bfdbfe; }
.agemis-scn-chip-start { background: #1e3a8a; color: #ffffff; border-color: #1e3a8a; font-weight: 600; }
.agemis-scn-arrow { color: #94a3b8; font-size: 0.85rem; }
.agemis-scn-tags { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.agemis-scn-tag { font-size: 0.8rem; line-height: 1.3; padding: 0.2rem 0.55rem; border-radius: 6px;
  background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0; }
.agemis-scn-open { font-size: 0.88rem; font-weight: 600; color: #1d4ed8; margin-top: 1rem; }
/* Inside the scenario pop-up */
.agemis-scn-whatif { font-size: 1.1rem; line-height: 1.5; color: #17233c; font-style: italic; margin: 0 0 0.8rem 0;
  padding: 0.75rem 1rem; background: #f8fafc; border-left: 4px solid #1e3a8a; border-radius: 6px; }
.agemis-scn-label { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
  color: #64748b; margin: 0.4rem 0 0.4rem 0; }
.agemis-scn-watch { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 0.6rem; margin-bottom: 0.8rem; }
.agemis-scn-watch-item { border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.92rem;
  line-height: 1.45; color: #334155; background: #ffffff; }
.agemis-scn-watch-item b { display: block; color: #17233c; margin-bottom: 0.2rem; }
.agemis-scn-notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.6rem; margin-bottom: 0.4rem; }
.agemis-scn-note { background: #f8fafc; border-radius: 8px; padding: 0.6rem 0.9rem; font-size: 0.95rem; line-height: 1.5; color: #1f2937; }
.agemis-scn-note .agemis-scn-label { margin-top: 0; }
[class*="st-key-agemis_scn_boxes_"] button { min-height: 3.4rem; border-radius: 10px; border: 2px solid #93c5fd;
  background: #ffffff; color: #17233c; font-weight: 600; }
[class*="st-key-agemis_scn_boxes_"] button p { font-size: 1rem; font-weight: 600; }
[class*="st-key-agemis_scn_boxes_"] button:hover { background: #eff6ff; border-color: #1e3a8a; color: #17233c; }
[class*="st-key-agemis_scn_boxes_"] button[kind="primary"], [class*="st-key-agemis_scn_boxes_"] button[data-testid="stBaseButton-primary"] {
  background: #1e3a8a; border-color: #1e3a8a; color: #ffffff; }
.agemis-scn-path { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem; margin-bottom: 0.35rem; }
.agemis-scn-path-label { font-size: 0.78rem; color: #64748b; margin: 0.35rem 0 0.2rem 0; }
.agemis-scn-domain { font-size: 0.8rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 999px;
  background: #ede9fe; color: #5b21b6; border: 1px solid #ddd6fe; }
.agemis-scn-stats { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.45rem; }
.agemis-scn-stat { font-size: 0.8rem; color: #334155; background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: 6px; padding: 0.15rem 0.5rem; }
.agemis-scn-stat b { color: #17233c; }
.agemis-scn-emerging { font-size: 0.82rem; color: #92400e; background: #fffbeb; border: 1px dashed #f59e0b;
  border-radius: 6px; padding: 0.35rem 0.6rem; margin-top: 0.55rem; }
.agemis-scn-note-line { font-size: 0.9rem; color: #475569; margin: 0 0 1rem 0; }
.agemis-scn-note-line b { color: #17233c; }
@media (max-width: 860px) { [class*="st-key-agemis_scn_grid"] { grid-template-columns: minmax(0, 1fr); } }
</style>
"""


def _card_html(index, scenario, label=None, spread_html=None):
    """Preview card: question, opening paragraph, how the shock spreads, and what to watch.
    `label` replaces "Scenario N"; `spread_html` replaces the default "How it spreads" chain."""
    parts = [f'<div class="agemis-scn-num">{_esc(label or f"Scenario {index + 1}")}</div>',
             f'<div class="agemis-scn-title">{_esc(scenario.get("title", "Untitled scenario"))}</div>']
    if scenario.get("counterfactual_assumption"):
        parts.append(f'<div class="agemis-scn-q">{_esc(scenario["counterfactual_assumption"])}</div>')
    ready = scenario.get("presentation_version") == 2 and scenario.get("narrated_by") == "llm"
    opening = (scenario.get("narration") or "").strip().split("\n\n")[0] if ready else ""
    if opening:
        parts.append(f'<div class="agemis-scn-lead">{_esc(opening)}</div>')
    start = (scenario.get("shock") or {}).get("target_module")
    stresses = scenario.get("stresses") or {}
    reached = [name for name, _ in sorted(stresses.items(), key=lambda item: -item[1]) if name != start][:3]
    if spread_html is not None:
        parts.append(spread_html)
    elif start or reached:
        chain = ([f'<span class="agemis-scn-chip agemis-scn-chip-start">{_esc(start)}</span>'] if start else []) + \
                [f'<span class="agemis-scn-chip">{_esc(name)}</span>' for name in reached]
        parts.append('<div class="agemis-scn-sub">How it spreads</div><div class="agemis-scn-chain">'
                     + '<span class="agemis-scn-arrow">→</span>'.join(chain) + '</div>')
    watch = [item.get("indicator") for item in scenario.get("watch_for") or [] if isinstance(item, dict)]
    if watch:
        parts.append('<div class="agemis-scn-sub">What to watch</div><div class="agemis-scn-tags">'
                     + "".join(f'<span class="agemis-scn-tag">{_esc(w)}</span>' for w in watch if w) + '</div>')
    parts.append('<div class="agemis-scn-open">Open the full scenario →</div>')
    return "".join(parts)


def _open_scenario(index, scenario, key_prefix="scn", file_prefix="scenario", header_html=None):
    def body():
        if header_html:
            st.markdown(header_html, unsafe_allow_html=True)
        render_scenario(scenario, key=f"{key_prefix}_{index}", show_title=False)
        st.download_button("Download scenario JSON", data=json.dumps(scenario, indent=2, ensure_ascii=False),
                           file_name=f"{file_prefix}_{index + 1}.json", mime="application/json",
                           key=f"{key_prefix}_dl_{index}")
    st.dialog(scenario.get("title", "Scenario"), width="large")(body)()


def render_scenarios():
    state_path = Path(__file__).resolve().parents[1] / "main_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info("No saved scenarios. Run the pipeline from the sidebar.")
        return
    except (OSError, ValueError) as exc:
        st.error(f"Could not read saved scenarios: {exc}")
        return
    if not isinstance(state, dict):
        st.error("Expected the saved pipeline state to be an object.")
        return
    scenarios = state.get("scenarios") or []
    if not scenarios and state.get("scenario"):
        scenarios = [state["scenario"]]
    if not isinstance(scenarios, list) or any(not isinstance(s, dict) for s in scenarios):
        st.error("Saved scenarios must be a list of scenario objects.")
        return
    st.markdown(SCENARIO_CSS, unsafe_allow_html=True)  # also styles the Contrastive scenarios tab
    if not scenarios:
        st.info("No scenarios saved. Use the pipeline controls in the sidebar.")
    else:
        with st.container(key="agemis_scn_grid"):
            for index, scenario in enumerate(scenarios):
                with st.container(key=f"agemis_scn_card_{index}"):
                    st.markdown(_card_html(index, scenario), unsafe_allow_html=True)
                    if st.button(f"Open scenario {index + 1}", key=f"agemis_scn_open_{index}"):
                        st.session_state.pop(f"scn_{index}_panel", None)
                        _open_scenario(index, scenario)
    st.button("Reload saved scenarios", key="reload_scenarios")
