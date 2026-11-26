import html
import json
from pathlib import Path

import streamlit as st


def _render_module(module):
    # Opens with the plain-language summary (phase_a/describe.py); the model's working note goes to the bottom.
    if module.get("summary"):
        st.markdown(f'<p class="agemis-module-summary">{html.escape(module["summary"])}</p>', unsafe_allow_html=True)

    # All seven ratings in one row of equal cells (wraps on narrow screens).
    metrics = [("Novelty", "novelty_score"), ("Impact", "impact_potential"), ("Certainty", "certainty_level"),
               ("Horizon", "temporal_horizon"), ("Velocity", "velocity"), ("Inertia", "inertia"),
               ("Sentiment", "sentiment")]
    cells = "".join(
        f'<div class="agemis-metric"><div class="agemis-metric-label">{label}</div>'
        f'<div class="agemis-metric-value">{html.escape(str(module.get(key) or "—"))}</div></div>'
        for label, key in metrics)
    st.markdown(f'<div class="agemis-metrics">{cells}</div>', unsafe_allow_html=True)

    if module.get("novelty_rationale"):
        st.markdown(f"**Novelty rationale:** {module['novelty_rationale']}")

    domains = module.get("steeps_intersections") or []
    if domains:
        formatted_domains = " · ".join(f"`{domain}`" for domain in domains)
        st.markdown(f"**STEEPS intersections:** {formatted_domains}")

    # Signal boxes show only the title; hovering (or tapping) one shows its full text below it.
    signals = module.get("evidence_signals") or []
    if signals:
        boxes = []
        for signal in signals:
            title = signal.get("title", "") if isinstance(signal, dict) else str(signal)
            content = signal.get("content", "") if isinstance(signal, dict) else ""
            hover = (f'<div class="agemis-signal-pop"><b>{html.escape(title)}</b>{html.escape(content)}</div>'
                     if content else "")
            boxes.append(f'<div class="agemis-signal" tabindex="0"><div class="agemis-signal-title">'
                         f'{html.escape(title)}</div>{hover}</div>')
        st.markdown(f'<div class="agemis-section-title">Signals composing this module (evidence sources) · '
                    f'{len(signals)}</div><div class="agemis-signal-hint">Hover over a signal to read it.</div>'
                    f'<div class="agemis-signals">{"".join(boxes)}</div>', unsafe_allow_html=True)

    if module.get("reasoning"):
        with st.expander("Working notes from building this module"):
            st.caption("The AI's own note on why it grouped this evidence into one module. "
                       "Item numbers refer to its internal working list.")
            st.markdown(module["reasoning"])


# Module boxes: up to 5 per row, same size (twice as wide as tall); dark blue border for modules, light blue for proto-modules.
GRID_CSS = """
<style>
.st-key-agemis_mod_grid, .st-key-agemis_proto_grid {
  display: grid !important; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; margin-bottom: 0.5rem; }
.st-key-agemis_mod_grid > div, .st-key-agemis_proto_grid > div { width: 100% !important; min-width: 0; }
.st-key-agemis_mod_grid button, .st-key-agemis_proto_grid button {
  width: 100%; aspect-ratio: 2 / 1; min-height: 6.5rem; height: auto; padding: 0.75rem 1rem; background: #ffffff;
  border: 2px solid #1e3a8a; border-radius: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  transition: background 0.15s, box-shadow 0.15s, transform 0.15s; }
.st-key-agemis_proto_grid button { border-color: #93c5fd; }
.st-key-agemis_mod_grid button:hover, .st-key-agemis_proto_grid button:hover {
  background: #eff6ff; box-shadow: 0 6px 16px rgba(30, 58, 138, 0.15); transform: translateY(-2px); }
.st-key-agemis_mod_grid button p, .st-key-agemis_proto_grid button p {
  font-size: 1.05rem; font-weight: 600; line-height: 1.3; color: #17233c; white-space: normal; text-align: center; }
.agemis-module-summary { font-size: 1.12rem; line-height: 1.55; color: #17233c; margin: 0 0 0.6rem 0;
  padding: 0.75rem 1rem; background: #f8fafc; border-left: 4px solid #1e3a8a; border-radius: 6px; }
.agemis-metrics { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 0.5rem; margin: 0.4rem 0 1rem 0; }
.agemis-metric { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.55rem 0.7rem; min-width: 0; }
.agemis-metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
.agemis-metric-value { font-size: 1.2rem; font-weight: 600; color: #17233c; margin-top: 0.15rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agemis-section-title { font-size: 1.05rem; font-weight: 700; color: #17233c; margin-top: 0.4rem; }
.agemis-signal-hint { font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem; }
.agemis-signals { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem;
  margin-bottom: 1rem; }
.agemis-signal { border: 2px solid #1e3a8a; border-radius: 12px; background: #ffffff; cursor: default; outline: none;
  transition: background 0.15s, box-shadow 0.15s; }
.agemis-signal:hover, .agemis-signal:focus-within { background: #eff6ff; box-shadow: 0 6px 16px rgba(30, 58, 138, 0.15); }
.agemis-signal-title { min-height: 6rem; display: flex; align-items: center; justify-content: center; text-align: center;
  padding: 0.6rem 0.8rem; font-size: 0.95rem; font-weight: 600; line-height: 1.3; color: #17233c; }
/* Hover window: as wide as the signal grid, placed next to the hovered box by a script in phase_ui.py. */
.agemis-signal-pop { display: none; position: fixed; z-index: 1000100;
  max-height: 22rem; overflow-y: auto; padding: 0.9rem 1.1rem; background: #ffffff; border: 1px solid #cbd5e1;
  border-radius: 10px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18); font-size: 0.92rem; line-height: 1.55; color: #1f2937; }
.agemis-signal-pop b { display: block; color: #17233c; margin-bottom: 0.35rem; }
.agemis-signal:hover .agemis-signal-pop, .agemis-signal:focus .agemis-signal-pop { display: block; }
@media (max-width: 900px) { .agemis-metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .agemis-signals { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .agemis-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.agemis-grid-title { font-size: 1.15rem; font-weight: 700; color: #17233c; margin: 1.6rem 0 0.1rem 0; }
@media (max-width: 1100px) { .st-key-agemis_mod_grid, .st-key-agemis_proto_grid { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
@media (max-width: 860px)  { .st-key-agemis_mod_grid, .st-key-agemis_proto_grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 560px)  { .st-key-agemis_mod_grid, .st-key-agemis_proto_grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; }
  .st-key-agemis_mod_grid button p, .st-key-agemis_proto_grid button p { font-size: 0.92rem; } }
</style>
"""


def _searchable(module):
    return f"{module.get('name', '')} {' '.join(module.get('steeps_intersections') or [])}".lower()


def _open_module(title, module):
    """Pop-up window with the module's full content; Streamlit adds the ✕ close button."""
    st.dialog(title, width="large")(_render_module)(module)


def _render_grid(items, key, kind, active_modules):
    with st.container(key=key):
        for i, module in enumerate(items):
            name = module.get("name") or ("Unnamed proto-module" if kind == "proto" else "Unnamed module")
            label = f"{name} 🟢" if name in active_modules else name
            if st.button(label, key=f"{key}_{i}", width="stretch"):
                _open_module(name, module)


def _load_state():
    state_path = Path(__file__).resolve().parents[1] / "main_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info("No saved results yet. Build the modules from the panel behind the gear first.")
        return
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse main_state.json: {exc}")
        return

    if not isinstance(state, dict):
        st.error("Expected the saved pipeline state to be an object.")
        return
    return state


def render_modules():
    state = _load_state()
    if state is None:
        return

    modules = state.get("modules") or []
    if not modules:
        st.info("No modules saved yet.")
        return

    active_modules = set(state.get("active_modules") or [])
    processed_events = state.get("processed_events") or []
    st.caption(
        f"{len(modules)} modules from {len(processed_events)} processed events"
    )

    query = st.text_input("Filter modules", placeholder="Search name or STEEPS domain…")
    query = query.lower().strip()
    st.markdown(GRID_CSS, unsafe_allow_html=True)

    shown = [m for m in modules if not query or query in _searchable(m)]
    if shown:
        _render_grid(shown, "agemis_mod_grid", "module", active_modules)
    else:
        st.caption("No module matches this search.")

    proto_modules = [p for p in state.get("proto_modules") or [] if not query or query in _searchable(p)]
    if proto_modules:
        st.markdown('<div class="agemis-grid-title">Proto-modules</div>', unsafe_allow_html=True)
        st.caption("Provisional groups of weak signals clustered at a looser threshold, with low certainty.")
        _render_grid(proto_modules, "agemis_proto_grid", "proto", set())

    weak_signals = state.get("weak_signals") or []
    if weak_signals:
        with st.expander(f"Weak signals ({len(weak_signals)} implications not yet backed by enough signals)"):
            for weak in weak_signals:
                flag = "🌱 " if weak.get("tier") == "emerging" else ""
                st.markdown(f"- {flag}{weak.get('statement', '')}  \n  _{weak.get('signal', '')}_")
