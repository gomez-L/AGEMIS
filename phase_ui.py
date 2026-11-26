import os
import json
import base64
import hashlib
import html
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from main import get_falkor_graph
from phase_a.ui import render_modules
from phase_c.ui import render_scenarios
from phase_c.ui_v2 import render_scenarios_v2
from pipeline_ui import render_pipeline_controls
from auth_gate import require_password

load_dotenv()

APP_BRAND = "AGEMIS"
APP_SUBTITLE = "Agentic Horizon Scanning and Scenario Generation"
APP_TITLE = f"{APP_BRAND}: {APP_SUBTITLE}"

# App icon shown before "AGEMIS" (embedded in the page, so it works locally and on Azure).
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "agemis_icon.png")
try:
    with open(ICON_PATH, "rb") as icon_file:
        ICON_DATA_URI = "data:image/png;base64," + base64.b64encode(icon_file.read()).decode("ascii")
except OSError:
    ICON_DATA_URI = None

def load_page_icon():
    """App icon for the browser tab, centred on a transparent square (tab icons are square)."""
    try:
        from PIL import Image
        icon = Image.open(ICON_PATH).convert("RGBA")
    except (OSError, ImportError):
        return None
    side = max(icon.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(icon, ((side - icon.width) // 2, (side - icon.height) // 2))
    return square


st.set_page_config(layout="wide", page_title="AGEMIS: Horizon & Scenarios", page_icon=load_page_icon(),
                   initial_sidebar_state="collapsed")

# Layout:
# - no Streamlit top bar (it pushes the title down and holds the default ">>" sidebar arrow);
# - title block at the very top: icon + "AGEMIS" (big, bold) on the first line and the subtitle
#   below it; sizes scale with the window width, leaving room for a small gear in the top right;
# - the sidebar (pipeline controls) is shown as a panel on the RIGHT, below the title row, opened
#   and closed with the gear. Streamlit only supports a left sidebar, hence the CSS overrides.
st.markdown(
    """
<style>
:root {
    --agemis-title-size: clamp(1.05rem, calc((100vw - 6.5rem) / 26.5), 4.4rem);
    --agemis-subtitle-size: max(0.5 * var(--agemis-title-size), 0.75rem);
    /* Icon spans both lines: from the top of the AGEMIS capitals to the subtitle baseline.
       Measured font metrics (line-height 1.2): capitals start 0.26 x font size below the top of
       their line, the baseline is 0.925 x font size below it. */
    --agemis-icon-top: calc(0.26 * var(--agemis-title-size));
    --agemis-icon-height: calc(0.94 * var(--agemis-title-size) + 0.925 * var(--agemis-subtitle-size));
    --agemis-icon-gap: calc(0.22 * var(--agemis-title-size));
    --agemis-panel-top: calc(1.2 * var(--agemis-title-size) + 1.2 * var(--agemis-subtitle-size) + 2.8rem);
}
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 0.6rem; padding-left: 1.5rem; padding-right: 1.5rem; }
.agemis-title {
    font-size: var(--agemis-title-size); line-height: 1.2; color: #17233c; margin: 0; min-width: 0;
    display: flex; align-items: flex-start; gap: var(--agemis-icon-gap);
}
.agemis-title-text { min-width: 0; flex: 1 1 auto; }
/* Streamlit pulls markdown blocks up by 1rem at the bottom and the row clips overflow, which cut
   off the subtitle: give the 1rem back and let the row show its full content. */
.st-key-agemis_title_row .agemis-title { margin-bottom: 1rem; }
.st-key-agemis_title_row { overflow: visible !important; }
.agemis-brand { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.agemis-title .agemis-icon {
    flex: none; width: auto; height: var(--agemis-icon-height) !important; max-width: none !important;
    margin-top: var(--agemis-icon-top); vertical-align: top !important;
}
/* subtitle on its own line, smaller and regular weight, under the word AGEMIS */
.agemis-subtitle {
    font-size: var(--agemis-subtitle-size); font-weight: 400; color: #334155; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-bottom: 0.15em;
}
section[data-testid="stSidebar"] {
    position: fixed !important; top: var(--agemis-panel-top) !important; right: 0 !important; left: auto !important;
    height: calc(100vh - var(--agemis-panel-top)) !important; transform: none !important; z-index: 999990;
    border-left: 1px solid #e2e8f0; border-top: 1px solid #e2e8f0; border-radius: 10px 0 0 0;
    box-shadow: -8px 0 24px rgba(15, 23, 42, 0.12);
}
section[data-testid="stSidebar"][aria-expanded="false"] { display: none !important; }
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarHeader"] { display: none !important; }
/* gear (30px) vertically centred on the title line (line height 1.2 x title size) */
.st-key-agemis_gear { margin-top: calc((1.2 * var(--agemis-title-size) - 30px) / 2); flex: 0 0 30px; }
/* the title may shrink (ellipsis) so the gear never wraps below it */
.st-key-agemis_title_row > div:first-child { min-width: 0; flex: 1 1 auto; }
</style>
""",
    unsafe_allow_html=True,
)

require_password(APP_TITLE)  # shared password on Azure (APP_PASSWORD); no-op locally

with st.container(horizontal=True, vertical_alignment="top", gap="small", wrap=False, key="agemis_title_row"):
    icon_html = f'<img class="agemis-icon" src="{ICON_DATA_URI}" alt="">' if ICON_DATA_URI else ""
    st.markdown(f'<div class="agemis-title" role="heading" aria-level="1">{icon_html}'
                f'<div class="agemis-title-text"><div class="agemis-brand">{html.escape(APP_BRAND)}</div>'
                f'<div class="agemis-subtitle">{html.escape(APP_SUBTITLE)}</div></div></div>',
                unsafe_allow_html=True, width="stretch")
    # Small gear in the top right corner: opens / closes the panel with the pipeline controls.
    with st.container(key="agemis_gear", width=30):  # styled via .st-key-agemis_gear
        components.html(
            """
    <style>
        html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; }
    #gear { width: 30px; height: 30px; padding: 0; border: 0; border-radius: 6px; background: transparent;
            cursor: pointer; display: flex; align-items: center; justify-content: center; }
    #gear:hover { background: #f1f5f9; }
    </style>
    <button id="gear" title="Settings and pipeline controls" aria-label="Settings and pipeline controls">
      <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
    </button>
    <script>
    document.getElementById("gear").addEventListener("click", function () {
      var doc = window.parent.document;
      var open = doc.querySelector('[data-testid="stExpandSidebarButton"]');
      if (open) { open.click(); return; }
      var close = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
      if (close) { (close.querySelector("button") || close).click(); }
    });
    // Hover windows of the signal boxes in the module pop-ups (phase_a/ui.py): placed on screen next to the
    // hovered box (below it, or above when there is more room there) so the pop-up's edge never cuts them off.
    // Installed once in the main page, so it keeps working when this small frame is redrawn.
    (function () {
      var doc = window.parent.document;
      if (doc.getElementById("agemis-signal-hover")) return;
      var script = doc.createElement("script");
      script.id = "agemis-signal-hover";
      script.textContent = "(" + function () {
        function place(event) {
          var box = event.target.closest && event.target.closest(".agemis-signal");
          if (!box) return;
          var pop = box.querySelector(".agemis-signal-pop");
          if (!pop || pop.contains(event.target)) return;
          var grid = box.parentElement.getBoundingClientRect(), r = box.getBoundingClientRect();
          var vh = window.innerHeight, below = vh - r.bottom - 12, above = r.top - 12;
          pop.style.left = grid.left + "px";
          pop.style.width = grid.width + "px";
          if (below >= 260 || below >= above) {
            pop.style.top = (r.bottom - 1) + "px"; pop.style.bottom = "auto";
            pop.style.maxHeight = Math.max(120, Math.min(below, 420)) + "px";
          } else {
            pop.style.bottom = (vh - r.top - 1) + "px"; pop.style.top = "auto";
            pop.style.maxHeight = Math.max(120, Math.min(above, 420)) + "px";
          }
        }
        document.addEventListener("mouseover", place, true);
        document.addEventListener("focusin", place, true);
      } + ")();";
      doc.head.appendChild(script);
    })();
    </script>
    """,
            width=30,
            height=30,
        )

# --- Define Schema for Graph Mutation (The JSON-Delta Approach) ---
class GraphMutationDelta(BaseModel):
    action_type: str = Field(description="Type of mutation: 'UPDATE_NODE', 'UPDATE_EDGE', 'ADD_EDGE', 'MERGE_NODES'")
    target_identifier: str = Field(description="Name of the node or source->target edge to modify")
    proposed_cypher: str = Field(description="The exact Cypher query to execute this mutation in FalkorDB")
    reasoning: str = Field(description="Why this change is necessary based on the expert's input")

mutation_parser = JsonOutputParser(pydantic_object=GraphMutationDelta)

intent_prompt = PromptTemplate(
    template="""You are the AGEMIS Graph Chat.
The user will either ask a question about the graph or request to modify it.

If they ask a question, respond with: {{"intent": "ask", "query": "..."}}
If they want to modify the graph (change weights, add nodes/edges), respond with: {{"intent": "modify", "query": "..."}}

User input: {input}
Output JSON:""",
    input_variables=["input"]
)

mutation_prompt = PromptTemplate(
    template="""You are the Policy Synthesizer.
The expert wants to modify the Phase B Combinatorial Horizon Graph.
Translate their request into a GraphMutationDelta JSON object containing the exact Cypher query needed.

Expert request: {input}
{format_instructions}""",
    input_variables=["input"],
    partial_variables={"format_instructions": mutation_parser.get_format_instructions()}
)

horizon_summary_prompt = PromptTemplate(
    template="""You brief a central-bank board member who has 20 seconds to read.

Based on the horizon scan below, tell them how the world looks right now for financial stability and systemic risk.

Rules:
- At most 3 sentences and at most 60 words in total.
- Short sentences, one idea each. Plain everyday words.
- No lists packed into a sentence, no semicolons, no dashes, no parentheses, no abbreviations.
- Do not try to cover everything. Pick the 2 or 3 things that matter most for a central bank.
- End with the one thing to watch most closely.
- Never mention graphs, modules, nodes, connections, scores or scenarios.

Horizon modules:
{modules}

Key connections:
{connections}

Central vulnerability:
{central_vulnerability}

Latest scenario context:
{scenarios}

Latest evaluation context:
{evaluation}""",
    input_variables=["modules", "connections", "central_vulnerability", "scenarios", "evaluation"],
)

summary_llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("API_KEY", "not-needed-local"),
    model_name=os.getenv("MODEL_NAME", "gemma-4"),
    request_timeout=int(os.getenv("HORIZON_SUMMARY_TIMEOUT_SECONDS", "8")),
)


# Chat client: same endpoint and model as the pipeline, but with a bounded wait. Without a timeout
# the OpenAI client waits up to 10 minutes per call and retries twice, so an unreachable endpoint
# froze the chat for a very long time before showing "Request timed out".
# Low reasoning effort: the models on the BFH endpoint otherwise "think" for hundreds of tokens
# before writing the Cypher query (~16-24 s instead of ~3 s), with no better queries.
llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("API_KEY", "not-needed-local"),
    model_name=os.getenv("MODEL_NAME", "gemma-4"),
    timeout=int(os.getenv("CHAT_TIMEOUT_SECONDS", "60")),
    max_retries=1,
    reasoning_effort=os.getenv("CHAT_REASONING_EFFORT", "low"),
)
# The final answer keeps the model's normal reasoning (only a few seconds): with low effort it
# tended to mention just one of several results returned by the graph.
answer_llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("API_KEY", "not-needed-local"),
    model_name=os.getenv("MODEL_NAME", "gemma-4"),
    timeout=int(os.getenv("CHAT_TIMEOUT_SECONDS", "60")),
    max_retries=1,
)


def load_horizon_summary_context():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_names = (
        "main_state.json",
        "main_state_v1.json",
        "main_state_2025.json",
    )
    state = None
    for candidate_name in candidate_names:
        candidate_path = os.path.join(base_dir, candidate_name)
        try:
            with open(candidate_path, "r", encoding="utf-8") as state_file:
                candidate = json.load(state_file)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(candidate, dict):
            continue
        horizon_map = candidate.get("horizon_map")
        modules = candidate.get("modules") or candidate.get("approved_modules") or []
        if isinstance(horizon_map, dict) and horizon_map.get("knowledge_graph_edges") and modules:
            state = candidate
            break
    if state is None:
        return None, None

    horizon_map = state.get("horizon_map")
    modules = state.get("modules") or state.get("approved_modules") or []
    if not isinstance(horizon_map, dict) or not modules:
        return None, None

    edges = horizon_map.get("knowledge_graph_edges") or []
    module_context = []
    for module in modules[:36]:
        if isinstance(module, dict):
            name = module.get("name", "Unnamed module")
            reasoning = module.get("reasoning", "")
            certainty = module.get("certainty_level", "")
            module_context.append(f"- {name}. {reasoning}".strip())
            if certainty:
                module_context[-1] += f" Certainty: {certainty}."

    connection_context = []
    for edge in edges[:60]:
        if isinstance(edge, dict):
            source = edge.get("source_module", "")
            target = edge.get("target_module", "")
            mechanism = edge.get("mechanism") or edge.get("interaction_type") or ""
            direction = edge.get("causal_directionality") or ""
            connection_context.append(
                f"- {source} -> {target}: {mechanism}. Direction: {direction}."
            )

    scenario_context = []
    for scenario in (state.get("scenarios") or [])[:6]:
        if isinstance(scenario, dict):
            title = scenario.get("title") or "Untitled scenario"
            narration = scenario.get("narration") or scenario.get("description") or ""
            scenario_context.append(f"- {title}: {narration}")

    evaluation = state.get("evaluation") or state.get("scenario") or {}
    if isinstance(evaluation, dict):
        evaluation_context = json.dumps(evaluation, ensure_ascii=False)[:3000]
    elif evaluation:
        evaluation_context = str(evaluation)[:3000]
    else:
        evaluation_context = "Not specified."

    context = {
        "modules": "\n".join(module_context),
        "connections": "\n".join(connection_context),
        "central_vulnerability": horizon_map.get("central_vulnerability") or "Not specified.",
        "scenarios": "\n".join(scenario_context) or "Not specified.",
        "evaluation": evaluation_context,
    }
    # The prompt is part of the signature, so editing it regenerates the cached overview.
    signature_source = json.dumps(
        {"context": context, "prompt": horizon_summary_prompt.template}, ensure_ascii=False, sort_keys=True
    )
    signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()
    return context, signature

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_mutation" not in st.session_state:
    st.session_state.pending_mutation = None
if "horizon_summary_signature" not in st.session_state:
    st.session_state.horizon_summary_signature = None
if "horizon_summary" not in st.session_state:
    st.session_state.horizon_summary = None

SUMMARY_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "horizon_summary.json")


def load_cached_horizon_summary(signature):
    """Return the pre-computed overview if it was generated for this exact graph state."""
    try:
        with open(SUMMARY_CACHE_PATH, "r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return cached.get("summary")
    return None


def save_cached_horizon_summary(signature, summary):
    try:
        os.makedirs(os.path.dirname(SUMMARY_CACHE_PATH), exist_ok=True)
        with open(SUMMARY_CACHE_PATH, "w", encoding="utf-8") as cache_file:
            json.dump({"signature": signature, "summary": summary}, cache_file, ensure_ascii=False, indent=2)
    except OSError:
        pass


horizon_context, horizon_signature = load_horizon_summary_context()
if (
    horizon_context
    and horizon_signature
    and st.session_state.horizon_summary_signature != horizon_signature
    and (cached_summary := load_cached_horizon_summary(horizon_signature))
):
    st.session_state.horizon_summary = cached_summary
    st.session_state.horizon_summary_signature = horizon_signature

if (
    horizon_context
    and horizon_signature
    and st.session_state.horizon_summary_signature != horizon_signature
):
    try:
        with st.spinner("Preparing a horizon summary..."):
            summary_prompt_text = horizon_summary_prompt.format(
                modules=horizon_context["modules"],
                connections=horizon_context["connections"],
                central_vulnerability=horizon_context["central_vulnerability"],
                scenarios=horizon_context["scenarios"],
                evaluation=horizon_context["evaluation"],
            )
            executor = ThreadPoolExecutor(max_workers=1)
            summary_future = executor.submit(summary_llm.invoke, summary_prompt_text)
            summary_response = summary_future.result(
                timeout=int(os.getenv("HORIZON_SUMMARY_TIMEOUT_SECONDS", "8"))
            )
            executor.shutdown(wait=False)
        st.session_state.horizon_summary = summary_response.content.strip()
        st.session_state.horizon_summary_signature = horizon_signature
        save_cached_horizon_summary(horizon_signature, st.session_state.horizon_summary)
    except Exception as exc:
        st.session_state.horizon_summary = (
            "The horizon overview could not be generated because the language service "
            "is temporarily unavailable. Please try again shortly."
        )
        st.session_state.horizon_summary_signature = horizon_signature

# Connect to Graph
kg = get_falkor_graph()
db_status = "🟢 Connected" if kg else "🔴 Disconnectd (Run docker FalkorDB)"

st.sidebar.markdown(f"**Database Status:** {db_status}")
if os.getenv("AGEMIS_HIDE_PIPELINE") != "1":
    render_pipeline_controls()

INTRO_HTML = """
<div class="agemis-intro">
  <p class="agemis-lead">AGEMIS helps you see where the world stands today and how it could change, from the point of view of
  financial stability. AI agents read signals, such as news, reports and other events, across the social,
  technological, economic, environmental, political and science (STEEPS) domains, and turn them into a
  picture you can use for policy and supervision.</p>
  <p>Use the tabs in order. Each one builds on the one before:</p>
  <div class="agemis-steps">
    <div class="agemis-step"><b>1 · Modules</b>Signals combined into meaningful blocks.</div>
    <div class="agemis-arrow" aria-hidden="true">→</div>
    <div class="agemis-step"><b>2 · Horizon</b>Modules connected, drawing the current status of the world.</div>
    <div class="agemis-arrow" aria-hidden="true">→</div>
    <div class="agemis-step"><b>3 · Scenarios</b>Plausible futures that could grow out of the horizon.</div>
  </div>
  <p class="agemis-note"><b>Contrastive scenarios</b> is a scenario method still under development. It looks for
  surprises that travel across domains.</p>
</div>
<style>
.agemis-intro { width: 100%; color: #1f2937; font-size: 0.95rem; line-height: 1.5; margin: 0 0 0.5rem 0; }
.agemis-intro p { margin: 0 0 0.6rem 0; }
.agemis-intro .agemis-lead { margin-bottom: calc(0.6rem + 1.5em); }  /* one blank line before the tab guide */
.agemis-steps { display: flex; align-items: stretch; gap: 0.5rem; margin: 0 0 0.7rem 0; }
.agemis-step { flex: 1 1 0; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
               padding: 0.6rem 0.8rem; font-size: 0.9rem; color: #334155; }
.agemis-step b { display: block; color: #17233c; font-size: 1rem; margin-bottom: 0.2rem; }
.agemis-arrow { align-self: center; color: #94a3b8; font-size: 1.3rem; }
.agemis-intro .agemis-note { color: #475569; font-size: 0.9rem; margin-bottom: calc(0.6rem + 1.5em); }  /* one blank line before the tabs */
/* Tabs drawn as separate boxes; the selected one opens onto the content below. */
[role="tablist"] { gap: 0.6rem !important; border-bottom: 1px solid #cbd5e1 !important; box-shadow: none !important; }
[role="tablist"]::before, [role="tablist"]::after { display: none !important; }
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none !important; }
[data-testid="stTab"] { height: auto !important; padding: 0.55rem 1.3rem !important; margin-bottom: -1px !important;
  background: #f1f5f9 !important; border: 1px solid #cbd5e1 !important; border-radius: 8px 8px 0 0 !important;
  color: #475569 !important; cursor: pointer; }
[data-testid="stTab"] p { font-size: 1.05rem !important; font-weight: 700 !important; color: inherit !important; }
[data-testid="stTab"]:hover { background: #e2e8f0 !important; color: #17233c !important; }
[data-testid="stTab"][aria-selected="true"] { background: #ffffff !important; color: #17233c !important;
  border-bottom-color: #ffffff !important; box-shadow: inset 0 3px 0 #1d4ed8 !important; }
@media (max-width: 600px) {  /* phones: tabs wrap onto two rows as rounded boxes instead of scrolling */
  [role="tablist"] { flex-wrap: wrap !important; row-gap: 0.45rem !important; border-bottom: 0 !important;
                     overflow: visible !important; padding-bottom: 0.3rem; }
  [data-testid="stTab"] { padding: 0.45rem 0.9rem !important; margin-bottom: 0 !important; border-radius: 8px !important; }
  [data-testid="stTab"] p { font-size: 0.98rem !important; }
  [data-testid="stTab"][aria-selected="true"] { border-color: #1d4ed8 !important; box-shadow: none !important; }
}
@media (max-width: 760px) {
  .agemis-steps { flex-direction: column; }
  .agemis-arrow { transform: rotate(90deg); line-height: 1; }
}
</style>
"""
st.markdown(INTRO_HTML, unsafe_allow_html=True)

modules_tab, graph_tab, scenarios_tab, contrastive_tab = st.tabs(
    ["Modules", "Horizon", "Scenarios", "Contrastive scenarios"]
)
with modules_tab:
    render_modules()
with scenarios_tab:
    render_scenarios()
with contrastive_tab:
    render_scenarios_v2()

col1, col2 = graph_tab.columns([1.5, 1])

with col1:
    st.markdown(
        """
<div style="border: 1px solid #d9e2ec; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background-color: #f8fafc;">
  <div style="display:flex; flex-wrap:wrap; gap:12px 24px; align-items:flex-start; color:#1f2937; font-size:13px;">
    <div style="min-width:280px; flex:1.4;">
      <strong>Modules (made of signals)</strong>
      <div style="margin-top:5px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#00d8ff;margin-right:6px;"></span><strong>Risk driver</strong> — sends influence to other modules.</div>
      <div style="margin-top:3px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#bd00ff;margin-right:6px;"></span><strong>Vulnerability</strong> — receives influence from other modules.</div>
      <div style="margin-top:3px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#7b2cbf;margin-right:6px;"></span><strong>Systemic hub</strong> — sends and receives influence.</div>
      <div style="margin-top:3px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#888888;margin-right:6px;"></span><strong>Stand-alone module</strong> — no connection yet.</div>
      <div style="margin-top:3px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#fff7d6;border:2px dashed #b7791f;margin-right:6px;"></span><strong>Emerging module</strong> — provisional, with limited evidence.</div>
    </div>
    <div style="min-width:260px; flex:1;">
      <strong>Connections</strong>
      <div style="margin-top:5px;">The number and line thickness show empirical trend strength from 1 (limited evidence) to 10 (strong evidence). One arrow shows influence in one direction; two arrows show a reinforcing relationship in both directions.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Load and render the pyvis graph generated by Phase B
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "out", "horizon_graph.html")
    if not os.path.exists(graph_path):
        # Fallback if generated in rot
        graph_path = os.path.join(base_dir, "horizon_graph.html")
        
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_html = f.read()
        # Fix to scroll in long modules
        tooltip_css = """
<style>
div.vis-tooltip {
    max-width: 480px;
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap !important;
    word-wrap: break-word;
    pointer-events: auto;
    z-index: 10;
}
</style>
"""
        graph_html = graph_html.replace("</head>", tooltip_css + "</head>", 1)
        # Show the whole graph on first view (same as the zoom-extends button). The graph is drawn
        # while its tab is still hidden (zero size), so refit whenever the drawing area gets a real
        # size and once the physics layout has settled, unless the user already zoomed or dragged.
        fit_on_load_js = """
<script>
(function () {
  if (typeof network === "undefined") return;
  var container = document.getElementById("mynetwork");
  var userMoved = false;
  function fitAll() {
    if (userMoved || !container || container.clientWidth === 0 || container.clientHeight === 0) return;
    network.redraw();
    network.fit({ animation: false });
  }
  network.on("zoom", function () { userMoved = true; });
  network.on("dragStart", function () { userMoved = true; });
  network.once("stabilizationIterationsDone", fitAll);
  network.on("stabilized", fitAll);
  if (window.ResizeObserver && container) { new ResizeObserver(fitAll).observe(container); }
  [300, 1000, 2500].forEach(function (ms) { setTimeout(fitAll, ms); });
})();
</script>
"""
        graph_html = graph_html.replace("</body>", fit_on_load_js + "</body>", 1)
        components.html(graph_html, height=820)
    else:
        st.warning(f"Graph not generated yet ({graph_path}). Build the horizon from the panel behind the gear first.")

with col2:
    with st.container(border=True):
        st.markdown(
            """
<div style="display:flex; align-items:center; gap:10px; padding:4px 2px 12px 2px;">
  <div style="width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
              background:linear-gradient(135deg,#1d4ed8,#7c3aed); color:white; font-size:27px;
              box-shadow:0 4px 10px rgba(67,56,202,.25);">🤖</div>
  <div>
    <div style="font-size:20px; line-height:1.1; font-weight:700; color:#17233c;">Human-In-The-Loop Chat</div>
    <div style="margin-top:4px; font-size:13px; color:#64748b;">Discuss this horizon and review proposed changes.</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.session_state.horizon_summary:
            summary_text = html.escape(st.session_state.horizon_summary).replace("\n", "<br>")
            st.markdown(
                f"""
<div style="border-left:4px solid #7c3aed; border-radius:8px; background:#f5f3ff;
            padding:12px 14px; margin:0 0 14px 0; color:#1f2937; font-size:14px; line-height:1.5;">
  <strong style="color:#5b21b6;">Horizon overview</strong><br>
  {summary_text}
</div>
""",
                unsafe_allow_html=True,
            )
        elif horizon_context:
            st.caption("The horizon summary is being prepared.")
        else:
            st.caption("Build the horizon graph to generate the overview.")

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "mutation" in msg:
                    mut = msg["mutation"]
                    st.markdown(f"**Action:** {mut['action_type']}\n**Target:** {mut['target_identifier']}")
                    st.info(mut["reasoning"])
                    st.code(mut["proposed_cypher"], language="cypher")
                    if msg.get("executed"):
                        st.success("✅ Executed in graph")
                else:
                    st.markdown(msg["content"])
                
    # Pending Mutation Approval Card
    if st.session_state.pending_mutation:
        st.warning("⚠️ Expert Validation Requird")
        mut = st.session_state.pending_mutation
        st.code(mut["proposed_cypher"], language="cypher")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Approve & Execute"):
                if kg:
                    try:
                        kg.query(mut["proposed_cypher"])
                        st.success("Graph updated in FalkorDB!")
                        st.session_state.messages[-1]["executed"] = True
                    except Exception as e:
                        st.error(f"FalkorDB Error: {e}")
                else:
                    st.error("Cannot execute: FalkorDB is disconnected.")
                    
                st.session_state.pending_mutation = None
                st.rerun()
                
        with col_btn2:
            if st.button("Reject"):
                st.session_state.pending_mutation = None
                st.session_state.messages.append({"role": "assistant", "content": "Mutation rejected by expert."})
                st.rerun()

    # Chat Input
    with st.container(border=True):
        prompt = st.chat_input("Interact with the horizon graph (ask or request changes)")

    if prompt:
        if st.session_state.pending_mutation:
            st.error("Please approve or reject the pending mutation first.")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent..."):
                try:
                    intent_res = llm.invoke(intent_prompt.format(input=prompt))
                    # Parse simple JSON from LLM response
                    intent_text = intent_res.content.replace("```json", "").replace("```", "").strip()
                    intent_data = json.loads(intent_text)
                    intent = intent_data.get("intent", "ask")
                except Exception as e:
                    intent = "ask" # Fallback to asking if parsing fails
            
            if intent == "modify":
                with st.spinner("Formulating Graph Delta..."):
                    try:
                        chain = mutation_prompt | llm | mutation_parser
                        mutation = chain.invoke({"input": prompt})
                        
                        # Add to state & prompt for refresh of UI to show validation card
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "Proposed a graph mutation. Awaiting expert validation.",
                            "mutation": mutation,
                            "executed": False
                        })
                        st.session_state.pending_mutation = mutation
                        st.rerun()
                    except Exception as e:
                        err_msg = f"Failed to generate mutation JSON: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                with st.spinner("Querying Knowledge Graph..."):
                    if kg:
                        try:
                            from langchain_community.graphs import FalkorDBGraph
                            from langchain_community.chains.graph_qa.falkordb import FalkorDBQAChain
                            from langchain_core.prompts import PromptTemplate
                            
                            # 1. Standalone Question Generator (Memory)
                            history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1][-4:]])
                            if history.strip():
                                rephrase_prompt = f"Chat History:\n{history}\n\nUser: {prompt}\nRewrite the user's input as a standalone question with context. Return only the question."
                                standalone_prompt = llm.invoke(rephrase_prompt).content.strip()
                            else:
                                standalone_prompt = prompt
                                
                            # 2. AGEMIS-Aware Cypher Prompt
                            cypher_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Schema:
{schema}

Note: In this AGEMIS graph, edges represent interactions between modules. 
- 'empirical_trend' is a weight 1-10 showing trend momentum.
- 'narrative' is market attention.
- 'institutional' is regulatory impact.
- The relationship 'type' property describes the action (e.g., 'mitigates', 'amplifies', 'causes leakages').

CRITICAL CYPHER RULES:
1. When the user asks about 'trend', they mean 'empirical_trend'.
2. ALWAYS use case-insensitive fuzzy matching for node names because user input often omits words. Use `WHERE toLower(n.name) CONTAINS toLower("...")` instead of exact string matching (`n.name = "..."`).
3. Always RETURN the node names (source and target) along with the edge properties so you can explain the relationship.
4. DIRECTION: (a:Module)-[:INTERACTS]->(b:Module) means a drives / influences / affects b.
   - "What drives / influences / feeds into X?" -> X is the TARGET:
     MATCH (s:Module)-[r:INTERACTS]->(t:Module) WHERE toLower(t.name) CONTAINS toLower("X") RETURN s.name, t.name, r.type, r.empirical_trend
   - "What does X drive / affect / influence?" -> X is the SOURCE:
     MATCH (s:Module)-[r:INTERACTS]->(t:Module) WHERE toLower(s.name) CONTAINS toLower("X") RETURN s.name, t.name, r.type, r.empirical_trend
   - "What is X connected to?" or unclear direction -> match both directions with -[r:INTERACTS]- and return both names.
5. Do not add LIMIT unless the question asks for a specific number of results.

Question: {question}"""
                            cypher_prompt = PromptTemplate(
                                template=cypher_template,
                                input_variables=["schema", "question"]
                            )
                            
                            falkor_graph = FalkorDBGraph(
                                database="main_kg",
                                host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
                                port=int(os.getenv("FALKORDB_PORT", "6379")),
                                password=os.getenv("FALKORDB_PASSWORD") or None,
                            )
                            qa_chain = FalkorDBQAChain.from_llm(
                                llm=answer_llm,
                                graph=falkor_graph,
                                cypher_prompt=cypher_prompt,
                                verbose=True,
                                allow_dangerous_requests=True
                            )
                            # Fast, low-reasoning model writes the query; answer_llm writes the reply.
                            qa_chain.cypher_generation_chain.llm = llm
                            
                            response = qa_chain.invoke({"query": standalone_prompt})
                            ans = response.get("result", "No answer found.")
                        except Exception as e:
                            ans = f"GraphRAG Error: Could not execute Cypher QA. {str(e)}"
                    else:
                        ans = "GraphRAG is disabled because FalkorDB is offline. Answering using baseline knowledge.\n\n"
                        fallback_response = llm.invoke(prompt)
                        ans += fallback_response.content
                        
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
