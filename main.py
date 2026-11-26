import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
import re
from typing import Any

class RobustJsonOutputParser(JsonOutputParser):
    """Retry once after deleting trailing commas from bad JSON."""

    def parse_result(self, result: list, *, partial: bool = False) -> Any:
        try:
            return super().parse_result(result, partial=partial)
        except Exception as original:
            if not result or not hasattr(result[0], 'text'):
                raise
            text = result[0].text
            cleaned = re.sub(r',\s*\}', '}', text)
            cleaned = re.sub(r',\s*\]', ']', cleaned)
            if cleaned == text:
                raise
            try:
                # parse() goes back to this override - call the base method direct.
                return super().parse_result([Generation(text=cleaned)], partial=partial)
            except Exception:
                raise original

import sys
from pydantic import BaseModel, Field

try:
    from falkordb import FalkorDB
    import redis.exceptions
    FALKOR_ENABLED = True
except ImportError:
    FALKOR_ENABLED = False

def get_falkor_graph():
    if not FALKOR_ENABLED: return None
    try:
        db = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "127.0.0.1"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
            password=os.getenv("FALKORDB_PASSWORD") or None,
        )
        return db.select_graph('main_kg')
    except (ConnectionRefusedError, redis.exceptions.ConnectionError):
        print("[FalkorDB] Connection refused. Is the FalkorDB Docker container running? Falling back to in-memory state.")
        return None
def clear_falkor_graph():
    """Delete the FalkorDB knowledge graph so a full reprocess restarts from scratch."""
    kg = get_falkor_graph()
    if kg is None:
        return
    try:
        kg.delete()
        print("[FalkorDB] Cleared graph 'main_kg'.")
    except redis.exceptions.ResponseError as exc:
        print(f"[FalkorDB] Didn't clear graph: {exc}")

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("API_KEY", "not-needed-local"),
    model_name=os.getenv("MODEL_NAME", "gemma-4"),
)

# Define Data Structures for Phases
class SignalEvidence(BaseModel):
    title: str = Field(description="Title of the raw input signal")
    content: str = Field(description="Brief summary of the symptom or evidence")

class Module(BaseModel):
    name: str = Field(description="Name of the overarching structural subsystem or 3D-Module (e.g., 'Global Tech-Decoupling')")
    velocity: str = Field(description="Vector for speed/velocity of this structural change")
    inertia: str = Field(description="Vector for inertia/resistance/structural momentum of this subsystem")
    steeps_intersections: List[str] = Field(description="STEEPS domains (Social, Tech, Econ, Env, Pol, Sys) intersecting in this module")
    evidence_signals: List[SignalEvidence] = Field(description="The raw signals integrated as proof-points or symptoms of this module")
    impact_potential: str = Field(description="High/Medium/Low potential systemic impact")
    certainty_level: str = Field(description="Degree of certainty based on the combined evidence")
    sentiment: str = Field(description="Sentiment of the overall subsystem (Negative/Neutral/Positive)")
    temporal_horizon: str = Field(description="Expected time of impact (Immediate, Short-term, Medium-term, Long-term)")
    novelty_score: int = Field(description="Score 1-10 on how unique/new this structural shift is relative to the baseline")
    novelty_rationale: str = Field(description="Detailed explanation of why this shift is novel")
    reasoning: str = Field(description="Detailed breakdown of how the raw signals were synthesized into this 3D module")

class ModuleList(BaseModel):
    modules: List[Module] = Field(description="List of detected modules")

class UpdateModuleAction(BaseModel):
    module_name: str = Field(description="Name of the existing module to update")
    updated_velocity: str = Field(description="New velocity vector")
    updated_inertia: str = Field(description="New inertia vector")
    updated_impact_potential: str = Field(description="New impact potential")
    updated_certainty_level: str = Field(description="New certainty level")
    reasoning: str = Field(description="Reasoning for this update and how the signal impacts the vectors")

class MergeModuleAction(BaseModel):
    module_a: str = Field(description="Name of the first existing module to merge")
    module_b: str = Field(description="Name of the second existing module to merge")
    merged_module: Module = Field(description="The newly synthesized 3D module resulting from the merge")
    reasoning: str = Field(description="Why these two modules are fundamentally the same structural subsystem")

# New: Action to discard an event mostly interesting for audit and HITL
class DiscardAction(BaseModel):
    event_title: str = Field(description="Title of the discarded event")
    reasoning: str = Field(description="Why this event was deemed noise or irrelevant to systemic risk.")

class SynthesisResult(BaseModel):
    updates: List[UpdateModuleAction] = Field(description="Updates to apply to existing modules")
    creations: List[Module] = Field(description="Entirely new modules to create")
    merges: Optional[List[MergeModuleAction]] = Field(default=None, description="Optional list of merges to consolidate overlapping modules")
    discards: Optional[List[DiscardAction]] = Field(default=None, description="List of events discarded as noise or irrelevant")

class GraphEdge(BaseModel):
    source_module: str = Field(description="Name of the source module")
    target_module: str = Field(description="Name of the target module")
    interaction_type: str = Field(description="Short label of 2-5 words for the interaction (e.g., 'raises funding costs')")
    polarity: str = Field(description="'amplifying' if stress in the source increases stress in the target, 'buffering' if it reduces it")
    mechanism: str = Field(description="One sentence naming the concrete transmission channel from source to target")
    evidence: List[str] = Field(description="Evidence signal titles copied exactly from the two connected modules that support this link; empty if the link is inferred")
    causal_directionality: str = Field(description="Direction of causality (e.g., Unidirectional Source->Target, or Bidirectional feedback loop)")
    empirical_trend_strength: int = Field(description="Weight from 1-10 indicating trend momentum")
    narrative_salience: int = Field(description="Weight from 1-10 indicating public or market attention")
    institutional_relevance: int = Field(description="Weight from 1-10 indicating regulatory/supervisory impact")

class HorizonMap(BaseModel):
    knowledge_graph_edges: List[GraphEdge] = Field(description="Structured edges forming the horizon knowledge graph")
    inflexion_weak_signals: List[str] = Field(description="Identified weak signals from where inflexion futures might emerge")
    perturbation_sensitivities: List[str] = Field(description="Specific nodes or edges identified as highly sensitive to small shocks (Module Perturbations)")
    central_vulnerability: str = Field(description="The core unstable present configuration carrying future implications")

class Scenario(BaseModel):
    title: str = Field(description="Name of the future scenario")
    narration: str = Field(description="Detailed plausible future trajectory")
    counterfactual_assumption: str = Field(description="The 'What If' non-linear assumption that diverges from simple trend extrapolation")
    novelty_element: str = Field(description="The surprising shift or tipping point in the scenario")
    trigger_signal: str = Field(description="The specific 'inflexion_weak_signal' from Phase B that triggered this trajectory")
    dynamic_weights_utilization: str = Field(description="Explanation of how empirical, narrative, and institutional weights from Phase B influenced the scenario logic")

class Evaluation(BaseModel):
    systemic_risk_score: int = Field(description="Score from 1-10 on systemic threat")
    plausibility_rationale: str = Field(description="Why this future is plausible")
    policy_action: str = Field(description="Recommended supervisory response")
    policy_cypher_simulation: Optional[str] = Field(default=None, description="A Cypher query that simulates the policy intervention by mutating the graph (e.g., deleting a risky edge or isolating a node).")

# Define the State for the LangGraph
class AgentState(TypedDict):
    input_events: List[Dict[str, Any]]
    baseline_knowledge: str # The 'Business as Usual' knowledge state
    modules: List[Dict[str, Any]]
    approved_modules: List[Dict[str, Any]]
    active_modules: Optional[List[str]] # Modules touched (created/updated/merged) in this run
    horizon_map: Optional[Dict[str, Any]]
    scenario: Optional[Dict[str, Any]]
    scenarios: Optional[List[Dict[str, Any]]] # Full scenario set from the Phase C world model
    evaluation: Optional[Dict[str, Any]]
    expert_guidance: Optional[str]             # HITL feedback text passed into the Phase B mapper prompt
    weak_signals: Optional[List[Dict[str, Any]]]  # Phase A V2 implications retained as weak signals
    stale: Optional[Dict[str, str]]            # Artifact name -> reason for regeneration
    proto_modules: Optional[List[Dict[str, Any]]]  # Provisional groups of weak implications
    include_emerging: Optional[bool]           # Include provisional nodes in Phase-B
# --- HITL Node ---
def dummy_hitl_approval_node(state: AgentState):
    print("\n" + "="*50)
    print("--- [HITL] EXPERT APPROVAL ---")
    print(f"INPUT (modules): {json.dumps(state['modules'], indent=2)}")
    
    # In a real app, this would pause and wait for user input
    print("HUMAN ACTION: Reviewing and approving all modules...")
    approved = state["modules"]
    active = state.get("active_modules") or []
    
    print(f"OUTPUT (approved): {json.dumps(approved, indent=2)}")
    return {"approved_modules": approved, "active_modules": active}

# --- Build LangGraph ---
def build_state_graph():
    # Import nodes inside to avoid circular dependencies
    from phase_a_decomposer import decompose_node
    from phase_b_mapper import map_node
    from phase_c_generator import generate_scenario_node
    from phase_d_evaluator import evaluate_node

    workflow = StateGraph(AgentState)
    
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("hitl_approve", dummy_hitl_approval_node)
    workflow.add_node("map_horizon", map_node)
    workflow.add_node("generate_scenario", generate_scenario_node)
    workflow.add_node("evaluate", evaluate_node)
    
    workflow.add_edge(START, "decompose")
    workflow.add_edge("decompose", "hitl_approve")
    workflow.add_edge("hitl_approve", "map_horizon")
    workflow.add_edge("map_horizon", "generate_scenario")
    workflow.add_edge("generate_scenario", "evaluate")
    workflow.add_edge("evaluate", END)
    
    return workflow.compile()

def run_agemis_poc(input_events: List[Dict[str, Any]], initial_modules: List[Dict[str, Any]] = None, baseline: str = None):
    print("Initializing Stateful LangGraph Pipeline")
    app = build_state_graph()
    
    # Default baseline if none provided
    if not baseline:
        baseline = ""
    
    initial_state = {
        "input_events": input_events,
        "baseline_knowledge": baseline,
        "modules": initial_modules or []
    }
    
    # Run the graph
    final_state = app.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    signals_file = os.path.join(os.path.dirname(__file__), "signals.json")
    state_file = os.path.join(os.path.dirname(__file__), "main_state.json")
    
    try:
        with open(signals_file, "r") as f:
            all_signals = json.load(f)
    except FileNotFoundError:
        print(f"Error: {signals_file} not found. Please create it or run the extraction.")
        all_signals = []

    # Load persisted state
    persisted_state = {}
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            persisted_state = json.load(f)
            
    processed_titles = set(persisted_state.get("processed_events", []))
    existing_modules = persisted_state.get("modules", [])
    existing_horizon_map = persisted_state.get("horizon_map", None)
    
    # Filter for new signals
    new_signals = [s for s in all_signals if s.get("title") not in processed_titles]
    
    force_run = "--force" in sys.argv
    graph_run = "--graph" in sys.argv
    scenarios_run = "--scenarios" in sys.argv
    
    if "--scenarios-v2" in sys.argv:
        # Separate cross-domain scenario set -. leaves v1 scenarios and the evaluation untouched.
        from phase_c.generator_v2 import regenerate_saved_scenarios_v2

        print("Running Phase C v2: cross-domain scenarios from saved modules and horizon graph")
        try:
            scenarios, info = regenerate_saved_scenarios_v2(
                state_file, llm, lambda model: RobustJsonOutputParser(pydantic_object=model))
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(f"Phase C v2 complete. {info['candidates']} candidate paths considered; "
              f"saved {len(scenarios)} scenarios to {state_file}.")
        for scenario in scenarios:
            print(f"    - {scenario['scenario_id']}: '{scenario['title']}' | {scenario['seed']['rationale']}")

    elif scenarios_run:
        if force_run or graph_run:
            print("Error: Use --scenarios on its own to run only Phase C.")
            sys.exit(1)
        from phase_c.regeneration import regenerate_saved_scenarios
        from phase_c_generator import generate_scenario_node

        print("Running Phase C only: regenerating scenarios from saved modules and horizon graph")
        try:
            count = regenerate_saved_scenarios(state_file, generate_scenario_node)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        print(f"Phase C complete. Saved {count} scenarios to {state_file}. Previous evaluation cleared.")

    elif graph_run:
        print("Running Phase B, C, and D: Horizon Mapping, Scenario Generation, and Evaluation")
        if not existing_modules:
            print("Error: No modules found in state file. Run Phase A first to ingest signals and create modules.")
            sys.exit(1)
            
        from phase_b_mapper import map_node
        from phase_c_generator import generate_scenario_node
        from phase_d_evaluator import evaluate_node
        
        # Build Phase B, C, and D Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("map_horizon", map_node)
        workflow.add_node("generate_scenario", generate_scenario_node)
        workflow.add_node("evaluate", evaluate_node)
        
        workflow.add_edge(START, "map_horizon")
        workflow.add_edge("map_horizon", "generate_scenario")
        workflow.add_edge("generate_scenario", "evaluate")
        workflow.add_edge("evaluate", END)
        
        app = workflow.compile()
        
        initial_state = {
            "input_events": [],
            "baseline_knowledge": "",
            "modules": existing_modules,
            "approved_modules": existing_modules,
            "horizon_map": existing_horizon_map,
            # Modules touched by the last Phase A run: enables the delta re-map
            "active_modules": persisted_state.get("active_modules") or None,
            "weak_signals": persisted_state.get("weak_signals") or [],
            "proto_modules": persisted_state.get("proto_modules") or [],
            "include_emerging": "--emerging-nodes" in sys.argv,
        }

        final_state = app.invoke(initial_state)

        # Save the graph, scenarios, and final modules, including downstream updates.
        state_to_save = {
            "model": persisted_state.get("model", os.getenv("MODEL_NAME", "gemma-local")),
            "processed_events": list(processed_titles),
            "modules": final_state.get("modules", existing_modules),
            "active_modules": [],
            "horizon_map": final_state.get("horizon_map", None),
            "scenarios": final_state.get("scenarios", None),
            "evaluation": final_state.get("evaluation", None),
        }
        # Preserve saved Phase A V2 metadata for downstream phases and the UI.
        for key in ("phase_a_version", "weak_signals", "proto_modules", "discarded_signals",
                    "failed_extractions", "phase_a_run_dir", "phase_a_metrics"):
            if key in persisted_state:
                state_to_save[key] = persisted_state[key]
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, indent=2)

        print(f"Phase B, C, and D complete. State persisted to {state_file}")
        
    elif "--force-v2" in sys.argv or "--v2" in sys.argv:
        from phase_a.v2 import PhaseAConfig, latest_run_dir, run_phase_a_v2
        reextract = "--force-v2" in sys.argv
        reuse_run = None if reextract or "--v2-cache-run" in sys.argv else latest_run_dir()
        if reextract:
            print("Force V2: re-extracting all signals with Phase A V2 (extract => cluster => synthesize) ...")
        else:
            print("Phase A V2: extracting new signals only, reusing saved extractions; "
                  "clustering and synthesis run over all signals ...")
        from phase_a.graphrag import get_embedder

        clear_falkor_graph()
        def v2_option(flag):
            if flag not in sys.argv:
                return None
            index = sys.argv.index(flag)
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise ValueError(f"{flag} requires a path")
            return sys.argv[index + 1]

        result = run_phase_a_v2(
            all_signals, llm.bind(temperature=0, seed=42), get_embedder(),
            config=PhaseAConfig.from_env(), run_dir=v2_option("--v2-run-dir"),
            cache_run=v2_option("--v2-cache-run"), reuse_run=reuse_run)
        modules = result["modules"]

        state_to_save = {
            "model": os.getenv("MODEL_NAME", "gemma-local"),
            "phase_a_version": "v2",
            "processed_events": [s.get("title") for s in all_signals
                                 if s not in [f["event"] for f in result["failed_extractions"]]],
            "modules": modules,
            "active_modules": [m["name"] for m in modules],
            "weak_signals": result["weak_signals"],
            "proto_modules": result["proto_modules"],
            "discarded_signals": result["discarded"],
            "failed_extractions": result["failed_extractions"],
            "phase_a_run_dir": result["run_dir"],
            "phase_a_metrics": result["metrics"],
            "horizon_map": None,
        }
        from phase_a.describe import add_module_summaries
        print("Writing plain-language module summaries ...")
        add_module_summaries(modules + list(result["proto_modules"] or []), llm)
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, indent=2)

        print(f"\nPhase A V2 completed: {len(modules)} modules, "
              f"{len(result['weak_signals'])} weak-signal implications, "
              f"{len(result['proto_modules'])} proto-modules, "
              f"{len(result['discarded'])} discarded signals, "
              f"{len(result['failed_extractions'])} failed extractions. Saved {state_file}.")

    elif new_signals or force_run:
        if force_run:
            print("Force flag detected. Clearing FalkorDB graph and reprocessing all signals...")
            clear_falkor_graph()
            signals_to_process = all_signals
        else:
            print(f"Found {len(new_signals)} new signals. Running Phase A (Decomposition) only...")
            signals_to_process = new_signals
            
        from phase_a_decomposer import decompose_node
        
        # Build Phase A Graph (decompose and hitl approve only)
        workflow = StateGraph(AgentState)
        workflow.add_node("decompose", decompose_node)
        workflow.add_node("hitl_approve", dummy_hitl_approval_node)
        
        workflow.add_edge(START, "decompose")
        workflow.add_edge("decompose", "hitl_approve")
        workflow.add_edge("hitl_approve", END)
        
        app = workflow.compile()
        
        initial_state = {
            "input_events": signals_to_process,
            "baseline_knowledge": "",
            # A forced rerun cleared FalkorDB, so start from no modules too; otherwise old modules are invisible to retrieval and get duplicated
            "modules": [] if force_run else existing_modules,
            "horizon_map": existing_horizon_map
        }
        
        final_state = app.invoke(initial_state)
        
        # Save state after Phase A
        new_processed = list(processed_titles.union(s.get("title") for s in signals_to_process))
        state_to_save = {
            "model": os.getenv("MODEL_NAME", "gemma-local"),
            "processed_events": new_processed,
            "modules": final_state.get("modules", []),
            "horizon_map": None  # Graph is cleared and must be regenerated in the next step
        }
        from phase_a.describe import add_module_summaries
        print("Writing plain-language module summaries ...")
        add_module_summaries(state_to_save["modules"], llm)
        with open(state_file, "w") as f:
            json.dump(state_to_save, f, indent=2)
            
        print(f"\nPhase A complete. Processed events and modules saved to {state_file}.")
        print("="*50)
        print("HITL INTERVENTION WINDOW:")
        print("You can now review and edit the modules inside 'main_state.json' by hand,")
        print("Once you are ready, run the script with the '--graph' flag to generate the Horizon Knowledge Graph:")
        print("  python3 main.py --graph")
        print("="*50)
        
    else:
        if existing_horizon_map is None or persisted_state.get("stale"):
            reason = "not yet been generated" if existing_horizon_map is None else "been marked stale"
            print(f"Modules exist in state file, but the Horizon Knowledge Graph has {reason}.")
            print("Run the script with the '--graph' flag to (re)generate the graph:")
            print("  python3 main.py --graph")
        else:
            print("No new signals to process. State is up to date.")
            print("Run with --force to rerun Phase A, or --graph to regenerate the horizon graph.")

