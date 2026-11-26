import json
from time import perf_counter

from phase_c import world_model as wm
from phase_c.financial_stability import evidence_context, validate_scenario
from phase_c.prompts import get_narrator_prompt
from phase_c.schemas import ScenarioNarration

WEIGHTS_NOTE = (
    "Transmission strengths are the composite of Phase B's"
    f" weights ({wm.COUPLING_WEIGHTS}). Buffering-polarity edges reduce the "
    f"target's stress at x{wm.BUFFERING_ATTENUATION} strength."
)

SCENARIO_BUDGET = 3
MAX_SEED_OVERLAP = 0.5   # Treat affected-module Jaccard overlap above this threshold as a repeat.

def trace_to_events(trace: list) -> list:
    return [
        {
            "id": f"e{i}",
            "epoch": event["round"],
            "source_module": event["source"],
            "affected_module": event["target"],
            "interaction_type": event["type"],
            "polarity": event["polarity"],
            "mechanism": event.get("mechanism", ""),
            "evidence_titles": list(event.get("evidence_titles") or []),
            "provisional": bool(event.get("provisional")),
            "stress_delta": event["transmitted"],
            "stress_after": event["stress_after"],
        }
        for i, event in enumerate(trace, start=1)
    ]


def _shock_summary(shock: dict) -> str:
    if shock.get("weak_signal"):
        areas = ", ".join([shock["target_module"], *shock.get("co_targets", [])])
        return ("Consider that this early, so far weakly evidenced development gains momentum: "
                f"{shock['weak_signal']} It would bear on: {areas}")
    return f"Consider a hypothetical adverse development associated with: {shock['target_module']}"

def _format_events(events: list, *, step_label="epoch") -> str:
    lines = []
    for event in events:
        lines.append(f"  [{event['id']}] {step_label} {event['epoch']}: '{event['source_module']}' -> "
                     f"'{event['affected_module']}' ({event['interaction_type']}, {event['polarity']}, "
                     f"{event['stress_delta']:+} stress)")
        relationship = {"mechanism": event.get("mechanism", ""),
                        "evidence_titles": event.get("evidence_titles") or [],
                        "provisional": bool(event.get("provisional"))}
        lines.append("    Supplied relationship: " + json.dumps(relationship, ensure_ascii=False))
    return "\n".join(lines) or "  (the shock did not propagate beyond its seed)"


def _narrate_symbolic(seed: dict, shock: dict, events: list) -> dict:
    """Preserve the trace without presenting it as an analyst-ready scenario."""
    return {
        "title": "Scenario needs a written assessment",
        "narration": "The model could not produce a readable scenario. The underlying evidence and calculated relationships are saved for review.",
        "counterfactual_assumption": "",
        "trigger_signal": seed["rationale"],
        "novelty_element": seed["rationale"],
        "narrated_by": "symbolic",
        "watch_for": [], "key_uncertainty": "", "analyst_question": "",
        "supporting_analysis": [],
    }


def _narrate_with_llm(shock: dict, events: list, stresses: dict, context: dict, llm, parser_factory,
                      failure=None) -> dict:
    chain = get_narrator_prompt() | llm | parser_factory(ScenarioNarration)
    narration = chain.invoke({
        "shock_summary": _shock_summary(shock),
        "events": _format_events(events),
        "financial_context": json.dumps(context, ensure_ascii=False),
    })
    if failure is not None:
        failure.update(stage="validation", rejected_output=narration)
    narration = validate_scenario(narration, events, context)
    narration["narrated_by"] = "llm"
    return narration


def _generate_narration(seed: dict, shock: dict, events: list, stresses: dict,
             context: dict, llm=None, parser_factory=None) -> dict:
    if llm is not None and parser_factory is not None:
        started = perf_counter()
        failure = {"stage": "generation"}
        try:
            narration = _narrate_with_llm(shock, events, stresses, context, llm, parser_factory, failure)
            narration.setdefault("counterfactual_assumption", f"What if a {_shock_summary(shock)}?")
            narration.setdefault("novelty_element", seed["rationale"])
            narration.setdefault("trigger_signal", seed["rationale"])
            print(f"  [Phase C] Narration complete ({perf_counter() - started:.1f}s)", flush=True)
            return narration
        except Exception as e:
            failure["error"] = str(e)[:2000]
            if getattr(e, "llm_output", None) is not None:
                failure.update(stage="parsing", rejected_output=e.llm_output)
            reason = failure["error"].partition("\n")[0] or type(e).__name__
            print(f"  [Phase C] Narration failed ({perf_counter() - started:.1f}s, "
                  f"{failure['stage']}): {reason}", flush=True)
            return {**_narrate_symbolic(seed, shock, events), "narration_failure": failure}
    return _narrate_symbolic(seed, shock, events)


def _weak_signal_seed(modules: list, weak_signals: list, ranked: list):
    """Choose a two-module weak-signal seed by domain overlap, then combined seed score."""
    score_of = {seed["nodes"][0]: seed["score"] for seed in ranked}
    domains_of = {m.get("name"): set(m.get("steeps_intersections") or []) for m in modules or []}
    best = None
    for weak in weak_signals or []:
        if not isinstance(weak, dict) or not weak.get("statement"):
            continue
        weak_domains = set(weak.get("domains") or [])
        matches = sorted(
            ((len(weak_domains & domains_of.get(name, set())), score_of[name], name) for name in score_of),
            reverse=True,
        )[:2]
        if len(matches) < 2 or matches[1][0] == 0:
            continue
        key = (sum(m[0] for m in matches), sum(m[1] for m in matches))
        if best is None or key > best[0]:
            best = (key, weak, [m[2] for m in matches])
    if best is None:
        return None
    key, weak, nodes = best
    return {
        "motif_type": "weak_signal_trigger",
        "nodes": nodes,
        "edges": [],
        "score": round(key[1], 3),
        "rationale": f"Weak signal: {weak['statement']}",
        "weak_signal": weak,
    }


def _overlap(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _build_scenario(index, seed, shock, narration, events, stresses, activation_round):
    scenario = {key: narration[key] for key in
                ("title", "narration", "counterfactual_assumption",
                 "novelty_element", "trigger_signal")}
    scenario.update({
        "dynamic_weights_utilization": WEIGHTS_NOTE,
        "scenario_id": f"scn_{index}_{seed['motif_type']}",
        "seed": seed,
        "shock": shock,
        "seeds": {node: shock["intensity"] for node in seed["nodes"]},
        "events": events,
        "stresses": {n: v for n, v in stresses.items() if v >= wm.ACTIVATION_THRESHOLD},
        "activation_round": activation_round,
        "narrated_by": narration["narrated_by"],
        "generator": "world_model_v1_symbolic",
        "presentation_version": 2,
        **{key: narration[key] for key in
           ("watch_for", "key_uncertainty", "analyst_question", "supporting_analysis")},
    })
    # Diagnostics: says why a scenario could not be writtn or which references were repaired.
    for key in ("narration_failure", "reference_repairs"):
        if narration.get(key):
            scenario[key] = narration[key]
    return scenario


def generate_scenarios(horizon_map: dict, modules: list, llm=None,
                       parser_factory=None, budget: int = SCENARIO_BUDGET, weak_signals=None):
    """Generate scenarios from ranked seeds, preferring distinct affected-module sets.

    Insert a matching weak-signal compound seed as the second candidate when
    available. Defer overlapping candidates and reuse them only if needed to fill the budget.
    """
    modules = wm.with_emerging(horizon_map, modules)
    G = wm.build_world_graph(horizon_map, modules)
    info = {
        "world": {"modules": G.number_of_nodes(), "edges": G.number_of_edges()},
        "seeds_considered": 0,
    }
    if G.number_of_edges() == 0:
        info["error"] = "world graph has no edges => nothing to propagate"
        return [], info

    seeds_ranked = wm.rank_vulnerability_seeds(G, limit=None)
    info["seeds_considered"] = len(seeds_ranked)

    candidates = list(seeds_ranked)
    weak_seed = _weak_signal_seed(modules, weak_signals, seeds_ranked)
    if weak_seed:
        candidates.insert(1, weak_seed)   # Keep the highest-ranked hotspot as the first candidate.

    # Simulate candidates in order, deferring those too similar to selected runs.
    selected, repeats = [], []
    for seed in candidates:
        if len(selected) >= budget:
            break
        intensity = wm.SEED_STRESS if len(seed["nodes"]) == 1 else wm.COMPOUND_SEED_STRESS
        run = wm.propagate_stress(G, {node: intensity for node in seed["nodes"]})
        affected = {n for n, v in run[0].items() if v >= wm.ACTIVATION_THRESHOLD}
        entry = (seed, intensity, run, affected)
        if any(_overlap(affected, other[3]) > MAX_SEED_OVERLAP for other in selected):
            repeats.append(entry)
        else:
            selected.append(entry)
    info["seeds_skipped_as_repeats"] = len(repeats)
    selected += repeats[:budget - len(selected)]   # Fill remaining slots with deferred candidates.

    scenarios = []
    for index, (seed, intensity, run, _) in enumerate(selected, start=1):
        stresses, activation_round, trace = run
        weak = seed.get("weak_signal")
        shock = {
            "shock_type": "compound_weak_signal" if weak else "node_shock",
            "target_module": seed["nodes"][0],
            "intensity": intensity,
            "rationale": f"Stress the {seed['motif_type']} trigger.",
        }
        if weak:
            shock["co_targets"] = seed["nodes"][1:]
            shock["weak_signal"] = weak["statement"]
        events = trace_to_events(trace)
        context = evidence_context(modules, shock, events)
        if weak:
            title = str(weak.get("signal") or "")[:250]
            source = next((entry for entry in context["evidence"]
                           if title.strip() and entry["title"].strip().casefold() == title.strip().casefold()), None)
            if source is not None:
                source["weak_implication"] = {"statement": weak["statement"], "mechanism": weak.get("mechanism") or ""}
            else:
                context["evidence"].append({
                    "id": f"s{len(context['evidence']) + 1}", "module": "weak signal",
                    "title": title,
                    "content": f"{weak['statement']} {weak.get('mechanism') or ''}".strip()[:1200],
                })
        print(f"  [Phase C] Scenario {index}/{len(selected)}: {shock['target_module']}", flush=True)
        narration = _generate_narration(seed, shock, events, stresses, context, llm, parser_factory)
        scenario = _build_scenario(index, seed, shock, narration, events, stresses, activation_round)
        scenario["financial_evidence"] = context
        scenarios.append(scenario)

    return scenarios, info


def scenario_digest(scenario: dict) -> str:
    return (f"{scenario['scenario_id']}: '{scenario['title']}' | "
            f"{len(scenario.get('events', []))} events, "
            f"{len(scenario.get('stresses', {}))} modules activated, "
            f"narration {scenario.get('narrated_by', '?')}")
