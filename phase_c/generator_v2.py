"""Scenario generator v2: ranks paths by cross-domain reach instead of origin fragility."""
import json
import os
import tempfile
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter

import networkx as nx

from phase_c import world_model as wm
from phase_c.financial_stability import evidence_context, validate_scenario
from phase_c.generator import _build_scenario, _format_events, _narrate_symbolic, _overlap, trace_to_events
from phase_c.prompts import get_narrator_prompt
from phase_c.schemas import ScenarioNarration

SCENARIO_BUDGET = 2
MIN_CHAIN_HOPS = 2
MAX_OVERLAP = 0.34
OUTSIDE_IN_BONUS = 1.5
FAMILIARITY_DISCOUNT = 0.5
EMERGING_BONUS = 1.5  # Boost paths starting at a provisional node.
TOP_CANDIDATES_SAVED = 10
FINANCIAL_MARKERS = ("financial stability", "banking", "capital markets", "payment", "monetary")


def _labels(module):
    labels = module.get("steeps_intersections") or []
    return [labels] if isinstance(labels, str) else [str(label) for label in labels]


def _domain(module):
    labels = _labels(module)
    return labels[0].split(":", 1)[0].strip() if labels else "Unknown"


def _is_financial(module):
    labels = _labels(module)
    return bool(labels) and any(marker in labels[0].lower() for marker in FINANCIAL_MARKERS)


def _titles(module):
    return {str(s.get("title")) for s in module.get("evidence_signals") or [] if isinstance(s, dict)}


def _profile(paths, by_name):
    steps = [(a, b) for path in paths for a, b in zip(path, path[1:])]
    get = lambda name: by_name.get(name, {})
    crossings = sum(_domain(get(a)) != _domain(get(b)) for a, b in steps)
    familiar = sum(bool(_titles(get(a)) & _titles(get(b))) for a, b in steps) / len(steps)
    outside_in = (_is_financial(get(paths[0][-1]))
                  and all(not _is_financial(get(path[0])) for path in paths))
    return {
        "paths": paths, "hops": len(steps), "domain_crossings": crossings,
        "domains": list(dict.fromkeys(_domain(get(n)) for path in paths for n in path)),
        "outside_in": outside_in, "familiar_share": round(familiar, 2),
        "emerging": [n for n in dict.fromkeys(path[0] for path in paths) if get(n).get("provisional")],
    }


def _factors(profile, end_stress):
    """Return the ranking multipliers recorded with each candidate."""
    return {"hops": profile["hops"], "domain_crossings": 1 + profile["domain_crossings"],
            "outside_in": OUTSIDE_IN_BONUS if profile["outside_in"] else 1.0,
            "emerging": EMERGING_BONUS if profile["emerging"] else 1.0,
            "familiarity": 1 - FAMILIARITY_DISCOUNT * profile["familiar_share"],
            "end_stress": end_stress ** 0.5}


def _score(profile, end_stress):
    score = 1.0
    for factor in _factors(profile, end_stress).values():
        score *= factor
    return round(score, 3)


def _candidate(G, motif, nodes, paths, by_name, intensity):
    run = wm.propagate_stress(G, {node: intensity for node in nodes})
    end_stress = run[0].get(paths[0][-1], 0.0)
    if end_stress < wm.ACTIVATION_THRESHOLD:
        return None
    profile = _profile(paths, by_name)
    return {"motif_type": motif, "nodes": list(nodes), "intensity": intensity, "run": run,
            "affected": {n for n, v in run[0].items() if v >= wm.ACTIVATION_THRESHOLD},
            "score": _score(profile, end_stress), "surprise": profile,
            "end_stress": end_stress, "factors": _factors(profile, end_stress)}


def find_candidates(G, modules):
    by_name = {m.get("name"): m for m in modules or []}
    H = nx.DiGraph((u, v) for u, v, d in G.edges(data=True) if d.get("polarity") != "buffering")
    reach = {n: nx.single_source_shortest_path(H, n) for n in H.nodes}
    candidates = []

    for seed, paths in reach.items():
        options = [c for path in paths.values() if len(path) > MIN_CHAIN_HOPS
                   for c in [_candidate(G, "cross_domain_chain", [seed], [path], by_name, wm.SEED_STRESS)] if c]
        if options:
            candidates.append(max(options, key=lambda c: c["score"]))

    for a, b in combinations(sorted(reach), 2):
        if _domain(by_name.get(a, {})) == _domain(by_name.get(b, {})) or b in reach[a] or a in reach[b]:
            continue
        options = [c for meet in set(reach[a]) & set(reach[b])
                   if set(reach[a][meet]) & set(reach[b][meet]) == {meet}
                   for c in [_candidate(G, "convergence", [a, b], [reach[a][meet], reach[b][meet]],
                                        by_name, wm.COMPOUND_SEED_STRESS)] if c]
        if options:
            candidates.append(max(options, key=lambda c: c["score"]))

    candidates.sort(key=lambda c: (-c["score"], c["nodes"]))
    return candidates


def select_candidates(candidates, budget=SCENARIO_BUDGET):
    selected = candidates[:1]
    while candidates and len(selected) < budget:
        rest = [c for c in candidates if c not in selected]
        distinct = [c for c in rest
                    if all(_overlap(c["affected"], s["affected"]) <= MAX_OVERLAP for s in selected)]
        new_kind = lambda pool: [c for c in pool if all(c["motif_type"] != s["motif_type"] for s in selected)]
        pool = new_kind(distinct) or distinct or new_kind(rest) or rest
        if not pool:
            break
        selected.append(pool[0])
    return selected


def _shock_summary(candidate, by_name):
    paths = candidate["surprise"]["paths"]
    domain = lambda name: _domain(by_name.get(name, {}))
    chains = "; ".join(" -> ".join(path) for path in paths)
    if candidate["motif_type"] == "convergence":
        a, b = candidate["nodes"]
        opening = (f"Consider two developments that are unrelated to each other occurring together: "
                   f"{a} ({domain(a)}) and {b} ({domain(b)}). Neither drives the other, but both bear on "
                   f"{paths[0][-1]} through: {chains}. The central story is this coincidence")
    else:
        path = paths[0]
        opening = (f"Consider a hypothetical adverse development associated with: {path[0]} ({domain(path[0])}). "
                   f"Follow it along this chain where the evidence allows: {chains}. The central story is how it "
                   f"reaches {path[-1]} ({domain(path[-1])}), a connection that looks remote at first sight")
    early = candidate["surprise"].get("emerging")
    if early:
        opening += (f". Note that {' and '.join(early)} is an early development backed by little evidence so far; "
                    "treat it as something that might gain momentum, and say so plainly")
    return (f"{opening}. Narrate the whole route rather than only its first step, keep every unobserved "
            "link conditional, and use novelty_element to say what a single-domain view would miss.")


def _candidate_log(candidates, selected):
    """Record the top candidates and any selected outside that cutoff."""
    chosen = {id(candidate): n for n, candidate in enumerate(selected, start=1)}
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        if rank > TOP_CANDIDATES_SAVED and id(candidate) not in chosen:
            continue
        rows.append({
            "rank": rank,
            "selected_as": chosen.get(id(candidate)),
            "motif_type": candidate["motif_type"],
            "paths": candidate["surprise"]["paths"],
            "domains": candidate["surprise"]["domains"],
            "score": candidate["score"],
            "factors": {k: round(v, 3) for k, v in candidate["factors"].items()},
            "end_stress": round(candidate["end_stress"], 4),
            "familiar_share": candidate["surprise"]["familiar_share"],
            "emerging": candidate["surprise"]["emerging"],
            "affected_modules": len(candidate["affected"]),
        })
    return rows


def _path_evidence(modules, horizon_map, shock, events, steps):
    """Include all relevant evidence, with path citations first."""
    cited = {str(title).strip().casefold()
             for edge in (horizon_map or {}).get("knowledge_graph_edges") or []
             if {(edge.get("source_module"), edge.get("target_module")),
                 (edge.get("target_module"), edge.get("source_module"))} & steps
             for title in [*(edge.get("evidence") or []), *(edge.get("shared_signals") or [])]}
    return evidence_context(modules, shock, events, cited_titles=cited)


def _narrate(candidate, seed, shock, events, context, by_name, llm, parser_factory):
    if llm is not None and parser_factory is not None:
        started = perf_counter()
        raw = None
        try:
            raw = (get_narrator_prompt() | llm | parser_factory(ScenarioNarration)).invoke({
                "shock_summary": _shock_summary(candidate, by_name),
                "events": _format_events(events, step_label="step"),
                "financial_context": json.dumps(context, ensure_ascii=False),
            })
            narration = validate_scenario(raw, events, context)
            narration["narrated_by"] = "llm"
            narration["trigger_signal"] = narration.get("trigger_signal") or seed["rationale"]
            narration["novelty_element"] = narration.get("novelty_element") or seed["rationale"]
            print(f"  [Phase C v2] Narration complete ({perf_counter() - started:.1f}s)", flush=True)
            return narration
        except Exception as exc:
            # Preserve validation failures with the symbolic fallback.
            failure = {"error": str(exc)[:2000], "stage": "validation" if raw is not None else "generation"}
            if raw is not None:
                failure["rejected_output"] = raw
            elif getattr(exc, "llm_output", None) is not None:
                failure.update(stage="parsing", rejected_output=exc.llm_output)
            reason = failure["error"].partition("\n")[0] or type(exc).__name__
            print(f"  [Phase C v2] Narration failed ({perf_counter() - started:.1f}s, "
                  f"{failure['stage']}): {reason}", flush=True)
            return {**_narrate_symbolic(seed, shock, events), "narration_failure": failure}
    return _narrate_symbolic(seed, shock, events)


def generate_scenarios_v2(horizon_map, modules, llm=None, parser_factory=None, budget=SCENARIO_BUDGET):
    modules = wm.with_emerging(horizon_map, modules)
    G = wm.build_world_graph(horizon_map, modules)
    by_name = {m.get("name"): m for m in modules or []}
    candidates = find_candidates(G, modules)
    info = {"world": {"modules": G.number_of_nodes(), "edges": G.number_of_edges()},
            "candidates": len(candidates)}
    selected = select_candidates(candidates, budget)
    info["top_candidates"] = _candidate_log(candidates, selected)
    scenarios = []
    for index, candidate in enumerate(selected, start=1):
        stresses, activation_round, trace = candidate["run"]
        surprise = candidate["surprise"]
        steps = {(a, b) for path in surprise["paths"] for a, b in zip(path, path[1:])}
        events = trace_to_events([t for t in trace if (t["source"], t["target"]) in steps] or trace)
        seed = {"motif_type": candidate["motif_type"], "nodes": candidate["nodes"],
                "edges": [list(step) for step in sorted(steps)], "score": candidate["score"],
                "rationale": f"{' / '.join(' -> '.join(p) for p in surprise['paths'])} crosses "
                             f"{' -> '.join(surprise['domains'])}."}
        shock = {"shock_type": "compound_convergence" if len(candidate["nodes"]) > 1 else "node_shock",
                 "target_module": candidate["nodes"][0], "intensity": candidate["intensity"],
                 "rationale": f"Stress the {candidate['motif_type']} trigger."}
        if len(candidate["nodes"]) > 1:
            shock["co_targets"] = candidate["nodes"][1:]
        context = _path_evidence(modules, horizon_map, shock, events, steps)
        print(f"  [Phase C v2] Scenario {index}/{len(selected)}: {shock['target_module']}", flush=True)
        narration = _narrate(candidate, seed, shock, events, context, by_name, llm, parser_factory)
        scenario = _build_scenario(index, seed, shock, narration, events, stresses, activation_round)
        scenario.update(scenario_id=f"scn_v2_{index}_{candidate['motif_type']}",
                        generator="world_model_v2_surprise", surprise=surprise,
                        financial_evidence=context)
        scenarios.append(scenario)
    return scenarios, info


def inputs_hash(state):
    value = {"modules": [m.get("name") for m in state.get("modules") or []],
             "edges": (state.get("horizon_map") or {}).get("knowledge_graph_edges") or []}
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def regenerate_saved_scenarios_v2(state_path, llm, parser_factory):
    state_path = Path(state_path)
    original = state_path.read_text(encoding="utf-8")
    state = json.loads(original)
    if not state.get("modules"):
        raise ValueError("No saved modules yet. Phase A has to be run before.")
    if state.get("stale") or not (state.get("horizon_map") or {}).get("knowledge_graph_edges"):
        raise ValueError("No current saved horizon graph. Phases A-B have to be run before.")
    scenarios, info = generate_scenarios_v2(state["horizon_map"], state["modules"], llm, parser_factory)
    if not scenarios:
        raise ValueError(f"No cross-domain path of at least {MIN_CHAIN_HOPS} steps and no convergence found "
                         "in the horizon graph. Saved state was left unchanged.")
    if state_path.read_text(encoding="utf-8") != original:
        raise ValueError("Saved inputs changed during generation. Retry using the updated graph.")
    state.update(scenarios_v2=scenarios, scenarios_v2_info={**info, "inputs_hash": inputs_hash(state)})
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state_path.parent,
                                         prefix=".scenarios-v2-", suffix=".json", delete=False) as output:
            temporary = Path(output.name)
            json.dump(state, output, indent=2, ensure_ascii=False)
            output.write("\n")
        os.replace(temporary, state_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return scenarios, info
