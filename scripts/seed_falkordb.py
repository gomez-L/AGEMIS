"""Rebuild the FalkorDB graph 'main_kg' from the pre-computed main_state.json.

Used at container start in the Azure deployment so the chat queries the same graph the
tabs show. Can also be run locally:  ./.venv/bin/python scripts/seed_falkordb.py
Node and edge schema follow phase_a/graphrag.py (Module nodes, without the retrieval
vectors) and phase_b/graph_persistence.py (INTERACTS edges, EdgeMutation log).
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from main import clear_falkor_graph, get_falkor_graph  # noqa: E402
from phase_b.graph_persistence import persist_falkordb_edges  # noqa: E402


def wait_for_graph(timeout_seconds=60):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        kg = get_falkor_graph()
        if kg is not None:
            try:
                kg.query("RETURN 1")
                return kg
            except Exception:
                pass
        time.sleep(2)
    return None


def as_json(value):
    return value if isinstance(value, str) else json.dumps(value or [], ensure_ascii=False)


def main():
    state = json.loads((BASE_DIR / "main_state.json").read_text(encoding="utf-8"))
    modules = state.get("modules") or state.get("approved_modules") or []
    horizon_map = state.get("horizon_map") or {}

    kg = wait_for_graph()
    if kg is None:
        print("[seed] FalkorDB not reachable; the chat will answer without the graph.")
        return

    clear_falkor_graph()
    kg = get_falkor_graph()
    now = datetime.now(timezone.utc).isoformat()
    # Emerging modules (dashed nodes in the horizon graph) are Modules flagged provisional.
    emerging = [dict(node, provisional=True) for node in horizon_map.get("emerging_nodes") or []]
    for module in [dict(m, provisional=False) for m in modules if isinstance(m, dict)] + emerging:
        if not module.get("name"):
            continue
        kg.query(
            """CREATE (:Module {
                name: $name, velocity: $velocity, inertia: $inertia,
                impact_potential: $impact, certainty_level: $certainty,
                sentiment: $sentiment, temporal_horizon: $temporal,
                novelty_score: $novelty_score, novelty_rationale: $novelty_rationale,
                reasoning: $reasoning, steeps_intersections: $steeps,
                evidence_signals: $evidence, provisional: $provisional,
                emerging_score: $emerging_score, created_at: $now, updated_at: $now
            })""",
            params={
                "name": str(module["name"]),
                "velocity": str(module.get("velocity", "")),
                "inertia": str(module.get("inertia", "")),
                "impact": str(module.get("impact_potential", "")),
                "certainty": str(module.get("certainty_level", "")),
                "sentiment": str(module.get("sentiment", "")),
                "temporal": str(module.get("temporal_horizon", "")),
                "novelty_score": int(module.get("novelty_score") or 0),
                "novelty_rationale": str(module.get("novelty_rationale", "")),
                "reasoning": str(module.get("reasoning", "")),
                "steeps": as_json(module.get("steeps_intersections")),
                "evidence": as_json(module.get("evidence_signals")),
                "provisional": bool(module["provisional"]),
                "emerging_score": float(module.get("emerging_score") or 0),
                "now": now,
            },
        )
    persist_falkordb_edges(horizon_map)

    nodes = kg.query("MATCH (m:Module) RETURN count(m)").result_set[0][0]
    edges = kg.query("MATCH (:Module)-[r:INTERACTS]->(:Module) RETURN count(r)").result_set[0][0]
    print(f"[seed] main_kg ready: {nodes} modules ({len(emerging)} emerging), {edges} connections.")


if __name__ == "__main__":
    main()
