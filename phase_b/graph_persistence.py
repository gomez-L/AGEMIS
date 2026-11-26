import json
from datetime import datetime, timezone

from main import get_falkor_graph


def _clamp_weight(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(1, min(10, value))


def persist_falkordb_edges(horizon_map: dict):
    kg = get_falkor_graph()
    if not kg:
        return

    for edge in horizon_map.get("knowledge_graph_edges", []):
        params = {
            "src": str(edge.get("source_module", "")),
            "tgt": str(edge.get("target_module", "")),
            "interaction": str(edge.get("interaction_type", "INTERACTS")),
            "causality": str(edge.get("causal_directionality", "")),
            "emp": _clamp_weight(edge.get("empirical_trend_strength")),
            "nar": _clamp_weight(edge.get("narrative_salience")),
            "inst": _clamp_weight(edge.get("institutional_relevance")),
            "now": datetime.now(timezone.utc).isoformat(),
        }
        if not params["src"] or not params["tgt"] or params["src"] == params["tgt"]:
            continue

        q = """
        MATCH (a:Module {name: $src}), (b:Module {name: $tgt})
        MERGE (a)-[r:INTERACTS]->(b)
        SET r.type = $interaction,
            r.causality = $causality,
            r.empirical_trend = $emp,
            r.narrative = $nar,
            r.institutional = $inst,
            r.updated_at = $now
        """
        kg.query(q, params=params)

    for idx, mutation in enumerate(horizon_map.get("edge_mutation_log", [])):
        params = {
            "id": f"{mutation.get('changed_at', '')}:{idx}:{mutation.get('source_module', '')}->{mutation.get('target_module', '')}",
            "changed_at": str(mutation.get("changed_at", "")),
            "src": str(mutation.get("source_module", "")),
            "tgt": str(mutation.get("target_module", "")),
            "previous": json.dumps(mutation.get("previous"), ensure_ascii=False, sort_keys=True),
            "updated": json.dumps(mutation.get("updated"), ensure_ascii=False, sort_keys=True),
        }
        kg.query(
            """
            CREATE (:EdgeMutation {
                id: $id,
                changed_at: $changed_at,
                source_module: $src,
                target_module: $tgt,
                previous: $previous,
                updated: $updated
            })
            """,
            params=params,
        )
