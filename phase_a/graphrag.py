import json
import random
from datetime import datetime, timezone

import torch
from langchain_core.embeddings import Embeddings
from main import get_falkor_graph
from transformers import AutoModel, AutoTokenizer

DEFAULT_VECTOR_DIM = 1536


class CustomJinaEmbeddings(Embeddings):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


_global_embedder = None


def get_embedder():
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = CustomJinaEmbeddings("jinaai/jina-embeddings-v5-text-nano")
    return _global_embedder


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_json(value) -> str:
    return json.dumps(value or [], ensure_ascii=False, sort_keys=True)


def _module_embedding_text(module: dict) -> str:
    evidence = module.get("evidence_signals") or []
    evidence_text = " ".join(
        f"{item.get('title', '')}: {item.get('content', '')}"
        for item in evidence
        if isinstance(item, dict)
    )
    steeps = module.get("steeps_intersections") or []
    steeps_text = steeps if isinstance(steeps, str) else ", ".join(str(item) for item in steeps)
    parts = [
        module.get("name", ""),
        steeps_text,
        module.get("velocity", ""),
        module.get("inertia", ""),
        module.get("impact_potential", ""),
        module.get("certainty_level", ""),
        module.get("temporal_horizon", ""),
        module.get("novelty_rationale", ""),
        module.get("reasoning", ""),
        evidence_text,
    ]
    return "\n".join(str(part) for part in parts if part)


def _update_embedding_text(update: dict, event: dict) -> str:
    parts = [
        update.get("module_name", ""),
        update.get("updated_velocity", ""),
        update.get("updated_inertia", ""),
        update.get("updated_impact_potential", ""),
        update.get("updated_certainty_level", ""),
        update.get("reasoning", ""),
        event.get("title", ""),
        event.get("content", ""),
    ]
    return "\n".join(str(part) for part in parts if part)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(x * x for x in v2) ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def _lexical_score(query: str, module: dict) -> int:
    query_tokens = {token.lower() for token in query.split() if len(token) > 3}
    module_tokens = {
        token.lower()
        for token in _module_embedding_text(module).split()
        if len(token) > 3
    }
    return len(query_tokens & module_tokens)


def _fallback_modules(event_text: str, current_modules_dict: dict, top_k: int, embedder=None) -> list:
    modules = list(current_modules_dict.values())
    if not modules:
        return []
    if embedder:
        try:
            event_vec = embedder.embed_query(event_text)
            ranked = []
            for module in modules:
                module_vec = embedder.embed_query(_module_embedding_text(module))
                ranked.append((module, _cosine_similarity(event_vec, module_vec)))
            ranked.sort(key=lambda item: item[1], reverse=True)
            return [module for module, _ in ranked[:top_k]]
        except Exception:
            pass
    modules.sort(
        key=lambda module: (
            _lexical_score(event_text, module),
            _safe_int(module.get("novelty_score")),
            len(module.get("evidence_signals") or []),
        ),
        reverse=True,
    )
    return modules[:top_k]


def _embed_or_noise(embedder, text: str, dim: int) -> list[float]:
    try:
        return embedder.embed_query(text)
    except Exception:
        return [random.uniform(-0.01, 0.01) for _ in range(dim)]


def _ensure_vector_index(kg, dim: int):
    try:
        kg.query(f"CREATE VECTOR INDEX FOR (m:Module) ON (m.vector) OPTIONS {{dimension: {dim}, similarityFunction: 'cosine'}}")
    except Exception:
        pass


def _event_params(event: dict, now: str) -> dict:
    return {
        "event_title": str(event.get("title", "Unknown")),
        "event_content": str(event.get("content", "")),
        "now": now,
    }


def graphrag_retrieval(event_text: str, current_modules_dict: dict, top_k: int = 5) -> list:
    """GraphRAG implementation using FalkorDB's Vector Index."""
    kg = get_falkor_graph()
    if not kg or not current_modules_dict:
        return _fallback_modules(event_text, current_modules_dict, top_k)

    try:
        embedder = get_embedder()
        vec = embedder.embed_query(event_text)
        dim = len(vec)
        _ensure_vector_index(kg, dim)

        safe_top_k = max(1, int(top_k))
        q = f"CALL db.idx.vector.queryNodes('Module', 'vector', {safe_top_k}, vecf32($vector)) YIELD node RETURN node.name"
        res = kg.query(q, params={"vector": vec})
        retrieved_names = [record[0] for record in res.result_set]

        if not retrieved_names:
            return _fallback_modules(event_text, current_modules_dict, top_k, embedder)

        retrieved = [current_modules_dict[name] for name in retrieved_names if name in current_modules_dict]
        wanted = min(safe_top_k, len(current_modules_dict))
        if len(retrieved) < wanted:
            # Index  missing modules held in state (out of sync): top-up from in memory rank
            seen = {m["name"] for m in retrieved}
            extra = _fallback_modules(event_text, current_modules_dict, len(current_modules_dict), embedder)
            retrieved += [m for m in extra if m["name"] not in seen][: wanted - len(retrieved)]
        return retrieved
    except Exception as e:
        print(f"  [GraphRAG Error] {e}. Falling back to ranked in-memory retrieval.")
        return _fallback_modules(event_text, current_modules_dict, top_k)


def persist_falkordb_actions(actions: dict, event: dict):
    kg = get_falkor_graph()
    if not kg:
        return

    now = _utc_now()
    embedder = get_embedder()

    try:
        test_vec = embedder.embed_query("test")
        dim = len(test_vec)
    except Exception:
        dim = DEFAULT_VECTOR_DIM
    _ensure_vector_index(kg, dim)

    for create in actions.get("creations", []):
        _persist_creation(kg, embedder, create, event, now, dim)

    for update in actions.get("updates", []):
        _persist_update(kg, embedder, update, event, now, dim)

    for merge in actions.get("merges", []) or []:
        _persist_merge(kg, embedder, merge, event, now, dim)


def _persist_creation(kg, embedder, create: dict, event: dict, now: str, dim: int):
    m_name = str(create.get("name", ""))
    params = {
        "name": m_name,
        "velocity": str(create.get("velocity", "")),
        "inertia": str(create.get("inertia", "")),
        "impact": str(create.get("impact_potential", "")),
        "certainty": str(create.get("certainty_level", "")),
        "sentiment": str(create.get("sentiment", "")),
        "temporal": str(create.get("temporal_horizon", "")),
        "novelty_score": _safe_int(create.get("novelty_score")),
        "novelty_rationale": str(create.get("novelty_rationale", "")),
        "reasoning": str(create.get("reasoning", "")),
        "steeps": _as_json(create.get("steeps_intersections")),
        "evidence": _as_json(create.get("evidence_signals")),
        "vector": _embed_or_noise(embedder, _module_embedding_text(create), dim),
        **_event_params(event, now),
    }

    try:
        q_create = """CREATE (m:Module {
            name: $name,
            velocity: $velocity,
            inertia: $inertia,
            impact_potential: $impact,
            certainty_level: $certainty,
            sentiment: $sentiment,
            temporal_horizon: $temporal,
            novelty_score: $novelty_score,
            novelty_rationale: $novelty_rationale,
            reasoning: $reasoning,
            steeps_intersections: $steeps,
            evidence_signals: $evidence,
            vector: vecf32($vector),
            created_at: $now,
            updated_at: $now
        })"""
        kg.query(q_create, params=params)
        kg.query(
            """MATCH (m:Module {name: $name})
               CREATE (e:Event {title: $event_title, content: $event_content, observed_at: $now})
               CREATE (e)-[:EVIDENCE_FOR]->(m)""",
            params=params,
        )
    except Exception as e:
        print(f"  [FalkorDB Create Error] Failed to persist creation of {m_name}: {e}")


def _persist_update(kg, embedder, update: dict, event: dict, now: str, dim: int):
    m_name = str(update.get("module_name", ""))
    reasoning = str(update.get("reasoning", ""))
    params = {
        "name": m_name,
        "velocity": str(update.get("updated_velocity", "")),
        "inertia": str(update.get("updated_inertia", "")),
        "impact": str(update.get("updated_impact_potential", "")),
        "certainty": str(update.get("updated_certainty_level", "")),
        "reasoning": reasoning,
        "vector": _embed_or_noise(embedder, _update_embedding_text(update, event), dim),
        **_event_params(event, now),
    }

    try:
        q_update = """MATCH (m:Module {name: $name})
                      SET m.velocity = $velocity,
                          m.inertia = $inertia,
                          m.impact_potential = $impact,
                          m.certainty_level = $certainty,
                          m.reasoning = coalesce(m.reasoning, '') + '\nUpdate Reasoning: ' + $reasoning,
                          m.vector = vecf32($vector),
                          m.updated_at = $now"""
        kg.query(q_update, params=params)
        kg.query(
            """MATCH (m:Module {name: $name})
               CREATE (e:Event {title: $event_title, content: $event_content, observed_at: $now})
               CREATE (e)-[:EVIDENCE_FOR]->(m)""",
            params=params,
        )
    except Exception as e:
        print(f"  [FalkorDB Update Error] Failed to persist update for {m_name}: {e}")


def _persist_merge(kg, embedder, merge: dict, event: dict, now: str, dim: int):
    ma = str(merge.get("module_a", ""))
    mb = str(merge.get("module_b", ""))
    merged_module = merge.get("merged_module") or {}
    merged_name = str(merged_module.get("name", ""))
    params = {
        "module_a": ma,
        "module_b": mb,
        "merged_name": merged_name,
        "vector": _embed_or_noise(embedder, _module_embedding_text(merged_module), dim),
        **_event_params(event, now),
    }

    try:
        kg.query(
            """MERGE (new:Module {name: $merged_name})
               SET new.vector = vecf32($vector),
                   new.created_at = coalesce(new.created_at, $now),
                   new.updated_at = $now""",
            params=params,
        )
        kg.query(
            """MATCH (a:Module {name: $module_a})<-[:EVIDENCE_FOR]-(e:Event),
                     (new:Module {name: $merged_name})
               MERGE (e)-[:EVIDENCE_FOR]->(new)""",
            params=params,
        )
        kg.query(
            """MATCH (b:Module {name: $module_b})<-[:EVIDENCE_FOR]-(e:Event),
                     (new:Module {name: $merged_name})
               MERGE (e)-[:EVIDENCE_FOR]->(new)""",
            params=params,
        )
        kg.query(
            """MATCH (new:Module {name: $merged_name})
               CREATE (ev:Event {title: $event_title, content: $event_content, observed_at: $now})
               CREATE (ev)-[:EVIDENCE_FOR]->(new)""",
            params=params,
        )

        safe_props = "type: coalesce(r.type, 'interacts'), causality: coalesce(r.causality, 'unidirectional'), empirical_trend: coalesce(r.empirical_trend, 0), narrative: coalesce(r.narrative, 0), institutional: coalesce(r.institutional, 0)"
        kg.query(f"MATCH (src)-[r:INTERACTS]->(a:Module {{name: $module_a}}), (new:Module {{name: $merged_name}}) MERGE (src)-[:INTERACTS {{{safe_props}}}]->(new)", params=params)
        kg.query(f"MATCH (src)-[r:INTERACTS]->(b:Module {{name: $module_b}}), (new:Module {{name: $merged_name}}) MERGE (src)-[:INTERACTS {{{safe_props}}}]->(new)", params=params)
        kg.query(f"MATCH (a:Module {{name: $module_a}})-[r:INTERACTS]->(tgt), (new:Module {{name: $merged_name}}) MERGE (new)-[:INTERACTS {{{safe_props}}}]->(tgt)", params=params)
        kg.query(f"MATCH (b:Module {{name: $module_b}})-[r:INTERACTS]->(tgt), (new:Module {{name: $merged_name}}) MERGE (new)-[:INTERACTS {{{safe_props}}}]->(tgt)", params=params)
        kg.query("MATCH (a:Module {name: $module_a}) DETACH DELETE a", params=params)
        kg.query("MATCH (b:Module {name: $module_b}) DETACH DELETE b", params=params)
    except Exception as e:
        print(f"  [FalkorDB Merge Error] Failed to persist merge {ma} + {mb}: {e}")
