import json
import os
import difflib
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import networkx as nx
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import main as agemis
from main import FALKOR_ENABLED
from phase_a.graphrag import get_embedder
from phase_b.emerging import MAX_PROVISIONAL_EMPIRICAL, emerging_nodes
from phase_b.graph_persistence import persist_falkordb_edges
from phase_b.visualizer import visualize_graph_pyvis
from phase_c.world_model import classify_polarity

# Calibrate similarity cutoffs from each run's observed distribution;
# mean-pooled embeddings are anisotropic, making fixed cutoffs unreliable.
SEMANTIC_NEIGHBOR_PERCENTILE = 0.90
CANDIDATE_EDGE_PERCENTILE = 0.90
MIN_SIMILARITY_FLOOR = 0.05
MIN_MODULES_FOR_CENTERING = 5
MAX_SEMANTIC_NEIGHBORS = 3
MAX_CANDIDATE_EDGES = 15
MAX_MODULES_BEFORE_TRUNCATION = 15
FUZZY_NAME_CUTOFF = 0.75
MAX_UNEVIDENCED_EMPIRICAL = 6   # Cap empirical strength when neither citations nor shared signals support an edge.
MAX_WEAK_SIGNALS_IN_PROMPT = 20
WEAK_SIGNAL_MATCH_CUTOFF = 0.6
MAX_BRIDGE_EDGES = 2   # Links one bridge call may propose for its focus component.
DEFAULT_EDGE_TYPE = "INTERACTS"
DEFAULT_CAUSALITY = "Unidirectional Source->Target"

SCALABLE_MAPPER_TEMPLATE = """Your task is to construct/update the horizon knowledge graph (cross-domain causal linkages) for the ACTIVE NEIGHBORHOOD of systemic modules.

CRITICAL ARCHITECTURAL INSTRUCTIONS:
1. ACTIVE NEIGHBORHOOD (QUALITY OVER DENSITY):
   - You are focusing on the "Active Neighborhood" of modules that have recently changed or are closely related (topologically or semantically).
   - Add an edge only where you can state, in one sentence, a concrete transmission mechanism from source to target.
     The modules' transmission_channels and implications (where present) are the best raw material for this.
   - A module with no defensible link stays unconnected. Isolated modules are a valid result; never add edges to make the graph look complete.
   - Modules marked "provisional": true are early developments with little evidence so far. Consider where each could act on,
     or be acted on by, an established module, and link it only through a concrete mechanism. Such links are plausible at most:
     keep their Empirical Trend Strength at 5 or below.

2. ALGORITHMIC CANDIDATE EDGES (GUIDELINES):
   - We have pre-computed which module pairs share source signals or have similar descriptions.
   - Evaluate the following candidate pairs for a causal connection:
{candidate_edges}
   - These are only candidates. Similar modules often describe overlapping themes rather than cause and effect, and causal links often join dissimilar modules, so also consider pairs not listed. Do not create an edge unless the relationship expresses a plausible causal or systemic interaction.

3. TOPOLOGICAL CENTRALITY CONTEXT (NETWORKX ANALYTICS):
   - NetworkX PageRank and degree calculations have identified the following structural properties of our stable graph:
{centrality_analysis}
   - Use this context to identify systemic vulnerability hotspots and potential shock amplification vectors. Do not create edges that contradict these structural hubs.

4. PRE-EXISTING RISK TOPOLOGY (DO NOT DUPLICATE OR CONTRADICT):
   - The following stable connections already exist in the background graph:
{existing_edges}

5. EDGE FIELDS, DIRECTIONALITY & DYNAMIC WEIGHTS:
   - interaction_type: a short label of 2-5 words (e.g. "raises funding costs"), not a sentence.
   - polarity: "amplifying" if stress in the source increases stress in the target, "buffering" if it reduces it.
   - mechanism: one sentence naming the channel from source to target.
   - evidence: titles of evidence signals, copied exactly from the two connected modules, that support the link. Leave empty if the link is inferred.
   - Specify the causal directionality (Unidirectional Source->Target or Bidirectional feedback loop). Use bidirectional only when you can name the mechanism in both directions.
   - Tag each interaction with three weights (1-10), scored independently of each other:
     - Empirical Trend Strength (momentum).
     - Narrative Salience (public attention).
     - Institutional Relevance (supervisory/regulatory impact).
   - Weight scale, use the full range:
     - 1-3: inferred or speculative, no direct support in the modules.
     - 4-6: plausible and indirectly supported.
     - 7-8: directly supported by at least one listed evidence signal.
     - 9-10: transmission is observed and ongoing right now. Rare.
     An Empirical Trend Strength above 6 is only accepted for edges that list at least one valid evidence title.
   - Prioritize edges that explain the current unstable configuration: transmission, amplification, mitigation, vulnerability accumulation, or feedback.
   - Avoid generic background associations that do not change the interpretation of the present risk topology.

6. PERTURBATION SENSITIVITY & WEAK SIGNALS:
   - Identify specific nodes or edges highly sensitive to small shocks.
   - inflexion_weak_signals: from the OBSERVED WEAK SIGNALS below, pick those that could tip a module or edge and copy each statement exactly.
     Do not invent weak signals. If none are listed, derive them only from the module evidence.
   - Never invent figures, thresholds, institutions or amounts that are not in the supplied modules.
   OBSERVED WEAK SIGNALS (implications not yet part of a module, most novel and high-impact first):
{weak_signals}

7. CENTRAL VULNERABILITY:
   - Formulate a unified, mathematically-grounded central vulnerability summarizing the unstable present configuration.

8. EXPERT GUIDANCE (HITL):
   - A human expert reviewed the current graph. Their instructions take precedence over algorithmic candidate signals when they conflict:
{expert_guidance}

Active Neighborhood Modules to Map:
{modules}

{format_instructions}"""

# For each group disconnected from the largest group, ask the LLM whether stress could
# travel between it & the rest of the graph in either direction, including across domains.
# Check the smallest groups first. A group can be a single module, a few connected modules or an emerging node.
BRIDGE_TEMPLATE = """Review the causal links in a financial stability horizon graph.
The FOCUS modules below remain disconnected after the first mapping pass. Assess whether any links were missed.

For each FOCUS module, check both directions: could stress reach an OTHER module from it, or reach it from an OTHER module?
- Propose a link only if you can describe a concrete transmission mechanism in one sentence. Pay particular attention
  to links the first pass may have missed across domains: environmental, technological, social, political or security
  developments affecting banks, markets, payments, public finances or monetary policy, and the reverse.
- A link must transmit stress or relief from source to target. A topic relationship alone ("regulation X covers Y") is insufficient.
- Each link must join one FOCUS module and one OTHER module. Copy both names exactly. Return at most {max_edges} links,
  strongest first.
- Return an empty list if no link is justified. Connectivity alone is not a reason to add a link.
- polarity: "amplifying" if stress in the source increases stress in the target, "buffering" if it reduces it.
- Score each weight independently on a 1-10 scale: 1-3 speculative, 4-6 plausible and indirectly supported,
  7-8 directly supported by a listed evidence signal. For evidence, copy titles exactly from the two modules;
  leave the list empty if the link is inferred.
- Do not invent figures, thresholds, institutions or amounts absent from the supplied modules.

FOCUS MODULE(S):
{focus}

OTHER MODULES:
{others}

{format_instructions}"""


class BridgeEdges(BaseModel):
    edges: list[agemis.GraphEdge] = Field(default_factory=list, description="Links between a FOCUS and an OTHER module; empty if none is defensible")


def _bridge_calls():
    # Read at run time. Zero disables the bridge pass.
    return max(0, int(os.getenv("PHASE_B_BRIDGE_CALLS", "8")))


def _module_brief(module):
    """Compact module description for the OTHER list of a bridge call."""
    brief = {"name": module.get("name"), "domains": module.get("steeps_intersections") or []}
    # Drivers and effects say what a module reacts to and what it causes; channels are the fallback.
    for key in ("drivers", "effects") if module.get("drivers") and module.get("effects") else ("transmission_channels",):
        if module.get(key):
            brief[key] = module[key][:3]
    brief["evidence_signals"] = sorted(_evidence_titles(module))
    if module.get("provisional"):
        brief["provisional"] = True
    return brief


def _next_unlinked_component(names, edges, asked):
    """Smallest component outside the largest one that has not been put to the bridge call yet."""
    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from((e["source_module"], e["target_module"]) for e in edges
                         if e.get("source_module") in graph and e.get("target_module") in graph)
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c))
    waiting = [c for c in components[1:] if not set(c) & asked]
    return min(waiting, key=lambda c: (len(c), c)) if waiting else None


def _percentile(sorted_values, fraction):
    """Return the value at the fractional index of a sorted list, or zero if empty."""
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]

def _centroid(vectors):
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(dim)]

def _unit(vector):
    norm = sum(x * x for x in vector) ** 0.5
    return [x / norm for x in vector] if norm else list(vector)

def _similarity_profile(sims, centered):
    values = sorted(sims.values())
    if not values:
        return {
            "pairs": 0,
            "centered": centered,
            "semantic_neighbor_threshold": MIN_SIMILARITY_FLOOR,
        }
    neighbor = max(MIN_SIMILARITY_FLOOR, _percentile(values, SEMANTIC_NEIGHBOR_PERCENTILE))
    return {
        "pairs": len(values),
        "centered": centered,
        "min": round(values[0], 4),
        "p50": round(_percentile(values, 0.50), 4),
        "p90": round(_percentile(values, 0.90), 4),
        "max": round(values[-1], 4),
        "semantic_neighbor_threshold": round(neighbor, 4),
    }

def _candidate_threshold(selectable_sims):
    """Derive the candidate cutoff from similarities of selectable module pairs."""
    return max(MIN_SIMILARITY_FLOOR, _percentile(sorted(selectable_sims), CANDIDATE_EDGE_PERCENTILE))

def _calibrated_similarities(module_embs):
    """Return pair similarities keyed by ordered names (a < b) and a calibration profile."""
    dims = Counter(len(vec) for vec in module_embs.values() if vec)
    if not dims:
        return {}, _similarity_profile({}, False)

    # Keep vectors of the most common dimension so the centroid has a consistent shape.
    dim = dims.most_common(1)[0][0]
    names = sorted(name for name, vec in module_embs.items() if len(vec or []) == dim)

    vectors = {name: module_embs[name] for name in names}
    centered = len(names) >= MIN_MODULES_FOR_CENTERING
    if centered:
        centroid = _centroid([vectors[name] for name in names])
        vectors = {name: [x - c for x, c in zip(vec, centroid)] for name, vec in vectors.items()}
    # Unit-normalize once so each pairwise similarity is a dot product.
    vectors = {name: _unit(vec) for name, vec in vectors.items()}

    sims = {
        (a, b): sum(x * y for x, y in zip(vectors[a], vectors[b]))
        for a, b in combinations(names, 2)
    }
    return sims, _similarity_profile(sims, centered)

def _directed_candidates(modules, embedder, exclude, limit=MAX_CANDIDATE_EDGES):
    """Rank (source, target, score) pairs by effect-to-driver similarity.

    Requires drivers and effects from Phase A v2. Uses the same centering rule
    and percentile cutoff as the similarity candidates.
    """
    items = [(m["name"], "; ".join(m["effects"]), "; ".join(m["drivers"]))
             for m in modules if m.get("name") and m.get("drivers") and m.get("effects")]
    if len(items) < 2:
        return []
    try:
        texts = [effect for _, effect, _ in items] + [driver for _, _, driver in items]
        vectors = np.asarray(embedder.embed_documents(texts), dtype=float)
    except Exception as exc:
        print(f"  [Directed Candidates] Skipped: embedding failed ({str(exc).splitlines()[0]})")
        return []
    if len(items) >= MIN_MODULES_FOR_CENTERING:
        vectors = vectors - vectors.mean(axis=0)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    # Rows are source effects; columns are target drivers.
    scores = vectors[:len(items)] @ vectors[len(items):].T
    pairs = [(items[a][0], items[b][0], float(scores[a, b]))
             for a in range(len(items)) for b in range(len(items))
             if a != b and (items[a][0], items[b][0]) not in exclude]
    threshold = _candidate_threshold([score for _, _, score in pairs])
    return sorted((p for p in pairs if p[2] >= threshold), key=lambda p: (-p[2], p[0], p[1]))[:limit]


def _lookup_sim(sims, a, b):
    return sims.get((a, b) if a < b else (b, a), 0.0)

def _utc_now():
    return datetime.now(timezone.utc).isoformat()

def _clamp_weight(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(1, min(10, value))

def _module_text(module):
    evidence = module.get("evidence_signals") or []
    evidence_text = " ".join(
        f"{item.get('title', '')}: {item.get('content', '')}"
        for item in evidence
        if isinstance(item, dict)
    )
    steeps = module.get("steeps_intersections") or []
    steeps_text = steeps if isinstance(steeps, str) else ", ".join(str(item) for item in steeps)
    return "\n".join(
        str(part)
        for part in [
            module.get("name", ""),
            steeps_text,
            module.get("velocity", ""),
            module.get("inertia", ""),
            module.get("impact_potential", ""),
            module.get("certainty_level", ""),
            module.get("novelty_rationale", ""),
            module.get("reasoning", ""),
            evidence_text,
        ]
        if part
    )

def _module_embedding(module, embedder):
    vector = module.get("vector")
    if isinstance(vector, list) and vector:
        return vector
    try:
        return embedder.embed_query(_module_text(module))
    except Exception:
        return [0.0] * 1536

def _resolve_module_name(name, approved_names, corrections):
    if not name:
        return None
    if name in approved_names:
        return name

    lower_index = {approved.lower(): approved for approved in approved_names}
    lowered = str(name).lower()
    if lowered in lower_index:
        resolved = lower_index[lowered]
        corrections.append({"input": name, "resolved": resolved, "method": "case_insensitive"})
        return resolved

    best = difflib.get_close_matches(str(name), approved_names, n=1, cutoff=FUZZY_NAME_CUTOFF)
    if best:
        corrections.append({"input": name, "resolved": best[0], "method": "fuzzy"})
        return best[0]
    corrections.append({"input": name, "resolved": None, "method": "unresolved"})
    return None

def _evidence_titles(module):
    return {
        str(item.get("title")).strip()
        for item in module.get("evidence_signals") or []
        if isinstance(item, dict) and item.get("title")
    }

def _normalize_edge(edge, approved_names, corrections, titles_by_module=None):
    src = _resolve_module_name(edge.get("source_module"), approved_names, corrections)
    tgt = _resolve_module_name(edge.get("target_module"), approved_names, corrections)
    if not src or not tgt or src == tgt:
        return None
    titles_by_module = titles_by_module or {}
    normalized = dict(edge)
    normalized["source_module"] = src
    normalized["target_module"] = tgt
    normalized["interaction_type"] = str(edge.get("interaction_type") or DEFAULT_EDGE_TYPE)
    polarity = str(edge.get("polarity") or "").strip().lower()
    # Infer polarity from the interaction label when the supplied value is invalid.
    normalized["polarity"] = polarity if polarity in ("amplifying", "buffering") else classify_polarity(normalized["interaction_type"])
    normalized["mechanism"] = str(edge.get("mechanism") or "").strip()
    # Match citations to either endpoint's evidence titles, ignoring case and outer spaces.
    known = {t.casefold(): t for t in titles_by_module.get(src, set()) | titles_by_module.get(tgt, set())}
    cited = edge.get("evidence") if isinstance(edge.get("evidence"), list) else []
    normalized["evidence"] = list(dict.fromkeys(
        known[str(t).strip().casefold()] for t in cited if str(t).strip().casefold() in known
    ))
    normalized["shared_signals"] = sorted(titles_by_module.get(src, set()) & titles_by_module.get(tgt, set()))
    normalized["causal_directionality"] = str(edge.get("causal_directionality") or DEFAULT_CAUSALITY)
    normalized["empirical_trend_strength"] = _clamp_weight(edge.get("empirical_trend_strength"))
    if not normalized["evidence"] and not normalized["shared_signals"]:
        normalized["empirical_trend_strength"] = min(normalized["empirical_trend_strength"], MAX_UNEVIDENCED_EMPIRICAL)
    normalized["narrative_salience"] = _clamp_weight(edge.get("narrative_salience"))
    normalized["institutional_relevance"] = _clamp_weight(edge.get("institutional_relevance"))
    normalized["normalized_at"] = _utc_now()
    return normalized

def _edge_changed(old_edge, new_edge):
    fields = [
        "interaction_type",
        "causal_directionality",
        "empirical_trend_strength",
        "narrative_salience",
        "institutional_relevance",
    ]
    return any(old_edge.get(field) != new_edge.get(field) for field in fields)

def _edge_history(prev_edges, new_edges):
    now = _utc_now()
    previous_by_key = {
        (edge.get("source_module"), edge.get("target_module")): edge
        for edge in prev_edges
    }
    history = []
    for edge in new_edges:
        key = (edge.get("source_module"), edge.get("target_module"))
        previous = previous_by_key.get(key)
        if previous and _edge_changed(previous, edge):
            history.append({
                "changed_at": now,
                "source_module": key[0],
                "target_module": key[1],
                "previous": previous,
                "updated": edge,
            })
        elif not previous:
            history.append({
                "changed_at": now,
                "source_module": key[0],
                "target_module": key[1],
                "previous": None,
                "updated": edge,
            })
    return history

def _clean_module_for_prompt(module):
    cleaned = dict(module)
    cleaned.pop("vector", None)
    if "evidence_signals" in cleaned and isinstance(cleaned["evidence_signals"], list):
        cleaned["evidence_signals"] = [
            sig.get("title") for sig in cleaned["evidence_signals"] if isinstance(sig, dict) and sig.get("title")
        ]
    if "reasoning" in cleaned and isinstance(cleaned["reasoning"], str) and len(cleaned["reasoning"]) > 500:
        cleaned["reasoning"] = cleaned["reasoning"][:500] + "..."
    return cleaned

def map_node(state: agemis.AgentState):
    print("\n" + "="*50)
    print("--- [Phase B] MAP HORIZON NODE ---")
    
    approved_modules = state.get("approved_modules", [])
    # Store provisional nodes in the horizon map alongside the Phase A modules.
    emerging = []
    if state.get("include_emerging"):
        emerging = emerging_nodes(
            state.get("weak_signals"), state.get("proto_modules"),
            [m.get("name") for m in approved_modules],
            agemis.llm, lambda model: agemis.RobustJsonOutputParser(pydantic_object=model))
        print(f"  [Emerging] {len(emerging)} provisional node(s): {', '.join(n['name'] for n in emerging) or 'none available'}")
        approved_modules = approved_modules + emerging
    provisional_names = {n["name"] for n in emerging}
    approved_names = [m.get("name") for m in approved_modules if m.get("name")]
    
    # 1. Load Pre-existing Graph Edges
    prev_edges = []
    if state.get("horizon_map") and "knowledge_graph_edges" in state["horizon_map"]:
        prev_edges = state["horizon_map"]["knowledge_graph_edges"]
        # Remove links to provisional nodes omitted from this build.
        gone = {n.get("name") for n in state["horizon_map"].get("emerging_nodes") or []} - set(approved_names)
        prev_edges = [e for e in prev_edges
                      if e.get("source_module") not in gone and e.get("target_module") not in gone]
        
    # 2. Build NetworkX Graph and Calculate Centralities
    G = nx.DiGraph()
    G.add_nodes_from(approved_names)
    for edge in prev_edges:
        src = edge.get("source_module")
        tgt = edge.get("target_module")
        if src in G and tgt in G:
            G.add_edge(src, tgt)
            
    # Calculate PageRank
    if len(G.edges) > 0:
        try:
            pr = nx.pagerank(G, alpha=0.85)
        except Exception:
            pr = {node: 1.0 / len(G) if len(G) > 0 else 0 for node in G}
    else:
        pr = {node: 1.0 / len(G) if len(G) > 0 else 0 for node in G}
        
    in_deg = nx.in_degree_centrality(G)
    out_deg = nx.out_degree_centrality(G)
    
    # Format Centrality Text
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    sorted_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    
    centrality_text = "   - Top Systemic Anchors (PageRank):\n"
    for name, val in sorted_pr:
        centrality_text += f"     * {name} (score: {val:.3f})\n"
    centrality_text += "   - Top Risk Drivers (Out-Degree Centrality):\n"
    for name, val in sorted_out:
        centrality_text += f"     * {name} (score: {val:.3f})\n"
    centrality_text += "   - Top Systemic Vulnerabilities (In-Degree Centrality):\n"
    for name, val in sorted_in:
        centrality_text += f"     * {name} (score: {val:.3f})\n"
        
    # 3. Determine Active Neighborhood (Topological & Semantic)
    active_names = state.get("active_modules") or []
    if not active_names:
        print("  [Delta Mapping] No active modules flagged. Defaulting to full module set.")
        active_names = approved_names
    # Include provisional nodes in delta mapping.
    active_names = list(dict.fromkeys([*active_names, *provisional_names]))
        
    active_neighborhood = set(active_names)
    
    # Add topological 1-step neighbors
    for u in active_names:
        if G.has_node(u):
            for v in G.predecessors(u):
                active_neighborhood.add(v)
            for v in G.successors(u):
                active_neighborhood.add(v)
                
    # Add semantic neighbors via calibrated embeddings
    print("  [Semantic Analysis] Computing calibrated similarities...")
    module_embs = {}
    embedder = get_embedder()
    for m in approved_modules:
        name = m.get("name")
        if name:
            module_embs[name] = _module_embedding(m, embedder)

    sims, sim_profile = _calibrated_similarities(module_embs)
    neighbor_threshold = sim_profile["semantic_neighbor_threshold"]
    if sim_profile["pairs"]:
        print(
            f"  [Similarity Calibration] {sim_profile['pairs']} pairs | centered={sim_profile['centered']} | "
            f"min {sim_profile['min']:.3f} / p50 {sim_profile['p50']:.3f} / "
            f"p90 {sim_profile['p90']:.3f} / max {sim_profile['max']:.3f}"
        )
        print(
            f"  [Similarity Calibration] neighbour cutoff p{int(SEMANTIC_NEIGHBOR_PERCENTILE * 100)}"
            f" = {neighbor_threshold:.3f}"
        )

    # Pull top-3 semantic neighbors for each active module
    for u in active_names:
        if u not in module_embs: continue
        ranked = sorted(
            ((v, _lookup_sim(sims, u, v)) for v in approved_names if v != u),
            key=lambda item: item[1],
            reverse=True,
        )[:MAX_SEMANTIC_NEIGHBORS]
        for v, sim in ranked:
            if sim >= neighbor_threshold:
                active_neighborhood.add(v)

    # Clamp active neighborhood to approved modules only
    active_neighborhood = {name for name in active_neighborhood if name in approved_names}
    print(f"  [Active Neighborhood] Selected {len(active_neighborhood)} modules out of {len(approved_names)} total approved modules.")
    
    # 4. Generate Algorithmic Candidate Edges
    # Similarity is symmetric, so each unordered pair is offered once
    selectable = [
        (u, v, _lookup_sim(sims, u, v))
        for u, v in combinations(sorted(active_neighborhood), 2)
        if not (G.has_edge(u, v) or G.has_edge(v, u))
    ]
    candidate_threshold = _candidate_threshold([sim for _, _, sim in selectable])
    candidates = sorted(
        (item for item in selectable if item[2] >= candidate_threshold),
        key=lambda item: item[2],
        reverse=True,
    )[:MAX_CANDIDATE_EDGES]
    sim_profile["candidate_edge_threshold"] = round(candidate_threshold, 4)
    sim_profile["candidate_pairs_offered"] = len(candidates)
    print(
        f"  [Candidate Edges] cutoff p{int(CANDIDATE_EDGE_PERCENTILE * 100)} = {candidate_threshold:.3f}"
        f" over {len(selectable)} selectable pairs -> {len(candidates)} offered"
    )

    # Offer pairs with shared evidence titles as additional candidates for causal review.
    titles_by_module = {m.get("name"): _evidence_titles(m) for m in approved_modules if m.get("name")}
    shared_pairs = sorted(
        ((u, v, len(titles_by_module.get(u, set()) & titles_by_module.get(v, set())))
         for u, v in combinations(sorted(active_neighborhood), 2)),
        key=lambda item: -item[2],
    )
    shared_pairs = [item for item in shared_pairs if item[2] > 0][:MAX_CANDIDATE_EDGES]

    candidate_text = ""
    for src, tgt, count in shared_pairs:
        candidate_text += f"   - '{src}' ──(shares {count} source signal(s))── '{tgt}'\n"
    if candidates:
        for src, tgt, sim in candidates:
            candidate_text += f"   - '{src}' ──(similarity: {sim:.2f})── '{tgt}'\n"
    elif not shared_pairs:
        candidate_text += f"   - [None identified above the calibrated cutoff of {candidate_threshold:.2f}]\n"

    # Add effect-to-driver candidates for modules that carry causal terms.
    directed = _directed_candidates(
        [m for m in approved_modules if m.get("name") in active_neighborhood], embedder, set(G.edges))
    for src, tgt, score in directed:
        candidate_text += (f"   - '{src}' ──(its effects resemble the drivers of the target; "
                           f"possible direction source -> target, score {score:.2f})──> '{tgt}'\n")
    if directed:
        print(f"  [Directed Candidates] {len(directed)} effect -> driver pairs offered")
    sim_profile["directed_pairs_offered"] = len(directed)

    # 4.5. Delta Merging Prep (Identify Kept Background Edges)
    kept_edges = []
    active_neighborhood_set = set(active_neighborhood)
    for edge in prev_edges:
        src = edge.get("source_module")
        tgt = edge.get("target_module")
        # Keep pre-existing edges that are stable outside the active re-evaluation zone
        if src not in active_neighborhood_set or tgt not in active_neighborhood_set:
            kept_edges.append(edge)

    # Format Pre-existing Edges Context (Only send kept background edges to avoid prompt conflict)
    existing_edges_text = ""
    if kept_edges:
        for edge in kept_edges:
            existing_edges_text += f"   - {edge.get('source_module')} -> {edge.get('target_module')} ({edge.get('interaction_type', 'INTERACTS')})\n"
    else:
        existing_edges_text += "   - [None (First run / Wiped database or active neighborhood covers all edges)]\n"
        
    # Filter modules list passed to prompt for size scaling
    modules_to_send = approved_modules
    if len(approved_modules) > MAX_MODULES_BEFORE_TRUNCATION:
        print("  [Scale Control] Truncating module descriptions sent to prompt to active neighborhood only.")
        modules_to_send = [m for m in approved_modules if m.get("name") in active_neighborhood]
        
    # Clean modules to remove vector and raw evidence text to avoid API token limits (especially for Gemma)
    modules_to_send = [_clean_module_for_prompt(m) for m in modules_to_send]
        
    # 5. Call LLM for Delta Graph Generation
    scalable_prompt = PromptTemplate(
        template=SCALABLE_MAPPER_TEMPLATE,
        input_variables=["modules", "candidate_edges", "centrality_analysis", "existing_edges", "expert_guidance", "weak_signals"],
        partial_variables={"format_instructions": agemis.RobustJsonOutputParser(pydantic_object=agemis.HorizonMap).get_format_instructions()},
    )
    
    chain = scalable_prompt | agemis.llm | agemis.RobustJsonOutputParser(pydantic_object=agemis.HorizonMap)
    
    # Deduplicate Phase A weak-signal statements and keep the highest-ranked emerging issues;
    # older states have no rank (order is then preserved) or no weak signals at all.
    ranked_weak = sorted(
        (w for w in state.get("weak_signals") or [] if isinstance(w, dict) and w.get("statement")),
        key=lambda w: -(w.get("emerging_score") or 0),
    )
    weak_statements = list(dict.fromkeys(
        str(w.get("statement")).strip() for w in ranked_weak
    ))[:MAX_WEAK_SIGNALS_IN_PROMPT]
    weak_text = "".join(f"   - {s}\n" for s in weak_statements) or "   - [None observed]\n"

    print("  [LLM] Invoking Mapper LLM for active sub-graph update...")
    response_map = chain.invoke({
        "weak_signals": weak_text,
        "modules": json.dumps(modules_to_send, indent=2),
        "candidate_edges": candidate_text,
        "centrality_analysis": centrality_text,
        "existing_edges": existing_edges_text,
        "expert_guidance": state.get("expert_guidance") or "   - [None provided]\n"
    })
            
    # Normalize newly generated edges
    normalized_new_edges = []
    name_corrections = []
    # Track candidate origins, rejected edges and applied weight caps.
    offered = {
        "shared_signal": {frozenset((u, v)) for u, v, _ in shared_pairs},
        "similarity": {frozenset((u, v)) for u, v, _ in candidates},
        "directed": {(u, v) for u, v, _ in directed},
        "directed_reversed": {(v, u) for u, v, _ in directed},
    }
    accounting = {"offered": {"shared_signal": len(shared_pairs), "similarity": len(candidates),
                              "directed": len(directed)},
                  "returned": 0, "accepted": [], "dropped": []}
    def accept(edge, origins=None, focus=None):
        """Normalize one returned edge, apply the caps and record what happened to it."""
        accounting["returned"] += 1
        seen_corrections = len(name_corrections)
        normalized = _normalize_edge(edge, approved_names, name_corrections, titles_by_module)
        reason = None
        if not normalized:
            unresolved = any(c["resolved"] is None for c in name_corrections[seen_corrections:])
            reason = "unresolved_module" if unresolved else "self_loop_or_missing_endpoint"
        elif focus is not None and len({normalized["source_module"], normalized["target_module"]} & focus) != 1:
            reason = "bridge_not_between_components"
        if reason:
            accounting["dropped"].append({
                "source_module": edge.get("source_module"), "target_module": edge.get("target_module"),
                "reason": reason})
            return None
        key = (normalized["source_module"], normalized["target_module"])
        requested = _clamp_weight(edge.get("empirical_trend_strength"))
        caps = []
        if normalized["empirical_trend_strength"] < requested:
            caps.append({"reason": "no_evidence", "from": requested, "to": normalized["empirical_trend_strength"]})
        if set(key) & provisional_names:
            normalized["provisional"] = True
            before = normalized["empirical_trend_strength"]
            normalized["empirical_trend_strength"] = min(before, MAX_PROVISIONAL_EMPIRICAL)
            if normalized["empirical_trend_strength"] < before:
                caps.append({"reason": "provisional_node", "from": before, "to": normalized["empirical_trend_strength"]})
        cited = edge.get("evidence") if isinstance(edge.get("evidence"), list) else []
        accounting["accepted"].append({
            "source_module": key[0], "target_module": key[1],
            # A pair can have several origins or be proposed only by the model alone
            "origins": origins or [name for name, pairs in offered.items()
                                   if (frozenset(key) if name in ("shared_signal", "similarity") else key) in pairs] or ["unprompted"],
            "empirical_caps": caps,
            "citations_dropped": len(cited) - len(normalized["evidence"]),
        })
        normalized_new_edges.append(normalized)
        return normalized

    for edge in response_map.get("knowledge_graph_edges", []):
        accept(edge)

    # Bridge pass (Phase A v2 modules only): one call per component the first pass left unlinked,
    # smallest first. Each asks for links between that component and the rest of the graph, so
    # isolated modules, small islands and provisional nodes get a second, focused look.
    accounting["bridge"] = []
    calls = _bridge_calls() if any(m.get("implications") for m in approved_modules) else 0
    if calls:
        bridge_parser = agemis.RobustJsonOutputParser(pydantic_object=BridgeEdges)
        bridge_chain = PromptTemplate(
            template=BRIDGE_TEMPLATE, input_variables=["focus", "others"],
            partial_variables={"max_edges": str(MAX_BRIDGE_EDGES),
                               "format_instructions": bridge_parser.get_format_instructions()},
        ) | agemis.llm | bridge_parser
        by_name = {m.get("name"): m for m in approved_modules if m.get("name")}
        asked = set()
        while len(accounting["bridge"]) < calls:
            focus = _next_unlinked_component(approved_names, kept_edges + normalized_new_edges, asked)
            if focus is None:
                break
            asked.update(focus)
            record = {"focus": focus, "returned": 0, "accepted": []}
            accounting["bridge"].append(record)
            print(f"  [Bridge {len(accounting['bridge'])}] {' + '.join(focus)}", flush=True)
            try:
                reply = bridge_chain.invoke({
                    "focus": json.dumps([_clean_module_for_prompt(by_name[n]) for n in focus], indent=2),
                    "others": json.dumps([_module_brief(m) for n, m in by_name.items() if n not in focus], indent=2),
                })
                proposed = reply.get("edges") if isinstance(reply, dict) else reply
                proposed = [e for e in proposed or [] if isinstance(e, dict)][:MAX_BRIDGE_EDGES]
            except Exception as exc:
                record["error"] = str(exc)[:500]
                print(f"    bridge call failed ({str(exc).splitlines()[0]})")
                continue
            record["returned"] = len(proposed)
            for edge in proposed:
                bridged = accept(edge, ["bridge"], set(focus))
                if bridged:
                    record["accepted"].append([bridged["source_module"], bridged["target_module"]])
            print(f"    {len(record['accepted'])} link(s) added" if record["accepted"] else "    stays unlinked")

    # Map model selections back to supplied weak signals using fuzzy matching;
    # discard unmatched selections and duplicates.
    inflexion_signals = response_map.get("inflexion_weak_signals", [])
    if weak_statements:
        picked = []
        for item in inflexion_signals or []:
            match = difflib.get_close_matches(str(item), weak_statements, n=1, cutoff=WEAK_SIGNAL_MATCH_CUTOFF)
            if match and match[0] not in picked:
                picked.append(match[0])
        inflexion_signals = picked

    # Combine kept edges with new normalized edges
    combined_edges = kept_edges + normalized_new_edges
    mutation_log = _edge_history(prev_edges, normalized_new_edges)
    
    # Deduplicate edges
    unique_edges = []
    seen = set()
    # New-edge accounting follows normalized_new_edges. Background edges come first.
    surviving = []
    for position, edge in enumerate(combined_edges):
        key = (edge.get("source_module"), edge.get("target_module"))
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)
            if position >= len(kept_edges):
                surviving.append(accounting["accepted"][position - len(kept_edges)])
        else:
            accounting["dropped"].append({"source_module": key[0], "target_module": key[1], "reason": "duplicate"})
    accounting["accepted"] = surviving
    accounting["kept_background_edges"] = len(kept_edges)
    accounting["origin_counts"] = dict(Counter(o for a in accounting["accepted"] for o in a["origins"]))
            
    # Construct final merged horizon map
    horizon_map = {
        "knowledge_graph_edges": unique_edges,
        "emerging_nodes": emerging,
        "inflexion_weak_signals": inflexion_signals,
        "perturbation_sensitivities": response_map.get("perturbation_sensitivities", []),
        "central_vulnerability": response_map.get("central_vulnerability", "N/A"),
        "edge_mutation_log": mutation_log,
        "edge_accounting": accounting,
        "name_resolution_log": name_corrections,
        "phase_b_config": {
            "semantic_neighbor_percentile": SEMANTIC_NEIGHBOR_PERCENTILE,
            "candidate_edge_percentile": CANDIDATE_EDGE_PERCENTILE,
            "min_similarity_floor": MIN_SIMILARITY_FLOOR,
            "max_semantic_neighbors": MAX_SEMANTIC_NEIGHBORS,
            "max_candidate_edges": MAX_CANDIDATE_EDGES,
            "fuzzy_name_cutoff": FUZZY_NAME_CUTOFF,
            # Record the calibrated thresholds and similarity distribution for this run.
            "similarity_profile": sim_profile,
        },
    }
    
    if FALKOR_ENABLED:
        persist_falkordb_edges(horizon_map)
        
    visualize_graph_pyvis(horizon_map, approved_modules)
    
    # Summary of generated graph
    print("\n  === HORIZON KNOWLEDGE GRAPH MERGED (SCALED DELTA) ===")
    print(f"  Stable kept edges: {len(kept_edges)} | Newly added edges: {len(normalized_new_edges)} | Total unique edges: {len(unique_edges)}")
    if accounting["bridge"]:
        print(f"  Bridge pass: {len(accounting['bridge'])} unlinked component(s) checked, "
              f"{sum(len(b['accepted']) for b in accounting['bridge'])} link(s) added")
    print(f"  Edge accounting: {accounting['returned']} returned | origins {accounting['origin_counts']} | "
          f"{len(accounting['dropped'])} dropped | "
          f"{sum(bool(a['empirical_caps']) for a in accounting['accepted'])} with a capped empirical weight")
    if name_corrections:
        print(f"  Name resolutions logged: {len(name_corrections)}")
    if mutation_log:
        print(f"  Edge mutations logged: {len(mutation_log)}")
    
    # 1. Edges
    print("\n [CROSS-DOMAIN EDGES]")
    for idx, edge in enumerate(horizon_map.get("knowledge_graph_edges", [])):
        direction = "➡️ " if edge.get("causal_directionality", "").lower().startswith("uni") else "🔁"
        print(f"  {idx+1}. {edge.get('source_module', 'Unknown')} {direction} {edge.get('target_module', 'Unknown')}")
        print(f"     └─ Type: {edge.get('interaction_type', 'N/A')}")
        print(f"     └─ Weights: Emp. Trend [{edge.get('empirical_trend_strength', '-')}/10] | Narrative [{edge.get('narrative_salience', '-')}/10] | Institutional [{edge.get('institutional_relevance', '-')}/10]")

    # 2. Weak Signals
    print("\n⚠️  [INFLEXION WEAK SIGNALS]")
    for signal in horizon_map.get("inflexion_weak_signals", []):
        print(f"  - {signal}")

    # 3. Sensitivities
    print("\n [PERTURBATION SENSITIVITIES]")
    for sens in horizon_map.get("perturbation_sensitivities", []):
        print(f"  - {sens}")

    # 4. Central Vulnerability
    print("\n [CENTRAL VULNERABILITY]")
    print(f"  > {horizon_map.get('central_vulnerability', 'N/A')}\n")
    
    print("="*60)
    
    return {"horizon_map": horizon_map}
