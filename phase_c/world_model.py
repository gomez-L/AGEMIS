import networkx as nx

COUPLING_WEIGHTS = {"empirical": 0.5, "narrative": 0.2, "institutional": 0.3}
PROPAGATION_ROUNDS = 6
COUPLING_SCALE = 0.9   # Maximum coupling when all three edge weights are 10.
ACTIVATION_THRESHOLD = 0.05
SEED_STRESS = 0.7
COMPOUND_SEED_STRESS = 0.5   # Initial stress per module in a two-module weak-signal shock.
MIN_TRANSMISSION = 0.01   # Skip transmissions at or below this amount before applyiyng stress.
BUFFERING_ATTENUATION = 0.4

IMPACT_FRAGILITY = {"high": 1.0, "medium": 0.6, "low": 0.2}
IMPACT_FRAGILITY_DEFAULT = 0.5
SUSCEPTIBILITY = {"high": 0.9, "medium": 0.7, "low": 0.5}
SUSCEPTIBILITY_DEFAULT = 0.7

# Word stems used to recognize buffering interaction labels.
_BUFFERING_MARKERS = (
    "mitigat", "stabiliz", "stabilis", "dampen", "buffer", "absorb", "hedg",
    "offset", "counteract", "reliev", "protect", "cushion", "moderat",
)


def _text(value) -> str:
    return str(value or "").strip().lower()


def _clamp(value, default=1) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(1, min(10, value))


def classify_polarity(interaction_type) -> str:
    return "buffering" if any(m in _text(interaction_type) for m in _BUFFERING_MARKERS) else "amplifying"


def composite_coupling(emp, nar, inst) -> float:
    w = COUPLING_WEIGHTS
    raw = (w["empirical"] * _clamp(emp)
           + w["narrative"] * _clamp(nar)
           + w["institutional"] * _clamp(inst)) / 10.0
    return raw * COUPLING_SCALE


def _uncertainty(certainty_level) -> float:
    text = _text(certainty_level)
    if "low" in text:
        return 1.0
    if "high" in text:
        return 0.3
    return 0.6


def fragility(module: dict) -> float:
    impact = IMPACT_FRAGILITY.get(_text(module.get("impact_potential")), IMPACT_FRAGILITY_DEFAULT)
    novelty = _clamp(module.get("novelty_score", 5), default=5) / 10.0
    return round(0.5 * impact + 0.3 * _uncertainty(module.get("certainty_level")) + 0.2 * novelty, 3)


def with_emerging(horizon_map: dict, modules: list) -> list:
    """Append provisional graph nodes whose names are absent from modules."""
    names = {m.get("name") for m in modules or []}
    extra = [n for n in (horizon_map or {}).get("emerging_nodes") or []
             if isinstance(n, dict) and n.get("name") and n["name"] not in names]
    return list(modules or []) + extra


def build_world_graph(horizon_map: dict, modules: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for module in modules or []:
        name = module.get("name")
        if not name:
            continue
        G.add_node(
            name,
            susceptibility=SUSCEPTIBILITY.get(_text(module.get("impact_potential")), SUSCEPTIBILITY_DEFAULT),
            fragility=fragility(module),
            impact_potential=module.get("impact_potential"),
            certainty_level=module.get("certainty_level"),
        )

    for edge in (horizon_map or {}).get("knowledge_graph_edges", []):
        src, tgt = edge.get("source_module"), edge.get("target_module")
        if not src or not tgt or src == tgt:
            continue
        for node in (src, tgt):
            if node not in G:
                G.add_node(node, susceptibility=SUSCEPTIBILITY_DEFAULT, fragility=0.5)

        # Use explicit polarity when valid; otherwise infer it from the interaction label.
        polarity = _text(edge.get("polarity"))
        if polarity not in ("amplifying", "buffering"):
            polarity = classify_polarity(edge.get("interaction_type"))
        coupling = composite_coupling(
            edge.get("empirical_trend_strength"),
            edge.get("narrative_salience"),
            edge.get("institutional_relevance"),
        )
        if polarity == "buffering":
            coupling *= BUFFERING_ATTENUATION

        attrs = {
            "interaction_type": edge.get("interaction_type", "INTERACTS"),
            "polarity": polarity,
            "coupling": coupling,
            "mechanism": edge.get("mechanism") or "",
            "evidence_titles": list(edge.get("evidence") or []),
            "provisional": bool(edge.get("provisional")),
        }
        G.add_edge(src, tgt, **attrs)
        if "bidirect" in _text(edge.get("causal_directionality")):
            G.add_edge(tgt, src, **attrs)
    return G


def rank_vulnerability_seeds(G: nx.DiGraph, limit: int = 5) -> list:
    """Rank shock origins by fragility and outgoing amplifying coupling, highest first."""
    seeds = []
    for node in G.nodes:
        # Exclude buffering edges from the outgoing coupling usd to rank shock origins
        successors = [v for v in G.successors(node) if G[node][v].get("polarity") != "buffering"]
        out_coupling = sum(G[node][v].get("coupling", 0.0) for v in successors)
        if out_coupling <= 0:
            continue
        frag = G.nodes[node].get("fragility", 0.5)
        seeds.append({
            "motif_type": "fragility_hotspot",
            "nodes": [node],
            "edges": [[node, v] for v in successors],
            "score": round(frag * (1.0 + out_coupling), 3),
            "rationale": f"'{node}' combines fragility {frag} with strong outgoing "
                         f"couplings (sum {out_coupling:.2f}), so a small shock here matters.",
        })
    seeds.sort(key=lambda seed: -seed["score"])
    return seeds if limit is None else seeds[:limit]


def propagate_stress(G: nx.DiGraph, seeds: dict, rounds: int = PROPAGATION_ROUNDS):
    """Propagate positive net stress gains over a bounded number of rounds.

    Each module transmits its net gain from the prev round rather than its acc stress. Skip direct returns to contributors from that round.
    Amplifying edges raise target stress; buffering edges lower it. Clamp stress
    to [0, 1] and record applied changes in the trace.
    """
    stress = {n: 0.0 for n in G.nodes}
    activation_round = {}
    trace = []

    # Module -> (net stress change last round, contributing modules).
    pending = {}
    for node, initial in seeds.items():
        if node in stress and initial > 0:
            stress[node] = min(1.0, float(initial))
            activation_round[node] = 0
            pending[node] = (stress[node], frozenset())

    for step in range(1, rounds + 1):
        if not pending:
            break
        gained = {}
        for u in sorted(pending):
            delta, came_from = pending[u]
            if delta <= 0:   # Propagate positive net changes only
                continue
            for v in G.successors(u):
                if v in came_from:
                    continue
                attrs = G[u][v]
                amount = (delta
                          * attrs.get("coupling", 0.0)
                          * G.nodes[v].get("susceptibility", SUSCEPTIBILITY_DEFAULT))
                if amount <= MIN_TRANSMISSION:
                    continue
                polarity = attrs.get("polarity", "amplifying")
                before = stress[v]
                stress[v] = min(1.0, max(0.0, before + (-amount if polarity == "buffering" else amount)))
                applied = stress[v] - before
                if abs(applied) < 1e-9:
                    continue
                if applied > 0:
                    activation_round.setdefault(v, step)
                trace.append({
                    "round": step,
                    "source": u,
                    "target": v,
                    "type": attrs.get("interaction_type", "INTERACTS"),
                    "polarity": polarity,
                    "mechanism": attrs.get("mechanism", ""),
                    "evidence_titles": list(attrs.get("evidence_titles") or []),
                    "provisional": bool(attrs.get("provisional")),
                    "stress_after": round(stress[v], 4),
                    "transmitted": round(applied, 4),
                })
                entry = gained.setdefault(v, [0.0, set()])
                entry[0] += applied
                entry[1].add(u)
        pending = {v: (delta, frozenset(sources)) for v, (delta, sources) in gained.items()}

    return {n: round(v, 6) for n, v in stress.items()}, activation_round, trace
