"""Build provisional nodes from proto-modules & emerging weak signals.
Nodes rank by emerging_score and populate horizon_map["emerging_nodes"]. Exclude signals already represented by a proto-module.
"""
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

MAX_EMERGING_NODES = 5
MAX_PROVISIONAL_EMPIRICAL = 5  # Cap provisional-node links at "plausible".
FALLBACK_NAME_WORDS = 7

NAME_TEMPLATE = """Name each development below for a node label in a financial stability horizon graph.
One name per statement, in the same order: a noun phrase of 3-7 words naming the development itself,
no dashes or subtitles, no verbs like "is" or "are".
STATEMENTS:
{statements}
{format_instructions}"""


class NodeNames(BaseModel):
    names: list[str] = Field(description="One short name per statement, in the given order")


def _impact(score):
    return "High" if score >= 8 else "Medium" if score >= 5 else "Low"


def _fallback_name(statement):
    words = str(statement).rstrip(".").split()
    return " ".join(words[:FALLBACK_NAME_WORDS]) + ("…" if len(words) > FALLBACK_NAME_WORDS else "")


def _name_statements(statements, llm, parser_factory):
    """Generate names in one LLM call. And use opening words as a fallback."""
    names = []
    if statements and llm is not None and parser_factory is not None:
        try:
            parser = parser_factory(NodeNames)
            prompt = PromptTemplate(template=NAME_TEMPLATE, input_variables=["statements"],
                                    partial_variables={"format_instructions": parser.get_format_instructions()})
            reply = (prompt | llm | parser).invoke(
                {"statements": "\n".join(f"{n}. {s}" for n, s in enumerate(statements, 1))})
            names = NodeNames.model_validate(reply).names
        except Exception as exc:
            print(f"  [Emerging] Naming failed ({str(exc).splitlines()[0]}); using statement openings.")
    if len(names) != len(statements):
        names = []
    return [str(names[n]).strip() if names and str(names[n]).strip() else _fallback_name(s)
            for n, s in enumerate(statements)]


def _single_node(weak):
    statement = str(weak.get("statement")).strip()
    link = f" Driver: {weak['driver']}; effect: {weak['effect']}." if weak.get("driver") and weak.get("effect") else ""
    return {
        "provisional": True,
        "impact_potential": _impact(weak.get("impact") or 5),
        "certainty_level": "Low",
        "novelty_score": weak.get("novelty") or 5,
        "novelty_rationale": "Scored when the implication was extracted; not yet corroborated.",
        "reasoning": f"Early development backed by a single signal so far. {statement}{link}",
        "steeps_intersections": list(weak.get("domains") or []),
        "evidence_signals": [{"title": weak.get("signal") or statement, "content": statement}],
        "implications": [statement],
        "transmission_channels": [weak["mechanism"]] if weak.get("mechanism") else [],
        "drivers": [weak["driver"]] if weak.get("driver") else [],
        "effects": [weak["effect"]] if weak.get("effect") else [],
        "emerging_score": weak.get("emerging_score") or 0,
    }


def emerging_nodes(weak_signals, proto_modules, taken_names, llm=None, parser_factory=None,
                   limit=MAX_EMERGING_NODES):
    """Return up to limit provisional nodes, ranked by emerging_score."""
    candidates = [dict(p, provisional=True, certainty_level="Low")
                  for p in proto_modules or [] if isinstance(p, dict) and p.get("name")]
    seen_signals = set()
    for weak in sorted((w for w in weak_signals or [] if isinstance(w, dict) and w.get("statement")),
                       key=lambda w: -(w.get("emerging_score") or 0)):
        # Skip signals in proto-modules or already represented individually.
        if weak.get("tier") != "emerging" or weak.get("proto_module") or weak.get("signal") in seen_signals:
            continue
        seen_signals.add(weak.get("signal"))
        candidates.append(_single_node(weak))
    candidates.sort(key=lambda c: -(c.get("emerging_score") or 0))
    candidates = candidates[:limit]

    unnamed = [c for c in candidates if not c.get("name")]
    for node, name in zip(unnamed, _name_statements([c["implications"][0] for c in unnamed], llm, parser_factory)):
        node["name"] = name
    taken = set(taken_names)
    for node in candidates:
        base, n = node["name"], 2
        if node["name"] in taken:
            node["name"] = f"{base} (emerging)"
        while node["name"] in taken:
            node["name"], n = f"{base} (emerging {n})", n + 1
        taken.add(node["name"])
    # Put name first in the mapper's module JSON.
    return [{"name": node["name"], **{k: v for k, v in node.items() if k != "name"}} for node in candidates]
