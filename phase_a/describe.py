"""Plain-language module summaries for readers.

A module's `reasoning` is the model's working note from building the module ("Implications 2, 3 and 5
form ...") and means little to readers. This step writes a short `summary` that opens the module in the
UI: what is happening, and how it could reach financial stability. The summary is regenerated only when
the module content it is based on changes (`summary_source`).
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import PromptTemplate

SUMMARY_PROMPT = PromptTemplate.from_template("""You write for central bankers and financial supervisors who
design monetary policy, regulation and supervision. Write the opening description of one topic from a
horizon-scanning tool, so that a reader who knows nothing about the tool understands in 15 seconds what
the topic is about.

Rules:
- Exactly 2 sentences, at most 55 words in total. Plain words, short sentences, no jargon you would not use in a briefing.
- Sentence 1: what is happening in the world. Be concrete: name the main actors, places or instruments from the material.
- Sentence 2: why it matters for financial stability, monetary policy or supervision: through which channel it could reach banks, markets, households or firms.
- Do not mention implications, modules, clusters, signals, numbers of items, the tool or the material itself.
- Do not start with the topic name. Do not invent facts that are not in the material.

Answer with the 2 sentences only, no heading, no quotes.

Topic name: {name}

Material:
{material}
""")


def _material(module):
    parts = {
        "What is driving it": module.get("drivers"),
        "What is changing": module.get("effects"),
        "Underlying developments": module.get("implications"),
        "How it could reach financial stability": module.get("transmission_channels"),
        "Why it is new": module.get("novelty_rationale"),
        "Evidence headlines": [s.get("title") for s in module.get("evidence_signals") or [] if isinstance(s, dict)],
        "Domains": module.get("steeps_intersections"),
    }
    lines = []
    for label, value in parts.items():
        if isinstance(value, list):
            value = [str(v) for v in value if v]
            if value:
                lines.append(f"{label}:\n" + "\n".join(f"- {v}" for v in value))
        elif value:
            lines.append(f"{label}: {value}")
    return "\n\n".join(lines)


def _source_hash(module):
    material = json.dumps({"name": module.get("name"), "material": _material(module),
                           "prompt": SUMMARY_PROMPT.template}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _summarize(module, llm):
    reply = llm.invoke(SUMMARY_PROMPT.format(name=module.get("name", ""), material=_material(module)))
    text = reply.content if isinstance(reply.content, str) else "".join(
        part.get("text", "") for part in reply.content if isinstance(part, dict))
    return " ".join(text.split()).strip('"“” ')


def add_module_summaries(modules, llm, max_workers=1, force=False):  # BFH key allows 2 parallel requests in total, shared with other callers
    """Fill `summary` on each module dict in place. Returns the number of summaries written.
    A failure on one module is reported and leaves that module without a new summary."""
    todo = [m for m in modules if isinstance(m, dict)
            and (force or not m.get("summary") or m.get("summary_source") != _source_hash(m))]
    if not todo:
        return 0

    def work(module):
        try:
            return module, _summarize(module, llm), None
        except Exception as exc:  # keep the pipeline going; the UI falls back to no summary
            return module, None, exc

    written = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for module, summary, error in pool.map(work, todo):
            if error is not None or not summary:
                print(f"[summary] Could not summarise '{module.get('name')}': {error}")
                continue
            module["summary"] = summary
            module["summary_source"] = _source_hash(module)
            written += 1
    return written
