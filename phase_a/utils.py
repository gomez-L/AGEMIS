import os
import json
from difflib import get_close_matches


def module_prompt_context(module):
    """Give the synthesizer enough scope & evidence to reuse a module with less effort."""
    evidence = {}
    for signal in module.get("evidence_signals") or []:
        title = (signal.get("title") or "").strip()
        identity = (title or signal.get("content") or "").strip().casefold()
        if identity:
            evidence.setdefault(identity, title)

    titles = [title for title in evidence.values() if title]
    # Show early evidence and latest title without sending the raw content
    sample_titles = titles if len(titles) <= 3 else titles[:2] + titles[-1:]
    reasoning = module.get("reasoning") or ""
    if len(reasoning) > 1000:
        reasoning = reasoning[:700] + "\n…\n" + reasoning[-300:]

    return {
        "name": module["name"],
        "velocity": module.get("velocity"),
        "inertia": module.get("inertia"),
        "steeps_intersections": module.get("steeps_intersections") or [],
        "temporal_horizon": module.get("temporal_horizon"),
        "reasoning": reasoning,
        "distinct_signal_count": len(evidence),
        "evidence_titles": [title[:180] for title in sample_titles],
    }


def load_domains():
    filepath = "domains.json"
    if not os.path.exists(filepath):
        return "Social, Technological, Economic, Environmental, Political, Security"
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        formatted_domains = []
        for category, subdomains in data.items():
            formatted_domains.append(f"{category}: {', '.join(subdomains)}")
        return "\n     - ".join(formatted_domains)
    except Exception as e:
        return "Social, Technological, Economic, Environmental, Political, Security"

def safe_get_module(m_name, modules_dict):
    if m_name in modules_dict:
        return m_name
    matches = get_close_matches(m_name, modules_dict.keys(), n=1, cutoff=0.8)
    return matches[0] if matches else None
