"""Scenario evidence context and reference checks."""

import re

from phase_c.schemas import ScenarioNarration

_REFERENCE_ID = re.compile(r"\b[se]\d+\b")


def evidence_context(modules, shock, events, *, cited_titles=None):
    """Include all unique signals from involved modules, with optional citation priority."""
    involved = {shock["target_module"], *(shock.get("co_targets") or [])}
    for event in events:
        involved.update((event["source_module"], event["affected_module"]))
    evidence, context, by_source = [], [], {}
    cited = {str(title).strip().casefold() for title in cited_titles or []}
    # Only include modules touched by the scenario with the trigger module first
    relevant = sorted((m for m in modules if m.get("name") in involved),
                      key=lambda m: m.get("name") != shock["target_module"])
    for module in relevant:
        context.append({key: str(module.get(key) or "")[:500] for key in
                        ("name", "temporal_horizon", "velocity", "inertia", "certainty_level")})
        context[-1]["provisional"] = bool(module.get("provisional"))
        for signal in module.get("evidence_signals") or []:
            if not isinstance(signal, dict):
                continue
            title = str(signal.get("title") or "")
            content = str(signal.get("content") or "")
            key = ("title", title.strip().casefold()) if title.strip() else ("content", content.strip())
            if key in by_source:
                entry = by_source[key]
                if module["name"] not in entry["modules"]:
                    entry["modules"].append(module["name"])
                # Prefer the more complete source when another module supplied only an excerpt
                if len(content) > len(entry["content"]):
                    entry["content"] = content
                continue
            entry = {"id": "", "module": module["name"], "modules": [module["name"]],
                     "title": title[:250], "content": content}
            if cited_titles is not None:
                entry["cited_by_path"] = title.strip().casefold() in cited
            by_source[key] = entry
            evidence.append(entry)
    if cited_titles is not None:
        evidence.sort(key=lambda entry: not entry["cited_by_path"])
    for number, entry in enumerate(evidence, start=1):
        entry["id"] = f"s{number}"
        entry["content"] = entry["content"][:1200]
    return {"modules": context, "evidence": evidence}


def validate_scenario(raw, events, context):
    """Validate references and downgrade under-supported claims rather than rejecting the scenario.
    Removes invalid reference IDs, then demotes any evidence/ simulation claim whose surviving refs no longer justify its basis to an assumption. But it never upgrades. Returns all changes in reference_repairs.
    """
    scenario = ScenarioNarration.model_validate(raw)
    event_ids = {event["id"] for event in events}
    evidence_ids = {signal["id"] for signal in context["evidence"]}
    repairs = []
    for number, step in enumerate(scenario.supporting_analysis, start=1):
        allowed = {"evidence": evidence_ids, "simulation": event_ids,
                   "assumption": evidence_ids | event_ids}[step.basis]
        # Models also write "[s1]", "S1, s2" or "e3 (bank run)". Get the IDs out of every entry.
        cited = list(dict.fromkeys(i for ref in step.references for i in _REFERENCE_ID.findall(str(ref).lower())))
        basis, kept = step.basis, [i for i in cited if i in allowed]
        if basis != "assumption" and not kept:
            basis, kept = "assumption", [i for i in cited if i in evidence_ids | event_ids]
        if basis != step.basis or kept != step.references:
            repairs.append({"step": number, "basis": step.basis, "repaired_basis": basis,
                            "references": list(step.references), "repaired_references": kept})
            step.basis, step.references = basis, kept
    result = scenario.model_dump()
    if repairs:
        result["reference_repairs"] = repairs
    return result
