"""Build Phase A modules by extracting, clustering, and synthesizing implications.

The pipeline stages are:
  1. Extract up to three structural implications from each relevant signal,
     using a separate LLM call for each signal.
  2. Cluster implication statements by embedding similarity. Apply distance,
     signal-count, and cohesion thresholds in code; recursively split oversized
     clusters while the splitting limits allow.
  3. Synthesize a module from each eligible cluster in one LLM call. The model
     may reject an incoherent cluster or identify individual outliers.
  4. Consolidate: module pairs with close centroids or shared signals are put
     to the LLM, which merges those describing one subsystem. A merged module
     lists the modules it absorbed as sub_themes.
  5. Reattach remaining weak implications to their nearest modules.
  6. Group leftover weak implications at a looser threshold and synthesize
     provisional proto-modules. Their members remain in the weak-signal pool.

Clusters with too few distinct signals or insufficient cohesion become weak
signals, along with implications rejected during synthesis. Weak signals are
ranked by novelty x impact x source trust; those scoring high on both novelty
and impact form the 'emerging' tier. Signals marked irrelevant or returned with
an empty implication list are recorded as discarded; failed extraction calls
are returned and checkpointed separately.

Thresholds are passed via PhaseAConfig. Environment defaults are read at run time.
Every run saves extraction, vectors, decisions, configuration, prompts and metrics.
Use cache_run to reuse extraction and embeddings for threshold comparisons.

Environment variables:
  PHASE_A_V2_DISTANCE      Cosine distance threshold (default 0.40).
                           Lower values produce finer clusters.
  PHASE_A_V2_LINKAGE       Clustering linkage: 'average' (default) or 'complete'.
                           Complete linkage constrains every pair in a cluster.
  PHASE_A_V2_MERGE_SIMILARITY  Centroid similarity (plus 0.05 per shared signal)
                           from which two modules are offered for merging
                           (default 0.7). Zero disables consolidation.
  PHASE_A_V2_PROTO_DISTANCE  Complete-linkage cosine distance for grouping leftover weak
                           implications into proto-modules (default 0.5). Zero disables it.
  PHASE_A_V2_MIN_SIGNALS   Minimum distinct signals per module (default 2).
  PHASE_A_V2_MAX_SHARE     Share of all signals used to set the cluster splitting
                           limit (default 0.25), subject to MIN_SIGNALS and the
                           splitting limits.
  PHASE_A_V2_MIN_COHESION  Minimum mean pairwise cosine similarity (default
                           1 - DISTANCE). Mainly relevant to average linkage
                           unless raised above the default.
"""
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from difflib import get_close_matches
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from main import RobustJsonOutputParser


@dataclass(frozen=True)
class PhaseAConfig:
    distance: float = 0.40
    min_signals: int = 2
    max_share: float = 0.25
    linkage: str = "average"
    min_cohesion: float | None = None
    split_factor: float = 0.7
    min_split_distance: float = 0.1
    max_steeps: int = 4
    reattach_candidates: int = 3  # Zero disables reattachment for comparison runs.
    merge_similarity: float = 0.7  # Zero disables consolidation.
    proto_distance: float = 0.5  # Zero disables the proto-module pass.

    def __post_init__(self):
        if self.min_cohesion is None:
            object.__setattr__(self, "min_cohesion", 1 - self.distance)
        if not 0 < self.distance <= 2 or not 0 <= self.min_cohesion <= 1:
            raise ValueError("Invalid distance or cohesion threshold")
        if self.linkage not in ("complete", "average"):
            raise ValueError("linkage must be complete or average")
        if self.min_signals < 1 or not 0 < self.max_share <= 1:
            raise ValueError("Invalid signal count or maximum share")
        if not 0 < self.split_factor < 1 or self.min_split_distance <= 0:
            raise ValueError("Invalid splitting limits")
        if self.max_steeps < 1 or self.reattach_candidates not in (0, 2, 3):
            raise ValueError("reattach_candidates must be 0, 2 or 3; max_steeps must be positive")
        if not 0 <= self.merge_similarity <= 1:
            raise ValueError("merge_similarity must be between 0 and 1")
        if not 0 <= self.proto_distance <= 2:
            raise ValueError("proto_distance must be between 0 and 2")

    @classmethod
    def from_env(cls):
        # Read at run time, never at import time.
        fields = {"distance": float, "min_signals": int, "max_share": float,
                  "linkage": str, "min_cohesion": float, "split_factor": float,
                  "min_split_distance": float, "max_steeps": int, "reattach_candidates": int,
                  "merge_similarity": float, "proto_distance": float}
        return cls(**{k: cast(os.environ[f"PHASE_A_V2_{k.upper()}"])
                      for k, cast in fields.items() if f"PHASE_A_V2_{k.upper()}" in os.environ})


# --- STEEPS taxonomy -------------------------------------------------------

def load_domain_labels(path="domains.json"):
    """Load canonical 'Category: Subdomain' labels, falling back to STEEPS categories."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [f"{cat}: {sub}" for cat, subs in data.items() for sub in subs]
    except (OSError, ValueError):
        return ["Social", "Technological", "Economic", "Environmental", "Political", "Security"]


def normalize_domains(raw, labels):
    """Map free-form domain strings onto canonical labels; drop what does not match."""
    by_subdomain = {label.split(": ", 1)[-1].casefold(): label for label in labels}
    by_full_label = {label.casefold(): label for label in labels}
    normalized = []
    for item in raw or []:
        key = str(item).strip().casefold()
        label = by_full_label.get(key) or by_subdomain.get(key)
        if not label:
            match = get_close_matches(
                key, list(by_full_label) + list(by_subdomain), n=1, cutoff=0.75
            )
            if match:
                label = by_full_label.get(match[0]) or by_subdomain.get(match[0])
        if label and label not in normalized:
            normalized.append(label)
    return normalized


# --- Schemas & prompts -----------------------------------------------------

class Implication(BaseModel):
    statement: str = Field(description="One sentence: the structural development this signal is evidence of, stated generally (not a news headline)")
    driver: str = Field(description="The causal force driving this structural development")
    effect: str = Field(description="The resulting structural change caused by that driver")
    mechanism: str = Field(description="How this development could transmit to financial stability (channel, not outcome)")
    domains: list[str] = Field(description="1-3 labels copied exactly from the allowed STEEPS list")
    novelty: int | str = Field(5, description="1-10, how new this development is against the well-known baseline of the past decade")
    impact: int | str = Field(5, description="1-10, potential systemic impact on financial stability if the development matures")


class Extraction(BaseModel):
    relevant: bool = Field(description="False if the signal is noise with no structural relevance")
    implications: list[Implication] = Field(description="1-3 distinct implications; empty if not relevant")


class ModuleSynthesis(BaseModel):
    coherent: bool = Field(description="False if the implications do not describe ONE structural subsystem with a shared transmission path")
    outliers: list[int] = Field(description="Numbers of implications that do not belong to this subsystem; empty if all belong")
    name: str = Field(description="Short canonical name, 3-7 words, no dashes or subtitles, no single institution or country unless the theme is inherently about it")
    velocity: str = Field(description="How fast this subsystem is evolving now")
    inertia: str = Field(description="Resistance or structural momentum of this subsystem")
    impact_potential: str = Field(description="High/Medium/Low potential systemic impact")
    certainty_level: str = Field(description="Degree of certainty based on the combined evidence")
    sentiment: str = Field(description="Negative/Neutral/Positive")
    temporal_horizon: str = Field(description="Immediate, Short-term, Medium-term or Long-term")
    novelty_score: int = Field(description="1-10, how new this structural shift is relative to the baseline")
    novelty_rationale: str = Field(description="Why this shift is or is not novel")
    reasoning: str = Field(description="How the listed implications combine into this subsystem")


EXTRACT_TEMPLATE = """You analyse signals for a financial stability horizon-scanning system (Swiss perspective).
Read ONE signal and list the 1-3 distinct STRUCTURAL developments it is evidence of.

Rules:
- State each implication generally, so that other signals about the same development would produce a similar sentence.
  Good: "Major central banks pause easing as inflation proves sticky."
  Bad:  "The Fed held rates at 4.25-4.50% on 29 January."
- For each implication, identify its causal driver and resulting structural effect.
- Do not split one development into several implications; at most 3.
- If the signal has no structural relevance, set relevant=false.
- novelty and impact: score each implication 1-10 independently and use the full range.
  A familiar, long-running development is novelty 1-3 even when its impact is high.
- domains: copy 1-3 labels EXACTLY from this list:
{domains}

SIGNAL:
{event}

{format_instructions}"""

SYNTH_TEMPLATE = """You are the Synthesis Engine of a financial stability horizon-scanning system (Swiss perspective).
The implications below were clustered together because they describe one structural subsystem (a 'module').
Write that module.

Granularity: a module is a structural subsystem with its own transmission path to financial stability.
- Too narrow: one institution's single decision, one report chapter, one incident.
- Too broad: "global geopolitics", "the UN agenda".
Name the subsystem, not the newest event.

The clustering is automatic and can be wrong:
- If the implications do not share one subsystem and transmission path, set coherent=false. Do not invent
  an umbrella theme ("polycrisis", "global fragility", "shock absorption") to cover unrelated items.
- If most belong together but some do not, list the numbers of those in outliers and write the module
  from the remaining implications only.

SUPPORTING SIGNALS: {signal_count}
IMPLICATIONS:
{implications}

{format_instructions}"""


# --- Pipeline steps --------------------------------------------------------

def _event_key(event):
    return (event.get("title") or "").strip()


def _score(value, default=5):
    try:
        return max(1, min(10, int(float(value))))
    except (TypeError, ValueError):
        return default


# Statements that share a driver and effect describe one mechanism; the statement alone
# clusters on shared vocabulary ("concentration", "single point of failure").
EMBED_TEXT = "{statement} {driver} -> {effect}"
EMERGING_MIN = 6  # Novelty and impact must both reach this for the emerging tier.
TRUST_WEIGHT = {"high": 1.0, "medium": 0.8, "low": 0.6}


def _emerging(m):
    """Rank a weak implication: novel, high-impact items from trusted sources come first."""
    trust = TRUST_WEIGHT.get(str(m["event"].get("source_trust_level", "")).lower(), 0.8)
    novelty, impact = _score(m.get("novelty")), _score(m.get("impact"))
    return {"novelty": novelty, "impact": impact,
            "emerging_score": round(novelty * impact * trust / 100, 3),
            "tier": "emerging" if min(novelty, impact) >= EMERGING_MIN else "weak"}


def _unique_name(base, modules):
    name, n = base, 2
    while name in {m["name"] for m in modules}:
        name, n = f"{base} ({n})", n + 1
    return name


def extract_implications(events, llm, labels, checkpoint=None, reuse=None):
    """Extract each signal, reusing cached results by event hash when available.

    An empty cached list marks a discarded signal.
    """
    reuse = reuse or {}
    parser = RobustJsonOutputParser(pydantic_object=Extraction)
    prompt = PromptTemplate(
        template=EXTRACT_TEMPLATE,
        input_variables=["event"],
        partial_variables={
            "domains": "\n".join(f"- {label}" for label in labels),
            "format_instructions": parser.get_format_instructions(),
        },
    )
    chain = prompt | llm | parser
    implications, discarded, failed = [], [], []
    for idx, event in enumerate(events):
        title = _event_key(event)
        saved = reuse.get(_hash(event))
        if saved is not None:
            print(f"  [EXTRACT {idx + 1}/{len(events)}] reused · {title[:90]}")
            if not saved:
                discarded.append({"signal_id": idx, "signal": title, "event": event})
            for item in saved:
                implications.append({**{k: v for k, v in item.items() if k not in ("id", "signal_id", "event")},
                                     "id": len(implications), "signal_id": idx, "event": event})
            if checkpoint:
                checkpoint({"implications": implications, "discarded": discarded,
                            "failed_extractions": failed, "processed_count": idx + 1})
            continue
        print(f"  [EXTRACT {idx + 1}/{len(events)}] {title[:100]}")
        try:
            result = chain.invoke({"event": json.dumps(event, ensure_ascii=False, indent=2)})
            validated = Extraction.model_validate(result)
            items = validated.implications[:3] if validated.relevant else []
            if any(not item.statement.strip() or not item.driver.strip() or not item.effect.strip()
                   for item in items):
                raise ValueError("Extraction contains an empty statement, driver or effect")
            if not items:
                discarded.append({"signal_id": idx, "signal": title, "event": event})
            for item in items:
                implications.append({
                    "id": len(implications), "signal_id": idx, "event": event,
                    "statement": item.statement.strip(), "driver": item.driver.strip(),
                    "effect": item.effect.strip(), "mechanism": item.mechanism.strip(),
                    "domains": normalize_domains(item.domains, labels),
                    "novelty": _score(item.novelty), "impact": _score(item.impact),
                })
        except Exception as exc:
            print(f"    extraction failed: {exc}")
            failed.append({"signal_id": idx, "signal": title, "event": event,
                           "error": str(exc)})
        if checkpoint:
            checkpoint({"implications": implications, "discarded": discarded,
                        "failed_extractions": failed, "processed_count": idx + 1})
    return implications, discarded, failed


def _cluster(vectors, threshold, config):
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage=config.linkage,
    )
    return model.fit_predict(vectors)


def _distinct_events(members):
    return {m.get("signal_id", _event_key(m["event"])) for m in members}


def _cohesion(vectors):
    """Mean pairwise cosine similarity of unit vectors (1.0 for a single vector)."""
    vector_count = len(vectors)
    if vector_count < 2:
        return 1.0
    similarities = vectors @ vectors.T
    return float(
        (similarities.sum() - np.trace(similarities)) / (vector_count * (vector_count - 1))
    )


def _split(implications, vectors, idxs, threshold, max_events, config):
    """Tighten oversized clusters until a size, member-count, or distance limit is reached."""
    if len(_distinct_events([implications[i] for i in idxs])) <= max_events or len(idxs) <= 2:
        return [idxs]
    tighter = threshold * config.split_factor
    if tighter < config.min_split_distance:
        print(f"  [SPLIT] cluster of {len(idxs)} implications still oversized at the minimum threshold")
        return [idxs]
    sub = _cluster(vectors[idxs], tighter, config)
    parts = [[idxs[k] for k in range(len(idxs)) if sub[k] == s] for s in sorted(set(sub))]
    if len(parts) == 1:  # Keep tightening when this round did not separate any members.
        return _split(implications, vectors, idxs, tighter, max_events, config)
    out = []
    for part in parts:
        out.extend(_split(implications, vectors, part, tighter, max_events, config))
    return out


def cluster_implications(implications, embedder, total_events, config=None, *, vectors=None, audit=None):
    config = config or PhaseAConfig.from_env()
    # The financial-stability channel is left out of the embedding: its generic wording
    # pulls unrelated themes together.
    if not implications:
        return [], []
    if vectors is None:
        vectors = _embed(implications, embedder)
    labels = _cluster(vectors, config.distance, config)

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    # Recursively split clusters that exceed the distinct-signal limit.
    max_events = max(config.min_signals, int(config.max_share * total_events))
    final = []
    for idxs in groups.values():
        parts = _split(implications, vectors, idxs, config.distance, max_events, config)
        if len(parts) > 1:
            print(f"  [SPLIT] catch-all cluster of {len(idxs)} implications into {len(parts)}")
        final.extend(parts)

    clusters, weak = [], []
    if audit is not None:
        audit.extend({"cluster_id": n, "implication_ids": idxs,
                      "cohesion": _cohesion(vectors[idxs]),
                      "signal_count": len(_distinct_events([implications[i] for i in idxs]))}
                     for n, idxs in enumerate(final))
    for idxs in final:
        members = [implications[i] for i in idxs]
        if len(_distinct_events(members)) < config.min_signals:
            weak.append(members)
            continue
        cohesion = _cohesion(vectors[idxs])
        if cohesion < config.min_cohesion:
            print(f"  [LOOSE] cluster of {len(idxs)} implications demoted to weak signals "
                  f"(cohesion {cohesion:.2f} < {config.min_cohesion:.2f})")
            weak.append(members)
            continue
        clusters.append(members)
    # Sort by distinct-signal count, then by the first statement to break ties.
    clusters.sort(key=lambda m: (-len(_distinct_events(m)), m[0]["statement"]))
    return clusters, weak


def _is_false(value):
    return value is False or str(value).strip().lower() in ("false", "no", "0")


def synthesize_module(members, llm, config=None, audit=None):
    """Return (module, rejected_members); reject incoherent or insufficiently supported clusters."""
    config = config or PhaseAConfig.from_env()
    parser = RobustJsonOutputParser(pydantic_object=ModuleSynthesis)
    prompt = PromptTemplate(
        template=SYNTH_TEMPLATE,
        input_variables=["implications", "signal_count"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    lines = [
        f"{n}. {m['statement']} (driver: {m.get('driver', '')}; effect: {m.get('effect', '')}; "
        f"channel: {m['mechanism']}) [signal: {_event_key(m['event'])[:120]}]"
        for n, m in enumerate(members, 1)
    ]
    module = (prompt | llm | parser).invoke({
        "implications": "\n".join(lines),
        "signal_count": len(_distinct_events(members)),
    })

    if audit is not None:
        audit["response"] = module
    module = ModuleSynthesis.model_validate(module).model_dump()
    # Rejected clusters and individual outliers become weak signals in the caller.
    coherent = not _is_false(module.pop("coherent", True))
    outliers = set()
    for n in module.pop("outliers", None) or []:
        try:
            outliers.add(int(n))
        except (TypeError, ValueError):
            pass
    kept = [m for n, m in enumerate(members, 1) if n not in outliers]
    if audit is not None:
        audit.update(coherent=coherent, outlier_ids=[m["id"] for n, m in enumerate(members, 1) if n in outliers])
    if not coherent or len(_distinct_events(kept)) < config.min_signals:
        if audit is not None:
            audit["status"] = "rejected"
        return None, members
    if audit is not None:
        audit["status"] = "accepted"
    rejected = [m for n, m in enumerate(members, 1) if n in outliers]
    members = kept

    # Derive domain rankings and deduplicated evidence from the retained members.
    domain_counts = Counter(d for m in members for d in m["domains"])
    module["steeps_intersections"] = [d for d, _ in domain_counts.most_common(config.max_steeps)]
    evidence, seen = [], set()
    for m in members:
        key = m.get("signal_id", _event_key(m["event"]))
        if key not in seen:
            seen.add(key)
            evidence.append({"title": _event_key(m["event"]), "content": m["event"].get("content", "")})
    module["evidence_signals"] = evidence
    module["implications"] = [m["statement"] for m in members]
    # Pass up to eight unique, nonempty mechanisms for Phase B for causal mapping
    module["transmission_channels"] = list(dict.fromkeys(m["mechanism"] for m in members if m["mechanism"]))[:8]
    # Retain causal terms for Phase B's directed candidates.
    module["drivers"] = list(dict.fromkeys(m["driver"] for m in members if m.get("driver")))[:8]
    module["effects"] = list(dict.fromkeys(m["effect"] for m in members if m.get("effect")))[:8]
    return module, rejected


class MergeVerdict(BaseModel):
    merge: bool = Field(description="True only if both modules describe the SAME structural subsystem and transmission path")
    reasoning: str = Field(description="Why the modules are one subsystem or two")


MERGE_TEMPLATE = """Two modules of a financial stability horizon-scanning system may describe the same structural subsystem.
Merge them when one would be a sub-theme or facet of the other: the same driver seen from two angles,
the same vulnerability in two regions or institutions, or one a narrower case of the other.
Keep them apart when they are distinct subsystems that merely share a topic or act on each other;
those are linked as cause and effect in a later step, and merging them would hide that link.
MODULE A: {a}
MODULE B: {b}
{format_instructions}"""


def _centroid(vectors, members):
    centroid = vectors[[m["id"] for m in members]].mean(axis=0)
    return centroid / max(np.linalg.norm(centroid), 1e-12)


def _consolidate(modules, memberships, vectors, llm, config, records, checkpoint):
    """Merge modules describing one subsystem in place; return members dropped as outliers."""
    parser = RobustJsonOutputParser(pydantic_object=MergeVerdict)
    prompt = PromptTemplate(template=MERGE_TEMPLATE, input_variables=["a", "b"],
                            partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm | parser

    def brief(module):
        return json.dumps({k: module.get(k) for k in ("name", "reasoning", "implications", "transmission_channels")},
                          ensure_ascii=False)

    tried, dropped = set(), []
    while config.merge_similarity:
        centroids = [_centroid(vectors, members) for members in memberships]
        signals = [_distinct_events(members) for members in memberships]
        best = None
        for a, b in combinations(range(len(modules)), 2):
            key = frozenset(tuple(m["id"] for m in memberships[n]) for n in (a, b))
            shared = len(signals[a] & signals[b])
            score = float(centroids[a] @ centroids[b]) + 0.05 * shared
            if key not in tried and score >= config.merge_similarity and (best is None or score > best[0]):
                best = (score, a, b, key, shared)
        if best is None:
            break
        score, a, b, key, shared = best
        tried.add(key)
        record = {"modules": [modules[a]["name"], modules[b]["name"]], "score": round(score, 3),
                  "shared_signals": shared, "status": "checking"}
        records.append(record)
        started = perf_counter()
        print(f"  [MERGE CHECK {len(records)}] {' + '.join(record['modules'])} (score {score:.3f})", flush=True)
        checkpoint()
        try:
            verdict = MergeVerdict.model_validate(chain.invoke({"a": brief(modules[a]), "b": brief(modules[b])}))
            record["response"] = verdict.model_dump()
            record["status"] = "kept_apart"
            if verdict.merge:
                print("    merge accepted; synthesizing combined module...", flush=True)
                synthesis = {}
                record["synthesis"] = synthesis
                merged, rejected = synthesize_module(memberships[a] + memberships[b], llm, config, synthesis)
                kept = [m for m in memberships[a] + memberships[b] if m not in rejected]
                # A merge that keeps nothing of one side is an eviction, not a merge.
                if merged is not None and all(any(m in kept for m in memberships[n]) for n in (a, b)):
                    others = [m for n, m in enumerate(modules) if n not in (a, b)]
                    merged["name"] = _unique_name(merged.get("name", "").strip() or modules[a]["name"], others)
                    merged["sub_themes"] = list(dict.fromkeys(
                        t for n in (a, b) for t in modules[n].get("sub_themes") or [modules[n]["name"]]))
                    modules[a], memberships[a] = merged, kept
                    del modules[b], memberships[b]
                    dropped.extend(rejected)
                    record.update(status="merged", merged_name=merged["name"])
                    print(f"  [MERGE] {' + '.join(record['modules'])} -> {merged['name']}")
                else:
                    record["status"] = "synthesis_declined"
        except Exception as exc:
            record.update(status="failed", error=str(exc))
        record["elapsed_seconds"] = round(perf_counter() - started, 2)
        print(f"    {record['status']} ({record['elapsed_seconds']:.1f}s)", flush=True)
        checkpoint()
    print(f"  [CONSOLIDATE DONE] {len(records)} checks, "
          f"{sum(r['status'] == 'merged' for r in records)} merges, {len(modules)} modules remain", flush=True)
    return dropped


class AttachmentVerdict(BaseModel):
    accept: bool = Field(description="True only if the implication shares this module's subsystem and transmission path")
    reasoning: str = Field(description="Why the implication belongs or should remain weak")


ATTACH_TEMPLATE = """Assess one weak implication for an existing financial stability horizon-scanning module.
Accept only if it belongs to the SAME structural subsystem and transmission path. Shared topic alone
is insufficient. Cross-domain evidence is welcome when the underlying process is shared.
Do not broaden the module to accommodate unrelated evidence.
MODULE: {module}
WEAK IMPLICATION: {implication}
{format_instructions}"""


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write(path, value):
    """Atomic stage checkpoints leave previous data readable after an interrupted write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_name(model):
    model = getattr(model, "bound", model)
    for attr in ("model_name", "model_id", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
    return f"{type(model).__module__}.{type(model).__name__}"


def latest_run_dir(root="out/phase_a_v2"):
    """Return the last run folder by name with an extraction checkpoint, or None."""
    runs = (
        sorted(p for p in Path(root).glob("*") if (p / "extraction.json").is_file())
        if Path(root).is_dir() else []
    )
    return runs[-1] if runs else None


def _vector_key(event_hash, item):
    return (event_hash, item["statement"], item.get("driver", ""), item.get("effect", ""))


def _reusable_extraction(source, identity, llm_name, embedder_name):
    """Load compatible extraction and vector caches from an earlier run.

    Returns implications keyed by event hash and vectors keyed by _vector_key.
    Failed extractions are excluded so the caller can retry them.
    """
    try:
        previous = _read(source / "config.json")
        extracted = _read(source / "extraction.json")
        events = _read(source / "events.json")
    except (OSError, ValueError) as exc:
        print(f"  Not reusing {source}: {exc}")
        return {}, {}
    identity_then = previous.get("cache_identity", {})
    extraction_llm = previous.get("extraction_provenance", {}).get("llm", previous.get("llm"))
    compatibility = (
        ("taxonomy", identity_then.get("labels_hash"), identity["labels_hash"]),
        ("extraction prompt", identity_then.get("extraction_prompt_hash"), identity["extraction_prompt_hash"]),
        ("LLM", extraction_llm, llm_name),
    )
    for what, then, now in compatibility:
        if then != now:
            print(f"  Not reusing {source}: {what} changed")
            return {}, {}
    implications = extracted.get("implications", [])
    if previous.get("implications_hash") != _hash(implications):
        print(f"  Not reusing {source}: saved implications were modified")
        return {}, {}
    failed = {f["signal_id"] for f in extracted.get("failed_extractions", [])}
    by_signal = {}
    for item in implications:
        by_signal.setdefault(item["signal_id"], []).append(item)
    reuse = {_hash(event): by_signal.get(idx, [])
             for idx, event in enumerate(events[:extracted.get("processed_count", 0)]) if idx not in failed}

    vectors = {}
    path = source / "embeddings.npy"
    if (path.is_file() and previous.get("embedder") == embedder_name
            and previous.get("embed_text") == EMBED_TEXT):
        try:
            saved = np.load(path, allow_pickle=False)
            _validate_vectors(saved, len(implications))
            if previous.get("embeddings_hash") == _hash(saved.tolist()):
                vectors = {_vector_key(_hash(item["event"]), item): saved[item["id"]] for item in implications}
        except (OSError, ValueError):
            vectors = {}
    return reuse, vectors


def _embed_reusing(implications, embedder, saved):
    """Combine cached unit vectors with embeddings for uncached implications."""
    keys = [_vector_key(_hash(m["event"]), m) for m in implications]
    missing = [m for m, key in zip(implications, keys) if key not in saved]
    fresh = iter(_embed(missing, embedder)) if missing else iter(())
    print(f"  Embeddings: {len(implications) - len(missing)} reused, {len(missing)} new")
    if not implications:
        return np.empty((0, 0))
    return np.vstack([saved[key] if key in saved else next(fresh) for key in keys])


def _embed(implications, embedder):
    if not implications:
        return np.empty((0, 0))
    texts = [EMBED_TEXT.format(statement=m["statement"], driver=m.get("driver", ""),
                               effect=m.get("effect", "")) for m in implications]
    vectors = np.asarray(embedder.embed_documents(texts), dtype=float)
    _validate_vectors(vectors, len(implications))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _validate_vectors(vectors, count):
    if vectors.ndim != 2 or len(vectors) != count or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a finite matrix aligned with the cached implications")
    if count and (not vectors.shape[1] or np.any(np.linalg.norm(vectors, axis=1) < 1e-12)):
        raise ValueError("Embeddings contain empty or zero vectors")


def _partition(audit, size):
    labels = [-1] * size
    for cluster in audit:
        for idx in cluster["implication_ids"]:
            labels[idx] = cluster["cluster_id"]
    return labels


def _reattach(weak, modules, memberships, vectors, llm, config, records, checkpoint):
    parser = RobustJsonOutputParser(pydantic_object=AttachmentVerdict)
    prompt = PromptTemplate(template=ATTACH_TEMPLATE, input_variables=["module", "implication"],
                            partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | llm | parser
    remaining = []
    for position, item in enumerate(weak, 1):
        if not modules or not config.reattach_candidates:
            remaining.append(item)
            continue
        # Recompute centroids after each accepted attachment.
        centroids = [vectors[[m["id"] for m in members]].mean(axis=0) for members in memberships]
        scores = [float(vectors[item["id"]] @ c / max(np.linalg.norm(c), 1e-12)) for c in centroids]
        nearest = sorted(range(len(modules)), key=lambda n: (-scores[n], n))[:config.reattach_candidates]
        attached = False
        for rank, n in enumerate(nearest, 1):
            record = {"implication_id": item["id"], "module_index": n,
                      "similarity": scores[n], "status": "checking"}
            records.append(record)
            started = perf_counter()
            print(f"  [REATTACH {position}/{len(weak)} - candidate {rank}/{len(nearest)}] "
                  f"implication {item['id']} -> {modules[n]['name']}", flush=True)
            checkpoint()
            try:
                verdict = AttachmentVerdict.model_validate(chain.invoke({
                    "module": json.dumps(modules[n], ensure_ascii=False),
                    "implication": json.dumps(item, ensure_ascii=False),
                }))
                record["response"] = verdict.model_dump()
                record["status"] = "declined"
                if verdict.accept:
                    print("    attachment accepted; synthesizing updated module...", flush=True)
                    # Refresh the module's description as well as its evidence. Decline only when
                    # the synthesizer flags the new item itself; existing members it flags return
                    # to the weak pool and are logged, never dropped silently.
                    synthesis = {}
                    record["synthesis"] = synthesis
                    candidate, rejected = synthesize_module(memberships[n] + [item], llm, config, synthesis)
                    if candidate is not None and item not in rejected:
                        candidate["name"] = modules[n]["name"]
                        if modules[n].get("sub_themes"):
                            candidate["sub_themes"] = modules[n]["sub_themes"]
                        modules[n] = candidate
                        memberships[n] = [m for m in memberships[n] + [item] if m not in rejected]
                        remaining.extend(rejected)
                        record.update(status="accepted", evicted_ids=[m["id"] for m in rejected])
                        attached = True
                    else:
                        record["status"] = "synthesis_declined"
            except Exception as exc:
                record.update(status="failed", error=str(exc))
            record["elapsed_seconds"] = round(perf_counter() - started, 2)
            print(f"    {record['status']} ({record['elapsed_seconds']:.1f}s)", flush=True)
            checkpoint()
            if attached:
                break
        if not attached:
            remaining.append(item)
    return remaining


def _proto_modules(weak, modules, vectors, llm, config, records):
    """Synthesize provisional modules from clusters of leftover weak implications.

    Complete linkage limits the distance between every pair. Accepted members
    remain in the weak pool.
    """
    if not config.proto_distance or len(weak) < 2:
        return []
    labels = _cluster(vectors[[m["id"] for m in weak]], config.proto_distance,
                      replace(config, linkage="complete"))
    groups = {}
    for member, label in zip(weak, labels):
        groups.setdefault(int(label), []).append(member)
    protos = []
    for members in sorted(groups.values(), key=lambda g: (-len(_distinct_events(g)), g[0]["id"])):
        if len(_distinct_events(members)) < max(2, config.min_signals):
            continue
        record = {"implication_ids": [m["id"] for m in members]}
        records.append(record)
        try:
            module, rejected = synthesize_module(members, llm, config, record)
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            continue
        if module is None:
            continue
        kept = [m for m in members if m not in rejected]
        module["name"] = _unique_name(module.get("name", "").strip() or f"Proto-module {len(protos) + 1}",
                                      modules + protos)
        # Proto-modules retain low certainty regardless of the synthesis response.
        module.update(provisional=True, certainty_level="Low",
                      implication_ids=[m["id"] for m in kept],
                      emerging_score=max(_emerging(m)["emerging_score"] for m in kept))
        protos.append(module)
        print(f"  [PROTO] {module['name']} ({len(_distinct_events(kept))} signals)")
    return protos


def _metrics(events, implications, weak, modules, memberships, vectors, clusters, verdicts,
             attachments, failed, discarded, stability):
    total = len(events)
    weak_ids = _distinct_events(weak)
    module_ids = _distinct_events([m for members in memberships for m in members])
    module_metrics = []
    for module, members in zip(modules, memberships):
        # Count each category at most once per implication, regardless of subdomain count.
        categories = Counter(d for m in members for d in {x.split(":", 1)[0] for x in m["domains"]})
        module_metrics.append({"name": module["name"], "signal_count": len(_distinct_events(members)),
                               "implication_count": len(members),
                               "cohesion": _cohesion(vectors[[m["id"] for m in members]]),
                               "steeps_counts": dict(categories), "steeps_spread": len(categories)})
    trials = len(verdicts)
    outliers = sum(len(v.get("outlier_ids", [])) for v in verdicts)
    offered = sum(len(v["implication_ids"]) for v in verdicts)
    return {
        "coverage": {"input_signals": total, "extracted_signals": len(_distinct_events(implications)),
                     "weak_signals": len(weak_ids), "weak_signal_share": len(weak_ids) / total if total else 0,
                     "module_signal_share": len(module_ids) / total if total else 0,
                     "weak_only_signal_share": len(weak_ids - module_ids) / total if total else 0,
                     "failed_extractions": len(failed), "discarded_signals": len(discarded),
                     "module_size_distribution": dict(Counter(m["signal_count"] for m in module_metrics))},
        "quality": {"clusters": clusters, "modules": module_metrics,
                    "synthesis_attempts": trials,
                    "synthesis_rejection_rate": sum(v.get("status") == "rejected" for v in verdicts) / trials if trials else 0,
                    "synthesis_failure_rate": sum(v.get("status") == "failed" for v in verdicts) / trials if trials else 0,
                    "outlier_rate": outliers / offered if offered else 0,
                    "outlier_implications": outliers, "synthesized_implications": offered},
        "reattachment": {"offers": len(attachments), "accepted": sum(r["status"] == "accepted" for r in attachments)},
        "stability": stability,
    }


def _print_metrics(metrics):
    c, q = metrics["coverage"], metrics["quality"]
    print("\n--- [Phase A V2] METRICS ---")
    print(f"  Coverage: {c['weak_signal_share']:.1%} of input signals have weak implications; "
          f"module sizes (signals: modules) {c['module_size_distribution']}; "
          f"failed extractions {c['failed_extractions']}")
    print(f"  Quality: synthesis rejected {q['synthesis_rejection_rate']:.1%}, "
          f"failed {q['synthesis_failure_rate']:.1%}, outliers {q['outlier_rate']:.1%}")
    if "consolidation" in metrics:
        print(f"  Consolidation: {metrics['consolidation']['merged']} merges from "
              f"{metrics['consolidation']['checks']} checks; "
              f"emerging issues {c.get('emerging_issues', 0)}; proto-modules {c.get('proto_modules', 0)}")
    print("  Cluster cohesion: " + ", ".join(f"{c['cluster_id']}: {c['cohesion']:.3f}" for c in q["clusters"]))
    for m in q["modules"]:
        print(f"  {m['name']}: cohesion {m['cohesion']:.3f}; STEEPS spread {m['steeps_spread']} {m['steeps_counts']}")
    for comparison in metrics["stability"]:
        print(f"  Stability: distance {comparison['distance']:.3f}, ARI {comparison['adjusted_rand_index']}")


def run_phase_a_v2(events, llm, embedder, config=None, *, run_dir=None, cache_run=None,
                   reuse_run=None, llm_name=None, embedder_name=None):
    """Run with disk checkpoints. Reuse extraction and vectors with cache_run=<previous run>.

    reuse_run accepts partial caches and sends only new or failed signals to the LLM.
    Clustering and later stages always process all signals.

    Returns modules, weak_signals, discarded and failed_extractions, plus metrics/run_dir.
    Extraction cache identity includes ordered inputs, taxonomy and extraction prompt/schema.
    Stability compares pre-synthesis partitions at distance +/- 0.05 on these same vectors.
    """
    config = config or PhaseAConfig.from_env()
    events = sorted(events, key=lambda e: (_event_key(e), _hash(e)))
    labels = load_domain_labels()
    prompts = {
        "extraction": {"template": EXTRACT_TEMPLATE, "schema": Extraction.model_json_schema()},
        "synthesis": {"template": SYNTH_TEMPLATE, "schema": ModuleSynthesis.model_json_schema()},
        "consolidation": {"template": MERGE_TEMPLATE, "schema": MergeVerdict.model_json_schema()},
        "reattachment": {"template": ATTACH_TEMPLATE, "schema": AttachmentVerdict.model_json_schema()},
    }
    identity = {"events_hash": _hash(events), "labels_hash": _hash(labels),
                "extraction_prompt_hash": _hash(prompts["extraction"])}
    folder = Path(run_dir) if run_dir else Path("out/phase_a_v2") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    metadata = {"version": 1, "config": asdict(config), "cache_identity": identity,
                "llm": llm_name or _model_name(llm), "embedder": embedder_name or _model_name(embedder),
                "llm_parameters": getattr(llm, "kwargs", {}), "embed_text": EMBED_TEXT,
                "prompt_hashes": {k: _hash(v) for k, v in prompts.items()},
                "cache_run": str(Path(cache_run).resolve()) if cache_run else None, "status": "started"}
    _write(folder / "config.json", metadata)
    _write(folder / "prompts.json", prompts)
    _write(folder / "events.json", events)
    _write(folder / "domains.json", labels)
    try:
        if cache_run:
            source = Path(cache_run)
            previous = _read(source / "config.json")
            if previous["cache_identity"] != identity:
                raise ValueError("Extraction cache does not match input signals, taxonomy or extraction prompt/schema")
            if (source / "embeddings.npy").exists() and previous["embedder"] != metadata["embedder"]:
                raise ValueError("Cached embeddings were produced by a different embedder")
            extracted = _read(source / "extraction.json")
            if extracted["processed_count"] != len(events):
                raise ValueError("Extraction cache is incomplete")
            if previous.get("implications_hash") != _hash(extracted["implications"]):
                raise ValueError("Cached implications have changed")
            metadata["extraction_provenance"] = previous.get("extraction_provenance", {
                "llm": previous["llm"], "llm_parameters": previous["llm_parameters"],
                "prompt_hash": previous["prompt_hashes"]["extraction"]})
            _write(folder / "extraction.json", extracted)
        else:
            reuse, saved_vectors = {}, {}
            if reuse_run:
                reuse, saved_vectors = _reusable_extraction(Path(reuse_run), identity, metadata["llm"],
                                                            metadata["embedder"])
                reused = sum(_hash(e) in reuse for e in events)
                metadata["reuse_run"] = str(Path(reuse_run).resolve())
                metadata["reused_signals"] = reused
                print(f"  Reusing extraction for {reused}/{len(events)} signals from {reuse_run}")
            print(f"\n--- [Phase A V2] EXTRACT ({len(events)} signals) ---")
            def checkpoint_extraction(data):
                _write(folder / "extraction.json", data)
                metadata["implications_hash"] = _hash(data["implications"])
                _write(folder / "config.json", metadata)

            implications, discarded, failed = extract_implications(
                events, llm, labels, checkpoint_extraction, reuse)
            extracted = {"implications": implications, "discarded": discarded,
                         "failed_extractions": failed, "processed_count": len(events)}
            _write(folder / "extraction.json", extracted)
        implications = extracted["implications"]
        discarded, failed = extracted["discarded"], extracted["failed_extractions"]
        metadata["implications_hash"] = _hash(implications)
        _write(folder / "config.json", metadata)
        # Vectors cached from a different embedding text are recomputed, not reused.
        reuse_vectors = bool(cache_run and (Path(cache_run) / "embeddings.npy").exists()
                             and previous.get("embed_text") == EMBED_TEXT)
        if reuse_vectors:
            vectors = np.load(Path(cache_run) / "embeddings.npy", allow_pickle=False)
        elif not cache_run and reuse_run:
            vectors = _embed_reusing(implications, embedder, saved_vectors)
        else:
            vectors = _embed(implications, embedder)
        _validate_vectors(vectors, len(implications))
        if reuse_vectors and previous.get("embeddings_hash") != _hash(vectors.tolist()):
            raise ValueError("Cached embeddings have changed")
        np.save(folder / "embeddings.npy", vectors, allow_pickle=False)
        metadata["embeddings_hash"] = _hash(vectors.tolist())
        _write(folder / "config.json", metadata)
        print(f"\n--- [Phase A V2] CLUSTER ({len(implications)} implications) ---")
        clusters_audit = []
        clusters, weak_groups = cluster_implications(implications, embedder, len(events), config,
                                                    vectors=vectors, audit=clusters_audit)
        eligible = {tuple(m["id"] for m in members) for members in clusters}
        for c in clusters_audit:
            c["status"] = "eligible" if tuple(c["implication_ids"]) in eligible else "weak"
            if c["status"] == "weak":
                c["reason"] = "insufficient_signals" if c["signal_count"] < config.min_signals else "low_cohesion"
        assignments = _partition(clusters_audit, len(implications))
        _write(folder / "clusters.json", {"clusters": clusters_audit, "assignments": assignments})
        modules, memberships, verdicts, attachments = [], [], [], []
        _write(folder / "synthesis.json", verdicts)
        weak = [m for group in weak_groups for m in group]
        for members in clusters:
            verdict = {"implication_ids": [m["id"] for m in members]}
            verdicts.append(verdict)
            try:
                module, rejected = synthesize_module(members, llm, config, verdict)
                weak.extend(rejected)
                if module is not None:
                    module["name"] = _unique_name(
                        module.get("name", "").strip() or f"Module {len(modules) + 1}", modules)
                    modules.append(module)
                    memberships.append([m for m in members if m not in rejected])
            except Exception as exc:
                verdict.update(status="failed", error=str(exc))
                weak.extend(members)
            _write(folder / "synthesis.json", verdicts)
            _write(folder / "modules.json", modules)
        print(f"\n--- [Phase A V2] CONSOLIDATE ({len(modules)} modules) ---")
        merges = []
        weak.extend(_consolidate(modules, memberships, vectors, llm, config, merges,
                                 lambda: _write(folder / "consolidation.json", merges)))
        _write(folder / "consolidation.json", merges)
        _write(folder / "modules.json", modules)
        _write(folder / "reattachment.json", attachments)
        def checkpoint_attachment():
            _write(folder / "reattachment.json", attachments)
            _write(folder / "modules.json", modules)
            _write(folder / "memberships.json", [[m["id"] for m in group] for group in memberships])

        _write(folder / "weak_before_reattachment.json", weak)
        print(f"\n--- [Phase A V2] REATTACH ({len(weak)} weak implications, "
              f"up to {config.reattach_candidates} candidates each) ---", flush=True)
        weak = _reattach(sorted(weak, key=lambda m: m["id"]), modules, memberships,
                         vectors, llm, config, attachments, checkpoint_attachment)
        print(f"\n--- [Phase A V2] PROTO-MODULES ({len(weak)} leftover weak implications) ---", flush=True)
        proto_attempts = []
        protos = _proto_modules(weak, modules, vectors, llm, config, proto_attempts)
        _write(folder / "proto_modules.json", {"modules": protos, "attempts": proto_attempts})
        proto_of = {i: p["name"] for p in protos for i in p["implication_ids"]}
        stability = []
        if cache_run and (Path(cache_run) / "clusters.json").exists():
            cached_assignments = _read(Path(cache_run) / "clusters.json")["assignments"]
            if len(cached_assignments) != len(implications):
                raise ValueError("Cached cluster assignments do not align with implications")
            stability.append({"distance": previous["config"]["distance"],
                              "comparison": "cached_run", "run_dir": str(Path(cache_run).resolve()),
                              "config": previous["config"], "stage": "pre_synthesis_all_implications",
                              "adjusted_rand_index": float(adjusted_rand_score(assignments, cached_assignments)) if implications else None})
        for distance in (config.distance - 0.05, config.distance + 0.05):
            if not 0 < distance <= 2:
                continue
            variant = asdict(config)
            variant["distance"] = distance
            # Hold every other setting fixed to isolate the distance threshold.
            other = []
            cluster_implications(implications, embedder, len(events), PhaseAConfig(**variant), vectors=vectors, audit=other)
            stability.append({"distance": distance, "comparison": "neighbour",
                              "config": variant, "stage": "pre_synthesis_all_implications",
                              "adjusted_rand_index": float(adjusted_rand_score(assignments, _partition(other, len(implications)))) if implications else None,
                              "assignments": _partition(other, len(implications))})
        metrics = _metrics(events, implications, weak, modules, memberships, vectors, clusters_audit,
                           verdicts, attachments, failed, discarded, stability)
        # Ranked, so consumers that take the first N get the most significant emerging issues.
        weak_signals = [{"id": m["id"], "signal_id": m["signal_id"], "statement": m["statement"],
                         "driver": m["driver"], "effect": m["effect"], "mechanism": m["mechanism"],
                         "domains": m["domains"], "signal": _event_key(m["event"]),
                         "proto_module": proto_of.get(m["id"]), **_emerging(m)}
                        for m in weak]
        weak_signals.sort(key=lambda w: (-w["emerging_score"], w["id"]))
        metrics["coverage"]["emerging_issues"] = sum(w["tier"] == "emerging" for w in weak_signals)
        metrics["coverage"]["proto_modules"] = len(protos)
        metrics["consolidation"] = {"checks": len(merges),
                                    "merged": sum(r["status"] == "merged" for r in merges)}
        result = {"modules": modules, "weak_signals": weak_signals, "proto_modules": protos,
                  "discarded": [d["signal"] for d in discarded], "failed_extractions": failed,
                  "metrics": metrics, "run_dir": str(folder.resolve())}
        _write(folder / "modules.json", modules)
        _write(folder / "memberships.json", [[m["id"] for m in members] for members in memberships])
        _write(folder / "weak_signals.json", weak_signals)
        _write(folder / "metrics.json", metrics)
        _write(folder / "result.json", result)
        metadata["status"] = "complete"
        _write(folder / "config.json", metadata)
        _print_metrics(metrics)
        print(f"  Run saved to {folder.resolve()}")
        return result
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        _write(folder / "config.json", metadata)
        raise
