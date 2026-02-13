"""
Build QA packets per bundle: load curated package + pruned entities, form bundles of 2–3 communities,
and produce community_cards + connectors_with_ids for each bundle.

Includes connector classification (action vs identity), Jaccard overlap on topic_signature
to avoid overly overlapping communities, and bundle eligibility (min connectors, min action-like).
"""

import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# Defaults for bundle quality (tunable without hardcoding elsewhere).
# Min connectors relaxed (1) so runs with few connectors per pair still yield packets.
# Jaccard filter on by default (0.8) to avoid extremely overlapping communities; use 1.0 to disable.
DEFAULT_MIN_CONNECTORS_PER_BUNDLE = 1
DEFAULT_MIN_ACTION_CONNECTORS_PER_BUNDLE = 0
DEFAULT_MAX_JACCARD_OVERLAP = 0.8  # max topic overlap allowed; 0.8 = moderate filter, 1.0 = no filter
REQUIRE_MULTIPLE_SUBJECTS_WHEN_MULTIPLE_CONNECTORS = True  # avoid hub: ≥2 connectors ⇒ ≥2 distinct subjects

# Substrings that suggest identity/alias relations (case-insensitive)
IDENTITY_RELATION_PATTERNS = frozenset({
    "also known as", "real name", "formerly", "alias", "identified as",
    "same as", "is the same", "birth name", "legal name", "aka ", " a.k.a.",
    " is ", " was ", "became ", "named ", "called ",
})
# Substrings that suggest action/causal/institutional relations
ACTION_RELATION_PATTERNS = frozenset({
    "appointed", "elected", "banned", "ordered", "announced", "sanctioned",
    "led to", "influenced", "criticized", "transitioned", "accused", "charged",
    "sued", "resigned", "replaced", "succeeded", "nominated", "approved",
    "signed", "passed", "enacted", "declared", "stated", "said that",
    "according to", "reported", "claimed", "denied", "supported", "opposed",
    "met with", "agreed", "refused", "decided", "ruled", "found ",
})


def _normalize_topic_signature(sig: Any) -> set[str]:
    """Turn topic_signature (list or iterable) into a set of lowercased tokens for Jaccard."""
    if sig is None:
        return set()
    if isinstance(sig, str):
        return {s.strip().lower() for s in re.split(r"\W+", sig) if s.strip()}
    try:
        return {str(x).strip().lower() for x in sig if x is not None and str(x).strip()}
    except TypeError:
        return set()


def jaccard_overlap(sig_a: set[str], sig_b: set[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. 0 = disjoint, 1 = identical."""
    if not sig_a and not sig_b:
        return 0.0
    union = sig_a | sig_b
    if not union:
        return 0.0
    inter = sig_a & sig_b
    return len(inter) / len(union)


def connector_relation_type(conn: dict) -> Literal["action", "identity", "other"]:
    """
    Classify connector by relation_text: action-like (causal/institutional), identity-like (alias), or other.
    Uses pattern sets so new patterns can be added without changing logic.
    """
    text = (conn.get("relation_text") or "").strip()
    if not text:
        return "other"
    lower = text.lower()
    # Prefer action over identity when both match (action is more useful for sensemaking)
    for pat in ACTION_RELATION_PATTERNS:
        if pat in lower:
            return "action"
    for pat in IDENTITY_RELATION_PATTERNS:
        if pat in lower:
            return "identity"
    return "other"


def load_curated_package(curated_artifacts_dir: Path) -> dict[str, Any]:
    """Load curated_llm_package.json from curated_artifacts_dir."""
    path = curated_artifacts_dir / "curated_llm_package.json"
    if not path.is_file():
        raise FileNotFoundError(f"Curated package not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_entities_pruned(curated_artifacts_dir: Path) -> pd.DataFrame:
    """Load entities_pruned.parquet; must have columns title, community."""
    path = curated_artifacts_dir / "entities_pruned.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"entities_pruned not found: {path}")
    df = pd.read_parquet(path)
    if "title" not in df.columns or "community" not in df.columns:
        raise ValueError("entities_pruned must have columns 'title' and 'community'")
    return df


def build_title_to_community(entities_df: pd.DataFrame) -> dict[str, Any]:
    """Map entity title -> community_id (int or str)."""
    out = {}
    for _, row in entities_df.iterrows():
        t = row.get("title")
        c = row.get("community")
        if t is not None and not (isinstance(t, float) and pd.isna(t)):
            out[str(t).strip()] = c
    return out


def connector_links_communities(
    conn: dict,
    title_to_community: dict[str, Any],
) -> tuple[Any, Any] | None:
    """
    If this connector links two different communities, return (comm_a, comm_b); else None.
    comm_a, comm_b are community ids (order arbitrary).
    """
    subj = (conn.get("subject") or "").strip()
    obj = (conn.get("object") or "").strip()
    ca = title_to_community.get(subj)
    cb = title_to_community.get(obj)
    if ca is None or cb is None or ca == cb:
        return None
    return (ca, cb)


def get_bundle_connectors(
    connector_relations: list[dict],
    title_to_community: dict[str, Any],
    bundle_community_ids: set,
) -> list[dict]:
    """
    Return connectors that link any pair of communities in bundle_community_ids.
    Each returned dict has index, subject, relation_text, object, weight, and is included in bundle.
    """
    result = []
    for c in connector_relations:
        pair = connector_links_communities(c, title_to_community)
        if pair is None:
            continue
        a, b = pair
        if a in bundle_community_ids and b in bundle_community_ids:
            result.append(c)
    return result


def build_community_cards_for_bundle(
    package: dict[str, Any],
    bundle_community_ids: list[int] | list[str],
) -> list[dict[str, Any]]:
    """
    Build COMMUNITY_CARDS for the bundle: each item has community_id, summary, top_findings
    with finding_id (0-based index). Only includes communities in bundle_community_ids.
    """
    communities = package.get("communities", [])
    cards = []
    for comm in communities:
        cid = comm.get("community_id")
        if cid not in bundle_community_ids:
            continue
        findings = comm.get("top_findings", [])
        if not isinstance(findings, list):
            findings = []
        findings_with_ids = []
        for i, f in enumerate(findings[:10]):
            if isinstance(f, dict):
                text = (f.get("summary") or "").strip() or str(f)
                snippet = (f.get("snippet") or "").strip()
                entry: dict[str, Any] = {"finding_id": i, "text": text}
                if snippet:
                    entry["snippet"] = snippet
                findings_with_ids.append(entry)
            else:
                findings_with_ids.append({"finding_id": i, "text": str(f).strip()})
        cards.append({
            "community_id": cid,
            "title": (comm.get("title") or "").strip(),
            "summary": (comm.get("summary") or "").strip(),
            "top_findings": findings_with_ids,
        })
    return cards


def build_connectors_with_ids(connector_list: list[dict], max_connectors: int = 12) -> list[dict]:
    """
    Return list of connector items with explicit connector_id (index) for prompt.
    Each item: connector_id, subject, relation_text, object, weight (optional).
    """
    out = []
    for i, c in enumerate(connector_list[:max_connectors]):
        out.append({
            "connector_id": c.get("index", i),
            "subject": (c.get("subject") or "").strip(),
            "relation_text": (c.get("relation_text") or "").strip()[:2000],
            "object": (c.get("object") or "").strip(),
            "weight": c.get("weight"),
        })
    return out


def _get_community_topic_signature(package: dict[str, Any], community_id: Any) -> set[str]:
    """Get topic_signature set for a community from the package."""
    for c in package.get("communities", []):
        if c.get("community_id") == community_id:
            return _normalize_topic_signature(c.get("topic_signature"))
    return set()


def enumerate_bundles(
    package: dict[str, Any],
    title_to_community: dict[str, Any],
    *,
    min_communities: int = 2,
    max_communities: int = 3,
    max_bundles: int = 5,
    min_connectors_per_bundle: int = DEFAULT_MIN_CONNECTORS_PER_BUNDLE,
    min_action_connectors_per_bundle: int = DEFAULT_MIN_ACTION_CONNECTORS_PER_BUNDLE,
    max_jaccard_overlap: float = DEFAULT_MAX_JACCARD_OVERLAP,
    require_multiple_subjects: bool = REQUIRE_MULTIPLE_SUBJECTS_WHEN_MULTIPLE_CONNECTORS,
) -> list[list[Any]]:
    """
    Enumerate bundles of 2–3 community IDs that meet global-sensemaking quality thresholds.

    Bundle selection: topic non-overlap via Jaccard(topic_signature_a, topic_signature_b) <= max_jaccard_overlap
    so the two communities represent meaningfully different themes. Bundles with lower Jaccard overlap
    (more diverse) are preferred when picking up to max_bundles.

    A bundle is included only if:
    - It has at least min_connectors_per_bundle connectors linking the pair,
    - At least min_action_connectors_per_bundle of them are action-like,
    - Jaccard(topic_signature_a, topic_signature_b) <= max_jaccard_overlap,
    - If require_multiple_subjects and len(connectors) >= 2: at least 2 distinct connector subjects.

    Returns list of bundles (each a list of community_id), ordered by ascending Jaccard so more diverse pairs come first.
    """
    connector_relations = package.get("connector_relations", [])
    communities = package.get("communities", [])
    comm_ids = [c.get("community_id") for c in communities if c.get("community_id") is not None]
    comm_ids = list(dict.fromkeys(comm_ids))

    # For each pair (a,b), collect connectors linking them and classify
    pair_connectors: dict[tuple[Any, Any], list[dict]] = {}
    for conn in connector_relations:
        pair = connector_links_communities(conn, title_to_community)
        if pair is None:
            continue
        a, b = pair
        if a not in comm_ids or b not in comm_ids or a == b:
            continue
        key = tuple(sorted([a, b], key=str))
        pair_connectors.setdefault(key, []).append(conn)

    # Collect candidates that pass filters; keep (jaccard, (a, b)) so we can sort by lower overlap first
    candidates: list[tuple[float, tuple[Any, Any]]] = []
    for (a, b) in pair_connectors.keys():
        conns = pair_connectors[(a, b)]
        if len(conns) < min_connectors_per_bundle:
            continue
        action_count = sum(1 for c in conns if connector_relation_type(c) == "action")
        if action_count < min_action_connectors_per_bundle:
            continue
        sig_a = _get_community_topic_signature(package, a)
        sig_b = _get_community_topic_signature(package, b)
        jacc = jaccard_overlap(sig_a, sig_b)
        if jacc > max_jaccard_overlap:
            continue
        if require_multiple_subjects and len(conns) >= 2:
            subjects = {str((c.get("subject") or "").strip()).lower() for c in conns}
            subjects.discard("")
            if len(subjects) < 2:
                continue
        candidates.append((jacc, (a, b)))

    # Prefer lower Jaccard overlap (more diverse communities) first
    candidates.sort(key=lambda x: (x[0], str(x[1])))
    bundles = [list(pair) for _, pair in candidates[:max_bundles]]
    return bundles


def build_qa_packets(
    curated_artifacts_dir: Path,
    *,
    K_CONNECTOR: int = 12,
    max_bundles: int = 3,
    min_connectors_per_bundle: int = DEFAULT_MIN_CONNECTORS_PER_BUNDLE,
    min_action_connectors_per_bundle: int = DEFAULT_MIN_ACTION_CONNECTORS_PER_BUNDLE,
    max_jaccard_overlap: float = DEFAULT_MAX_JACCARD_OVERLAP,
    require_multiple_subjects: bool = REQUIRE_MULTIPLE_SUBJECTS_WHEN_MULTIPLE_CONNECTORS,
) -> list[dict[str, Any]]:
    """
    Load curated package and entities, enumerate bundles (with quality filters), and build one QA packet per bundle.

    Each packet has:
      - community_cards: list of {community_id, title, summary, top_findings: [{finding_id, text, snippet?}, ...]}
      - connectors_with_ids: list of {connector_id, subject, relation_text, object, weight}
      - bundle_community_ids: list of community ids in this bundle

    Returns list of packets (one per bundle).
    """
    package = load_curated_package(curated_artifacts_dir)
    entities_df = load_entities_pruned(curated_artifacts_dir)
    title_to_community = build_title_to_community(entities_df)

    bundles = enumerate_bundles(
        package,
        title_to_community,
        min_communities=2,
        max_communities=3,
        max_bundles=max_bundles,
        min_connectors_per_bundle=min_connectors_per_bundle,
        min_action_connectors_per_bundle=min_action_connectors_per_bundle,
        max_jaccard_overlap=max_jaccard_overlap,
        require_multiple_subjects=require_multiple_subjects,
    )
    if not bundles:
        return []

    connector_relations = package.get("connector_relations", [])
    packets = []
    for bundle_ids in bundles:
        bundle_set = set(bundle_ids)
        conns = get_bundle_connectors(connector_relations, title_to_community, bundle_set)
        if len(conns) < min_connectors_per_bundle:
            continue
        community_cards = build_community_cards_for_bundle(package, bundle_ids)
        if len(community_cards) < 2:
            continue
        connectors_with_ids = build_connectors_with_ids(conns, max_connectors=K_CONNECTOR)
        packets.append({
            "bundle_community_ids": bundle_ids,
            "community_cards": community_cards,
            "connectors_with_ids": connectors_with_ids,
        })
    return packets
