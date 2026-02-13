"""
Curate artifacts from an iAgentBench run directory: pruned graph + compact LLM package.

Produces:
- curated_artifacts/curated_llm_package.json  (indexable: meta, connector_relations, communities)
- curated_artifacts/entities_pruned.parquet
- curated_artifacts/relationships_pruned.parquet
- curated_artifacts/curated_config.json
"""

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx


def _sanitize(s):
    """One line, no newlines, for clean text output."""
    if s is None or (isinstance(s, float) and (pd.isna(s) or np.isnan(s))):
        return ""
    s = str(s).strip().replace("\n", " ").replace("\r", " ")
    return " ".join(s.split())


def _snippet_from_explanation(explanation: str, max_chars: int = 450, max_sentences: int = 2) -> str:
    """First max_sentences sentences or first max_chars of explanation, sanitized. Empty if no content."""
    if not explanation or (isinstance(explanation, float) and (pd.isna(explanation) or np.isnan(explanation))):
        return ""
    s = str(explanation).strip()
    if not s:
        return ""
    s = _sanitize(s)
    if len(s) <= max_chars:
        return s
    parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=max_sentences)
    taken = " ".join(parts[:max_sentences]).strip()
    if len(taken) <= max_chars:
        return taken
    return taken[:max_chars].rsplit(" ", 1)[0] if " " in taken[:max_chars] else taken[:max_chars]


def _relation_text_short(description: str, max_sentences: int = 2, max_chars: int = 280) -> str:
    """Keep relation_text to at most a couple of sentences for use as an edge label. Long essays are truncated."""
    if not description or (isinstance(description, float) and (pd.isna(description) or np.isnan(description))):
        return ""
    s = _sanitize(str(description).strip())
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    parts = re.split(r"(?<=[.!?])\s+", s, maxsplit=max_sentences)
    taken = " ".join(parts[:max_sentences]).strip()
    if len(taken) <= max_chars:
        return taken
    return taken[:max_chars].rsplit(" ", 1)[0] if " " in taken[:max_chars] else taken[:max_chars]


def _to_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, list):
        return x
    return list(x)


def _is_run_dir(path: Path) -> bool:
    """True if path looks like an iAgentBench run directory (has graphs/entities.parquet and community_details.json)."""
    path = Path(path)
    if not path.is_dir():
        return False
    return (path / "graphs" / "entities.parquet").exists() and (path / "community_details.json").exists()


def find_run_dirs(parent_dir: Path) -> list[Path]:
    """
    Find all run directories under parent_dir (recursive).
    A run directory has graphs/entities.parquet and community_details.json.
    Does not descend into a directory once it is identified as a run directory.
    Returns sorted list of run directories for deterministic order.
    """
    parent_dir = Path(parent_dir)
    if not parent_dir.is_dir():
        return []
    run_dirs = []
    stack = [parent_dir]
    while stack:
        d = Path(stack.pop())
        if not d.is_dir():
            continue
        if _is_run_dir(d):
            run_dirs.append(d.resolve())
            continue
        try:
            for child in sorted(d.iterdir(), key=lambda p: p.name):
                if child.is_dir():
                    stack.append(child)
        except OSError:
            continue
    return sorted(run_dirs, key=lambda p: str(p))


def load_run_data(run_dir: Path):
    """Load entities, relationships, communities, community_reports, community_details from run_dir."""
    run_dir = Path(run_dir)
    graphs_dir = run_dir / "graphs"
    entities_df = pd.read_parquet(graphs_dir / "entities.parquet")
    relationships_df = pd.read_parquet(graphs_dir / "relationships.parquet")
    communities_df = pd.read_parquet(graphs_dir / "communities.parquet")
    community_reports_df = pd.read_parquet(graphs_dir / "community_reports.parquet")
    with open(run_dir / "community_details.json", "r", encoding="utf-8") as f:
        comm_data = json.load(f)
    comm_list = comm_data.get("communities", []) if isinstance(comm_data, dict) else (comm_data if isinstance(comm_data, list) else [])
    comm_id_to_details = {}
    for c in comm_list:
        if isinstance(c, dict):
            cid = c.get("community_id", c.get("id", c.get("community")))
            if cid is not None:
                comm_id_to_details[cid] = c
    return entities_df, relationships_df, communities_df, community_reports_df, comm_id_to_details


def load_text_units_and_documents(run_dir: Path):
    """
    Load text_units.parquet and documents.parquet from run_dir/graphs/ when present.
    Returns (text_units_df, documents_df). Either may be None if file is missing.
    """
    run_dir = Path(run_dir)
    graphs_dir = run_dir / "graphs"
    text_units_df = None
    documents_df = None
    if (graphs_dir / "text_units.parquet").exists():
        text_units_df = pd.read_parquet(graphs_dir / "text_units.parquet")
    if (graphs_dir / "documents.parquet").exists():
        documents_df = pd.read_parquet(graphs_dir / "documents.parquet")
    return text_units_df, documents_df


def _normalize_doc_title_for_url_lookup(title: str) -> str:
    """Normalize document title (filename) for URL manifest lookup: strip extension, lowercase."""
    if not title or (isinstance(title, float) and pd.isna(title)):
        return ""
    s = str(title).strip()
    if "." in s:
        s = s.rsplit(".", 1)[0]
    return s.lower()


def _extract_url_from_html_content(content: str) -> str:
    """Extract canonical or og:url from HTML head. Returns first match or empty string."""
    # Only look in first 32KB to avoid parsing huge files
    head = content[:32768] if len(content) > 32768 else content
    # canonical: <link rel="canonical" href="https://...">
    m = re.search(r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']', head, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r'<link\s+href=["\']([^"\']+)["\']\s+rel=["\']canonical["\']', head, re.I)
    if m:
        return m.group(1).strip()
    # og:url: <meta property="og:url" content="https://...">
    m = re.search(r'<meta\s+property=["\']og:url["\']\s+content=["\']([^"\']+)["\']', head, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:url["\']', head, re.I)
    if m:
        return m.group(1).strip()
    return ""


def _input_dir_from_run_dir(run_dir: Path) -> Path | None:
    """Infer input directory from run directory. run_dir = project_root/output/keyword/timestamp."""
    run_dir = Path(run_dir)
    if len(run_dir.parts) < 3:
        return None
    keyword = run_dir.parent.name
    project_root = run_dir.parent.parent.parent
    return project_root / "input" / "keywords" / keyword


def extract_urls_from_html(input_dir: Path) -> dict:
    """
    Scan HTML files in input_dir and extract canonical or og:url from each.
    Returns dict mapping normalized document title (filename without extension) -> url.
    Uses the URL embedded in the page (canonical/og:url), which is the correct source link.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return {}
    title_to_url = {}
    for html_path in input_dir.glob("*.html"):
        try:
            content = html_path.read_text(encoding="utf-8", errors="replace")
            url = _extract_url_from_html_content(content)
            if url:
                key = _normalize_doc_title_for_url_lookup(html_path.name)
                if key:
                    title_to_url[key] = url
        except Exception:
            continue
    return title_to_url


def load_url_manifest_from_dir(input_dir: Path) -> dict:
    """
    Load url_manifest.json from the given input directory, if it exists.
    Returns dict mapping normalized document title (filename without extension) -> url (str).
    """
    input_dir = Path(input_dir)
    manifest_path = input_dir / "url_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    entries = data.get("entries", data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if not isinstance(entries, list):
        return {}
    title_to_url = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        filename = item.get("filename", item.get("file", ""))
        if url and filename:
            key = _normalize_doc_title_for_url_lookup(filename)
            if key:
                title_to_url[key] = str(url).strip()
    return title_to_url


def load_url_manifest(run_dir: Path) -> dict:
    """
    Load url_manifest.json from the input directory for this run, if it exists.
    run_dir is e.g. project_root/output/New Pope chosen/20260116_164015.
    Input dir is project_root/input/keywords/{keyword}/.
    Returns dict mapping normalized document title (filename without extension) -> url (str).
    """
    run_dir = Path(run_dir)
    if len(run_dir.parts) < 3:
        return {}
    keyword = run_dir.parent.name
    project_root = run_dir.parent.parent.parent
    input_dir = project_root / "input" / "keywords" / keyword
    return load_url_manifest_from_dir(input_dir)


def resolve_input_dir_for_run(input_root: Path, run_dir: Path) -> Path | None:
    """
    Resolve the input directory for a run when using a batch-style input root.
    input_root is e.g. input/2025_seeds (then we use input_root/keywords/{keyword})
    or input/2025_seeds/keywords (then we use input_root/{keyword}).
    keyword is taken from run_dir.parent.name (e.g. output/batch/query/timestamp -> query).
    Returns the path to use for URL resolution, or None if neither candidate exists.
    """
    input_root = Path(input_root)
    run_dir = Path(run_dir)
    keyword = run_dir.parent.name
    # Batch layout: input_root/keywords/query
    candidate = input_root / "keywords" / keyword
    if candidate.is_dir():
        return candidate
    # Flat layout: input_root/query
    candidate = input_root / keyword
    if candidate.is_dir():
        return candidate
    # Return candidate anyway so we try URL resolution (manifest/HTML may not exist)
    return input_root / "keywords" / keyword


def build_doc_id_to_url(documents_df: pd.DataFrame | None, title_to_url: dict) -> dict:
    """Build doc_id -> url map from documents DataFrame and title_to_url (from url_manifest)."""
    if documents_df is None or len(documents_df) == 0 or not title_to_url:
        return {}
    out = {}
    for _, row in documents_df.iterrows():
        doc_id = row.get("id")
        if doc_id is None or (isinstance(doc_id, float) and pd.isna(doc_id)):
            continue
        title = row.get("title", "")
        key = _normalize_doc_title_for_url_lookup(title)
        url = title_to_url.get(key, "")
        out[str(doc_id)] = url
    return out


def compute_cohesion(community_id, communities_df, relationships_df, entities_df):
    """Internal edge density for a community."""
    comm_row = communities_df[communities_df["community"] == community_id]
    if len(comm_row) == 0:
        return 0.0
    entity_ids = _to_list(comm_row.iloc[0].get("entity_ids", []))
    if not entity_ids or len(entity_ids) < 2:
        return 0.0
    comm_entities = entities_df[entities_df["id"].isin(entity_ids)]
    entity_titles = set(comm_entities["title"].unique())
    internal = relationships_df[
        (relationships_df["source"].isin(entity_titles)) & (relationships_df["target"].isin(entity_titles))
    ]
    n = len(entity_titles)
    if n < 2:
        return 0.0
    possible = n * (n - 1)
    return len(internal) / possible if possible > 0 else 0.0


def extract_topic_signature(community_id, comm_id_to_details, entities_df, communities_df):
    """Extract topic signature keywords from title/summary and top entities."""
    keywords = []
    if community_id in comm_id_to_details:
        details = comm_id_to_details[community_id]
        text = (details.get("title", "") + " " + details.get("summary", "")).lower()
        words = re.findall(r"\b[a-z]{4,}\b", text)
        stopwords = {"the", "and", "for", "are", "was", "has", "had", "his", "her", "its", "our", "their", "this", "that", "with", "from", "into", "about", "during"}
        keywords.extend([w for w in words if w not in stopwords])
    comm_row = communities_df[communities_df["community"] == community_id]
    if len(comm_row) > 0:
        entity_ids = _to_list(comm_row.iloc[0].get("entity_ids", []))
        if entity_ids:
            comm_entities = entities_df[entities_df["id"].isin(entity_ids)]
            for t in comm_entities["title"].head(5).tolist():
                if isinstance(t, str):
                    keywords.append(t.lower())
    cnt = Counter(keywords)
    return [w for w, _ in cnt.most_common(10)]


def build_community_nodes(communities_df, community_reports_df, comm_id_to_details, entities_df, relationships_df):
    """Build community meta-graph nodes with size, rank, cohesion, topic_signature, evidence_mass."""
    nodes = []
    for _, comm_row in communities_df.iterrows():
        cid = comm_row.get("community", comm_row.get("human_readable_id"))
        if pd.isna(cid):
            continue
        entity_ids = _to_list(comm_row.get("entity_ids", []))
        size = comm_row.get("size", len(entity_ids))
        rank = None
        creport = community_reports_df[community_reports_df["community"] == cid]
        if len(creport) > 0:
            rank = creport.iloc[0].get("rank")
        if rank is None and cid in comm_id_to_details:
            rank = comm_id_to_details[cid].get("rank")
        cohesion = compute_cohesion(cid, communities_df, relationships_df, entities_df)
        topic_signature = extract_topic_signature(cid, comm_id_to_details, entities_df, communities_df)
        text_unit_ids = _to_list(comm_row.get("text_unit_ids", []))
        evidence_mass = len(set(text_unit_ids)) if text_unit_ids else 0
        title = comm_id_to_details.get(cid, {}).get("title", f"Community {cid}")
        nodes.append({
            "id": cid,
            "size": size,
            "rank": rank if rank is not None else 0.0,
            "cohesion": cohesion,
            "topic_signature": topic_signature,
            "evidence_mass": evidence_mass,
            "entity_ids": comm_row.get("entity_ids", []),
            "text_unit_ids": comm_row.get("text_unit_ids", []),
            "title": title,
        })
    return pd.DataFrame(nodes)


def build_aggregated_edges(communities_df, entities_df, relationships_df):
    """Cross-community edges aggregated by (source_comm, target_comm)."""
    entity_to_community = {}
    for _, comm_row in communities_df.iterrows():
        cid = comm_row.get("community", comm_row.get("human_readable_id"))
        eids = _to_list(comm_row.get("entity_ids", []))
        for eid in eids:
            entity_to_community[eid] = cid
    entity_title_to_comm = {}
    for _, row in entities_df.iterrows():
        eid = row["id"]
        if eid in entity_to_community:
            entity_title_to_comm[row["title"]] = entity_to_community[eid]
    edges = []
    for _, rel in relationships_df.iterrows():
        sc = entity_title_to_comm.get(rel["source"])
        tc = entity_title_to_comm.get(rel["target"])
        if sc is None or tc is None or sc == tc:
            continue
        tuids = _to_list(rel.get("text_unit_ids", []))
        evidence_count = len(set(tuids)) if tuids else 1
        w = rel.get("weight", 1.0)
        edges.append({"source": sc, "target": tc, "evidence_count": evidence_count, "edge_weight": evidence_count * w})
    if not edges:
        return pd.DataFrame(columns=["source", "target", "edge_weight", "evidence_count", "shared_evidence_mass"])
    agg = pd.DataFrame(edges).groupby(["source", "target"]).agg({"edge_weight": "sum", "evidence_count": "sum"}).reset_index()
    agg["shared_evidence_mass"] = agg["evidence_count"]
    return agg


def compute_influence_and_categorize(community_nodes_df, aggregated_edges):
    """Add centrality, influence_score, and category (Core/Bridge/Satellite/Other)."""
    G = nx.DiGraph()
    for _, row in community_nodes_df.iterrows():
        G.add_node(row["id"], size=row["size"], rank=row["rank"], title=row["title"], evidence_mass=row["evidence_mass"])
    if len(aggregated_edges) > 0:
        for _, row in aggregated_edges.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["edge_weight"])
    pagerank = nx.pagerank(G, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}
    betweenness = nx.betweenness_centrality(G, weight="weight")
    community_nodes_df = community_nodes_df.copy()
    community_nodes_df["pagerank"] = community_nodes_df["id"].map(pagerank).fillna(0.0)
    community_nodes_df["eigenvector"] = community_nodes_df["id"].map(eigenvector).fillna(0.0)
    community_nodes_df["betweenness"] = community_nodes_df["id"].map(betweenness).fillna(0.0)
    community_nodes_df["log_size"] = np.log1p(community_nodes_df["size"])
    community_nodes_df["centrality"] = community_nodes_df["pagerank"]

    def zscore(vals):
        a = np.array(vals)
        m, s = np.nanmean(a), np.nanstd(a)
        return np.nan_to_num((a - m) / s if s > 0 else np.zeros_like(a), nan=0.0)

    alpha = beta = gamma = delta = 0.25
    community_nodes_df["influence_score"] = (
        alpha * zscore(community_nodes_df["log_size"])
        + beta * zscore(community_nodes_df["centrality"])
        + gamma * zscore(community_nodes_df["betweenness"])
        + delta * zscore(community_nodes_df["evidence_mass"])
    )
    n_comm = len(community_nodes_df)
    k_core = max(5, int(0.3 * n_comm))
    b_bridge = max(3, int(0.2 * n_comm))
    core = set(community_nodes_df.nlargest(k_core, "influence_score")["id"].tolist())
    bridge = set(community_nodes_df.nlargest(b_bridge, "betweenness")["id"].tolist())
    core_bridge = core | bridge
    satellite = set()
    if len(aggregated_edges) > 0:
        thresh = aggregated_edges["edge_weight"].quantile(0.5)
        for _, row in aggregated_edges.iterrows():
            s, t, w = row["source"], row["target"], row["edge_weight"]
            if s in core_bridge and w >= thresh and t not in core_bridge:
                satellite.add(t)
            if t in core_bridge and w >= thresh and s not in core_bridge:
                satellite.add(s)

    def cat(cid):
        if cid in core:
            return "Core"
        if cid in bridge:
            return "Bridge"
        if cid in satellite:
            return "Satellite"
        return "Other"

    community_nodes_df["category"] = community_nodes_df["id"].apply(cat)
    return community_nodes_df


def prune_and_mark_connectors(entities_df, relationships_df, communities_df, community_nodes_df):
    """Prune to Core/Bridge/Satellite only; mark is_connector on relationships."""
    selected = set(community_nodes_df[community_nodes_df["category"].isin(["Core", "Bridge", "Satellite"])]["id"].tolist())
    entity_to_community = {}
    for _, comm_row in communities_df.iterrows():
        cid = comm_row.get("community", comm_row.get("human_readable_id"))
        for eid in _to_list(comm_row.get("entity_ids", [])):
            entity_to_community[eid] = cid
    if "community" not in entities_df.columns:
        entities_df = entities_df.copy()
        entities_df["community"] = entities_df["id"].map(entity_to_community)
    cleaned_entities = entities_df[entities_df["community"].isin(selected)].copy()
    comm_id_to_cat = community_nodes_df.set_index("id")["category"]
    cleaned_entities["community_category"] = cleaned_entities["community"].map(comm_id_to_cat)
    kept_titles = set(cleaned_entities["title"])
    cleaned_relationships = relationships_df[
        relationships_df["source"].isin(kept_titles) & relationships_df["target"].isin(kept_titles)
    ].copy()
    title_to_community = cleaned_entities.set_index("title")["community"]

    def is_connector(r):
        cs = title_to_community.get(r["source"])
        ct = title_to_community.get(r["target"])
        return cs is not None and ct is not None and cs != ct

    cleaned_relationships["is_connector"] = cleaned_relationships.apply(is_connector, axis=1)
    connector_count = int(cleaned_relationships["is_connector"].sum())
    return cleaned_entities, cleaned_relationships, connector_count


def _json_float(x):
    """JSON-serializable float; None for NaN/None."""
    if x is None:
        return None
    try:
        f = float(x)
        return None if (f != f) else f  # NaN != NaN
    except (TypeError, ValueError):
        return None


def _evidence_ids_to_list(eids):
    """Convert text_unit_ids / evidence to a JSON-serializable list."""
    if eids is None:
        return []
    if hasattr(eids, "tolist"):
        return eids.tolist()
    if isinstance(eids, list):
        return [str(x) for x in eids]
    return list(eids)


def _doc_id_to_title_map(documents_df: pd.DataFrame | None):
    """Build document id -> title (readable filename) map from documents DataFrame."""
    if documents_df is None or len(documents_df) == 0:
        return {}
    out = {}
    for _, row in documents_df.iterrows():
        doc_id = row.get("id")
        if doc_id is not None and (not isinstance(doc_id, float) or not pd.isna(doc_id)):
            title = row.get("title", "")
            out[str(doc_id)] = _sanitize(title) if title else str(doc_id)
    return out


def _resolve_evidence_excerpts(
    text_unit_ids: list,
    text_units_df: pd.DataFrame | None,
    doc_id_to_title: dict,
    doc_id_to_url: dict | None = None,
) -> tuple[list[dict], set]:
    """
    Resolve a list of text_unit ids to full excerpts with document traceability.
    Returns (list of { text_unit_id, excerpt, document_ids, document_titles, document_urls, n_tokens }, set of document ids seen).
    """
    if not text_unit_ids or text_units_df is None or len(text_units_df) == 0:
        return [], set()
    doc_id_to_url = doc_id_to_url or {}
    eids = _evidence_ids_to_list(text_unit_ids)
    id_col = "id"
    text_col = "text"
    doc_ids_col = "document_ids"
    n_tokens_col = "n_tokens"
    if id_col not in text_units_df.columns or text_col not in text_units_df.columns:
        return [], set()
    tu_map = {}
    for _, row in text_units_df.iterrows():
        uid = row.get(id_col)
        if uid is not None:
            tu_map[str(uid)] = row
    result = []
    doc_ids_seen = set()
    for uid in eids:
        row = tu_map.get(str(uid))
        if row is None:
            result.append({
                "text_unit_id": uid,
                "excerpt": "",
                "document_ids": [],
                "document_titles": [],
                "document_urls": [],
                "n_tokens": None,
            })
            continue
        excerpt = row.get(text_col)
        if excerpt is None or (isinstance(excerpt, float) and pd.isna(excerpt)):
            excerpt = ""
        else:
            excerpt = str(excerpt).strip()
        doc_ids = _to_list(row.get(doc_ids_col))
        doc_ids = [str(x) for x in doc_ids if x is not None and not (isinstance(x, float) and pd.isna(x))]
        doc_ids_seen.update(doc_ids)
        document_titles = [doc_id_to_title.get(d, d) for d in doc_ids]
        document_urls = [doc_id_to_url.get(d, "") for d in doc_ids]
        n_tokens = row.get(n_tokens_col)
        if n_tokens is not None and isinstance(n_tokens, (int, float)) and not (isinstance(n_tokens, float) and np.isnan(n_tokens)):
            n_tokens = int(n_tokens)
        else:
            n_tokens = None
        result.append({
            "text_unit_id": uid,
            "excerpt": excerpt,
            "document_ids": doc_ids,
            "document_titles": document_titles,
            "document_urls": document_urls,
            "n_tokens": n_tokens,
        })
    return result, doc_ids_seen


def build_curated_json(
    selected_communities_df,
    connector_relations_df,
    cleaned_entities_df,
    comm_by_id,
    *,
    K_CONNECTOR=12,
    M_FINDINGS=5,
    INCLUDE_COMMUNITIES_WITHOUT_FINDINGS=False,
    text_units_df: pd.DataFrame | None = None,
    documents_df: pd.DataFrame | None = None,
    doc_id_to_url: dict | None = None,
):
    """
    Build curated LLM package as a JSON-serializable dict for indexable access.

    Structure:
      meta: description, counts, K_CONNECTOR, M_FINDINGS, connector_total, connector_shown
      source_documents: list of { id, title, url } for all source docs (url = website when available)
      connector_relations: list of { index, subject, relation_text, object, weight, evidence_ids, evidence_excerpts }
      communities: list of { ..., evidence_text_unit_ids, evidence_excerpts }
    """
    doc_id_to_title = _doc_id_to_title_map(documents_df)
    doc_id_to_url = doc_id_to_url or {}
    all_doc_ids_seen = set()

    total_connector = len(connector_relations_df)
    k_actual = min(K_CONNECTOR, max(5, total_connector))
    top_connector = (
        connector_relations_df.nlargest(k_actual, "weight")
        if "weight" in connector_relations_df.columns
        else connector_relations_df.head(k_actual)
    )

    connector_relations = []
    for idx, (_, row) in enumerate(top_connector.iterrows()):
        subj = _sanitize(row.get("source", ""))
        raw_pred = _sanitize(row.get("description", "related_to"))
        pred = _relation_text_short(raw_pred, max_sentences=2, max_chars=280)
        if not pred:
            pred = raw_pred[:280].rsplit(" ", 1)[0] if len(raw_pred) > 280 else raw_pred
        obj = _sanitize(row.get("target", ""))
        if not subj or not obj:
            continue
        w = row.get("weight")
        if w is not None and (isinstance(w, float) and (pd.isna(w) or np.isnan(w))):
            w = None
        eids = _evidence_ids_to_list(row.get("text_unit_ids"))
        evidence_excerpts, doc_ids = _resolve_evidence_excerpts(
            eids, text_units_df, doc_id_to_title, doc_id_to_url
        )
        all_doc_ids_seen.update(doc_ids)
        connector_relations.append({
            "index": idx,
            "subject": subj,
            "relation_text": pred,
            "object": obj,
            "weight": _json_float(w),
            "evidence_ids": eids,
            "evidence_excerpts": evidence_excerpts,
        })

    communities = []
    for idx, (_, comm_row) in enumerate(selected_communities_df.iterrows()):
        cid = comm_row.get("id")
        title = comm_row.get("title", "")
        category = comm_row.get("category", "")
        influence = comm_row.get("influence_score", "")
        topic_sig = comm_row.get("topic_signature", [])
        if not isinstance(topic_sig, list):
            topic_sig = list(topic_sig) if hasattr(topic_sig, "__iter__") else []
        topic_signature = topic_sig[:12]
        details = comm_by_id.get(cid, {})
        summary = _sanitize(details.get("summary", ""))
        findings = details.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        top_findings = []
        for fnd in findings[:M_FINDINGS]:
            if isinstance(fnd, dict):
                summary = _sanitize(fnd.get("summary", ""))
                explanation = fnd.get("explanation", "")
                snippet = _snippet_from_explanation(explanation) if explanation else ""
                top_findings.append({"summary": summary or str(fnd), "snippet": snippet})
            else:
                top_findings.append({"summary": _sanitize(str(fnd)), "snippet": ""})
        if not top_findings and not INCLUDE_COMMUNITIES_WITHOUT_FINDINGS:
            continue
        comm_text_unit_ids = _to_list(comm_row.get("text_unit_ids", []))
        comm_text_unit_ids = _evidence_ids_to_list(comm_text_unit_ids)
        comm_evidence_excerpts, comm_doc_ids = _resolve_evidence_excerpts(
            comm_text_unit_ids, text_units_df, doc_id_to_title, doc_id_to_url
        )
        all_doc_ids_seen.update(comm_doc_ids)
        communities.append({
            "index": idx,
            "community_id": cid,
            "type": category,
            "influence_score": _json_float(influence),
            "title": _sanitize(title),
            "summary": summary,
            "top_findings": top_findings,
            "topic_signature": topic_signature,
            "evidence_text_unit_ids": comm_text_unit_ids,
            "evidence_excerpts": comm_evidence_excerpts,
        })

    source_documents = []
    for doc_id in sorted(all_doc_ids_seen):
        title = doc_id_to_title.get(doc_id, doc_id)
        url = doc_id_to_url.get(doc_id, "") if doc_id_to_url else ""
        source_documents.append({"id": doc_id, "title": title, "url": url})

    return {
        "meta": {
            "description": "CURATED KNOWLEDGE PACKAGE (for LLM consumption)",
            "communities_count": len(communities),
            "entities_count": len(cleaned_entities_df),
            "connector_total": total_connector,
            "connector_shown": len(connector_relations),
            "K_CONNECTOR": K_CONNECTOR,
            "M_FINDINGS": M_FINDINGS,
            "includes_evidence_excerpts": text_units_df is not None and len(text_units_df) > 0,
            "source_documents_count": len(source_documents),
        },
        "source_documents": source_documents,
        "connector_relations": connector_relations,
        "communities": communities,
    }


def run_curate(
    run_dir: Path,
    out_subdir: str = "curated_artifacts",
    K_CONNECTOR: int = 12,
    M_FINDINGS: int = 5,
    INCLUDE_COMMUNITIES_WITHOUT_FINDINGS: bool = False,
    input_dir: Path | None = None,
) -> dict:
    """
    Load run data, compute influence/categorization, prune graph, build curated package and config.
    Writes curated_artifacts/curated_llm_package.json, entities_pruned.parquet, relationships_pruned.parquet, curated_config.json.
    When input_dir is provided (e.g. for batch runs), URLs are resolved from that directory (url_manifest + HTML).
    Returns config dict (paths, counts, K_CONNECTOR, M_FINDINGS, etc.).
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    entities_df, relationships_df, communities_df, community_reports_df, comm_id_to_details = load_run_data(run_dir)
    community_nodes_df = build_community_nodes(
        communities_df, community_reports_df, comm_id_to_details, entities_df, relationships_df
    )
    aggregated_edges = build_aggregated_edges(communities_df, entities_df, relationships_df)
    community_nodes_df = compute_influence_and_categorize(community_nodes_df, aggregated_edges)
    cleaned_entities_df, cleaned_relationships_df, connector_count = prune_and_mark_connectors(
        entities_df, relationships_df, communities_df, community_nodes_df
    )
    selected_communities_df = community_nodes_df[
        community_nodes_df["category"].isin(["Core", "Bridge", "Satellite"])
    ]
    connector_relations_df = cleaned_relationships_df[cleaned_relationships_df["is_connector"]].copy()

    with open(run_dir / "community_details.json", "r", encoding="utf-8") as f:
        comm_details_raw = json.load(f)
    comm_list = comm_details_raw.get("communities", []) if isinstance(comm_details_raw, dict) else (comm_details_raw if isinstance(comm_details_raw, list) else [])
    comm_by_id = {c.get("id", c.get("community_id")): c for c in comm_list if isinstance(c, dict)}

    text_units_df, documents_df = load_text_units_and_documents(run_dir)
    # Resolve URLs: use explicit input_dir when provided (e.g. batch), else infer from run_dir
    if input_dir is not None:
        input_dir = Path(input_dir)
        title_to_url = load_url_manifest_from_dir(input_dir)
        html_urls = extract_urls_from_html(input_dir)
        title_to_url.update(html_urls)
    else:
        title_to_url = load_url_manifest(run_dir)
        _inferred_input = _input_dir_from_run_dir(run_dir)
        if _inferred_input:
            html_urls = extract_urls_from_html(_inferred_input)
            title_to_url.update(html_urls)
    doc_id_to_url = build_doc_id_to_url(documents_df, title_to_url)

    curated_data = build_curated_json(
        selected_communities_df,
        connector_relations_df,
        cleaned_entities_df,
        comm_by_id,
        K_CONNECTOR=K_CONNECTOR,
        M_FINDINGS=M_FINDINGS,
        INCLUDE_COMMUNITIES_WITHOUT_FINDINGS=INCLUDE_COMMUNITIES_WITHOUT_FINDINGS,
        text_units_df=text_units_df,
        documents_df=documents_df,
        doc_id_to_url=doc_id_to_url,
    )

    # Write outputs
    pkg_path = out_dir / "curated_llm_package.json"
    with open(pkg_path, "w", encoding="utf-8") as f:
        json.dump(curated_data, f, indent=2, ensure_ascii=False)
    entities_pruned_path = out_dir / "entities_pruned.parquet"
    relationships_pruned_path = out_dir / "relationships_pruned.parquet"
    cleaned_entities_df.to_parquet(entities_pruned_path, index=False)
    cleaned_relationships_df.to_parquet(relationships_pruned_path, index=False)

    config_path = out_dir / "curated_config.json"
    config = {
        "run_dir": str(run_dir),
        "curated_artifacts_dir": str(out_dir),
        "paths": {
            "curated_llm_package": str(pkg_path),
            "entities_pruned": str(entities_pruned_path),
            "relationships_pruned": str(relationships_pruned_path),
            "curated_config": str(config_path),
        },
        "K_CONNECTOR": K_CONNECTOR,
        "M_FINDINGS": M_FINDINGS,
        "INCLUDE_COMMUNITIES_WITHOUT_FINDINGS": INCLUDE_COMMUNITIES_WITHOUT_FINDINGS,
        "connector_count": int(connector_count),
        "connector_shown": min(K_CONNECTOR, max(5, len(connector_relations_df))),
        "includes_evidence_excerpts": text_units_df is not None and len(text_units_df) > 0,
        "entities_original": len(entities_df),
        "entities_pruned": len(cleaned_entities_df),
        "relationships_original": len(relationships_df),
        "relationships_pruned": len(cleaned_relationships_df),
        "communities_selected": len(selected_communities_df),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config
