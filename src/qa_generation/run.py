"""
Orchestrate QA generation: build packets, generate per bundle, verify with multiple LLM judges,
and write outputs to generated_artifacts/.

Outputs:
- qa_pairs_all.json: all generated QA candidates (PASS/FAIL) with judge audits
- qa_pairs_passed.json: only candidates that PASS all judges (fail-closed)
- llm_judge_<model>_response.jsonl: per-judge prompt/response logs
- generated_config.json + generate_config.yaml: reproducibility
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from src.qa_generation.generate import generate_qa_candidates
from src.qa_generation.packets import (
    DEFAULT_MAX_JACCARD_OVERLAP,
    DEFAULT_MIN_ACTION_CONNECTORS_PER_BUNDLE,
    DEFAULT_MIN_CONNECTORS_PER_BUNDLE,
    build_qa_packets,
)
from src.qa_generation.verify import verify_qa_candidates

logger = logging.getLogger(__name__)

PROGRESS_LOG_NAME = "qa_generation.log"
GENERATION_JSONL_NAME = "generation_prompt_response.jsonl"


def _safe_filename(s: str) -> str:
    """Sanitize model IDs for filesystem-safe filenames."""
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def _progress(log_path: Path, message: str) -> None:
    """Write a timestamped line to the progress log file and to the module logger."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {message}"
    logger.info(message)
    if log_path is not None and log_path.parent.exists():
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def run_qa_generation(
    run_dir: Path,
    *,
    project_root: Path,
    curated_subdir: str = "curated_artifacts",
    generated_subdir: str = "generated_artifacts",
    generation_model_id: str = "global.anthropic.claude-opus-4-5-20251101-v1:0",
    judge_model_ids: Optional[list[str]] = None,
    K_per_bundle: int = 3,
    max_bundles: int = 3,
    K_CONNECTOR: int = 12,
    target_pass_count: int = 5,
    min_connectors_per_bundle: int = DEFAULT_MIN_CONNECTORS_PER_BUNDLE,
    min_action_connectors_per_bundle: int = DEFAULT_MIN_ACTION_CONNECTORS_PER_BUNDLE,
    max_jaccard_overlap: float = DEFAULT_MAX_JACCARD_OVERLAP,
    region: Optional[str] = None,
    judge_max_attempts: int = 3,
    resume_if_completed: bool = True,
) -> dict[str, Any]:
    """
    Run full QA generation for one run directory.

    1. Read from run_dir/curated_artifacts/ (curated_llm_package.json, entities_pruned.parquet).
    2. Build QA packets (bundles of 2 communities + connectors).
    3. For each packet: generate K_per_bundle QA candidates, then verify with multiple judges.
       Candidate PASS is fail-closed (must pass all judges).
    4. Write qa_pairs_all.json, qa_pairs_passed.json, judge logs, and config to run_dir/generated_artifacts/.

    Returns config dict with paths and counts.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)
    curated_artifacts_dir = run_dir / curated_subdir
    generated_artifacts_dir = run_dir / generated_subdir

    if not curated_artifacts_dir.is_dir():
        raise FileNotFoundError(f"Curated artifacts directory not found: {curated_artifacts_dir}")

    prompts_dir = project_root / "prompts" / "qa_gen_prompts"
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    generated_artifacts_dir.mkdir(parents=True, exist_ok=True)

    if judge_model_ids is None:
        judge_model_ids = [
            "us.meta.llama4-scout-17b-instruct-v1:0",
            "google.gemma-3-12b-it",
            "openai.gpt-oss-20b-1:0",
        ]

    # Resume: if this run directory already has a successful completion marker, skip.
    qa_passed_path = generated_artifacts_dir / "qa_pairs_passed.json"
    config_json_path = generated_artifacts_dir / "generated_config.json"
    if resume_if_completed and qa_passed_path.is_file() and config_json_path.is_file():
        try:
            existing = json.loads(config_json_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and not existing.get("errors"):
                return existing
        except Exception:
            pass

    progress_log_path = generated_artifacts_dir / PROGRESS_LOG_NAME
    _progress(progress_log_path, "Starting QA generation.")

    packets = build_qa_packets(
        curated_artifacts_dir,
        K_CONNECTOR=K_CONNECTOR,
        max_bundles=max_bundles,
        min_connectors_per_bundle=min_connectors_per_bundle,
        min_action_connectors_per_bundle=min_action_connectors_per_bundle,
        max_jaccard_overlap=max_jaccard_overlap,
    )
    if not packets:
        _progress(
            progress_log_path,
            "No QA packets built. Effective filters: min_connectors=%s, min_action=%s, max_jaccard_overlap=%s. "
            "Relax with --min-connectors-per-bundle 1 --min-action-connectors-per-bundle 0 or raise --max-jaccard-overlap (e.g. 1.0)."
            % (min_connectors_per_bundle, min_action_connectors_per_bundle, max_jaccard_overlap),
        )
        logger.warning(
            "No QA packets built (filters: min_connectors=%s, min_action=%s, max_jaccard=%s)",
            min_connectors_per_bundle,
            min_action_connectors_per_bundle,
            max_jaccard_overlap,
        )
        out_all_path = generated_artifacts_dir / "qa_pairs_all.json"
        out_passed_path = generated_artifacts_dir / "qa_pairs_passed.json"
        config_path = generated_artifacts_dir / "generated_config.json"
        for p in (out_all_path, out_passed_path):
            with open(p, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
        config = {
            "run_dir": str(run_dir),
            "curated_artifacts_dir": str(curated_artifacts_dir),
            "generated_artifacts_dir": str(generated_artifacts_dir),
            "paths": {
                "qa_pairs_all": str(out_all_path),
                "qa_pairs_passed": str(out_passed_path),
                "generated_config": str(config_path),
                "progress_log": str(progress_log_path),
            },
            "packets_count": 0,
            "qa_pairs_all_count": 0,
            "qa_pairs_passed_count": 0,
            "generation_model_id": generation_model_id,
            "judge_model_ids": judge_model_ids,
            "min_connectors_per_bundle": min_connectors_per_bundle,
            "min_action_connectors_per_bundle": min_action_connectors_per_bundle,
            "max_jaccard_overlap": max_jaccard_overlap,
            "errors": [],
            "status": "completed",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        yaml_path = generated_artifacts_dir / "generate_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        return config

    _progress(progress_log_path, f"Built {len(packets)} packet(s).")

    gen_jsonl_path = generated_artifacts_dir / GENERATION_JSONL_NAME
    if (not gen_jsonl_path.exists()) or (not resume_if_completed):
        gen_jsonl_path.write_text("", encoding="utf-8")

    judge_jsonl_paths: dict[str, Path] = {}
    for mid in judge_model_ids:
        fname = f"llm_judge_{_safe_filename(mid)}_response.jsonl"
        p = generated_artifacts_dir / fname
        judge_jsonl_paths[mid] = p
        if (not p.exists()) or (not resume_if_completed):
            p.write_text("", encoding="utf-8")

    all_items: list[dict[str, Any]] = []
    passed_items: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, packet in enumerate(packets):
        if len(passed_items) >= target_pass_count:
            break
        _progress(progress_log_path, f"Bundle {i}: generating QA candidates...")
        try:
            qa_candidates = generate_qa_candidates(
                packet,
                model_id=generation_model_id,
                K=K_per_bundle,
                prompts_dir=prompts_dir,
                region=region,
                prompt_response_jsonl_path=gen_jsonl_path,
                bundle_index=i,
            )
            _progress(progress_log_path, f"Bundle {i}: generated {len(qa_candidates)} candidate(s).")
        except Exception as e:
            errors.append(f"Bundle {i} generation: {e}")
            _progress(progress_log_path, f"Bundle {i}: generation failed: {e}")
            logger.exception("Generation failed for bundle %s", i)
            continue

        _progress(progress_log_path, f"Bundle {i}: verifying...")
        try:
            verdicts_by_model: dict[str, list[dict[str, Any]]] = {}
            for mid in judge_model_ids:
                verdicts_by_model[mid] = verify_qa_candidates(
                    packet,
                    qa_candidates,
                    model_id=mid,
                    prompts_dir=prompts_dir,
                    region=region,
                    prompt_response_jsonl_path=judge_jsonl_paths[mid],
                    bundle_index=i,
                    max_attempts=judge_max_attempts,
                )

            # Candidate is PASS only if ALL judges PASS (fail-closed strict).
            per_candidate_pass: list[bool] = []
            for idx in range(len(qa_candidates)):
                all_pass = True
                for mid in judge_model_ids:
                    v = verdicts_by_model[mid][idx] if idx < len(verdicts_by_model[mid]) else {}
                    if (v.get("verdict") or "").upper() != "PASS":
                        all_pass = False
                        break
                per_candidate_pass.append(all_pass)

            pass_count = sum(1 for x in per_candidate_pass if x)
            _progress(progress_log_path, f"Bundle {i}: verification done ({pass_count} PASS by unanimity).")
        except Exception as e:
            errors.append(f"Bundle {i} verification: {e}")
            _progress(progress_log_path, f"Bundle {i}: verification failed: {e}")
            logger.exception("Verification failed for bundle %s", i)
            continue

        # Build lookup maps for post-processing expansions (ID -> text/context),
        # so downstream consumers can use enriched outputs without reloading the packet.
        community_context_by_id: dict[str, dict[str, Any]] = {}
        finding_by_id: dict[tuple[str, str], dict[str, Any]] = {}
        for card in packet.get("community_cards", []) or []:
            comm_id_raw = card.get("community_id")
            comm_id = str(comm_id_raw)
            community_context_by_id[comm_id] = {
                "community_id": comm_id_raw,
                "title": (card.get("title") or "").strip(),
                "summary": (card.get("summary") or "").strip(),
            }
            for f in card.get("top_findings", []) or []:
                fid_raw = f.get("finding_id")
                fid = str(fid_raw)
                entry: dict[str, Any] = {
                    "text": (f.get("text") or "").strip(),
                }
                snippet = (f.get("snippet") or "").strip()
                if snippet:
                    entry["snippet"] = snippet
                finding_by_id[(comm_id, fid)] = entry

        connector_by_id: dict[str, dict[str, Any]] = {}
        for c in packet.get("connectors_with_ids", []) or []:
            cid_raw = c.get("connector_id")
            connector_by_id[str(cid_raw)] = c

        for idx, qa in enumerate(qa_candidates):
            item = dict(qa)

            # Enrich citations for downstream tasks (post-processing only).
            supporting_findings_expanded: list[dict[str, Any]] = []
            cited_community_ids: set[str] = set()
            for sf in item.get("supporting_findings", []) or []:
                comm_id_raw = sf.get("community_id")
                finding_id_raw = sf.get("finding_id")
                comm_id = str(comm_id_raw)
                fid = str(finding_id_raw)
                cited_community_ids.add(comm_id)
                f = finding_by_id.get((comm_id, fid), {})
                expanded: dict[str, Any] = {
                    "community_id": comm_id_raw,
                    "finding_id": finding_id_raw,
                    "text": f.get("text", ""),
                }
                if "snippet" in f:
                    expanded["snippet"] = f["snippet"]
                supporting_findings_expanded.append(expanded)

            supporting_connectors_expanded: list[dict[str, Any]] = []
            for cid_raw in item.get("supporting_connectors", []) or []:
                c = connector_by_id.get(str(cid_raw), {}) or {}
                subj = (c.get("subject") or "").strip()
                rel = (c.get("relation_text") or "").strip()
                obj = (c.get("object") or "").strip()
                connector_text = " — ".join([p for p in (subj, rel, obj) if p])
                supporting_connectors_expanded.append(
                    {
                        "connector_id": cid_raw,
                        "subject": subj,
                        "relation_text": rel,
                        "object": obj,
                        "weight": c.get("weight"),
                        "text": connector_text,
                    }
                )

            # Minimal context: only communities actually cited by findings (plus required_communities if present).
            for rc in item.get("required_communities", []) or []:
                cited_community_ids.add(str(rc))
            item["community_context"] = {
                cid: community_context_by_id.get(cid, {"community_id": cid})
                for cid in sorted(cited_community_ids)
            }
            item["supporting_findings_expanded"] = supporting_findings_expanded
            item["supporting_connectors_expanded"] = supporting_connectors_expanded

            # Attach judge audits and determine final verdict (strict: any FAIL -> FAIL).
            judge_audits: dict[str, Any] = {}
            pass_votes = 0
            for mid in judge_model_ids:
                v = verdicts_by_model[mid][idx] if idx < len(verdicts_by_model[mid]) else {"verdict": "FAIL"}
                judge_audits[mid] = v
                if (v.get("verdict") or "").upper() == "PASS":
                    pass_votes += 1
            item["judge_audits"] = judge_audits
            item["vote_summary"] = {"pass_votes": pass_votes, "fail_votes": len(judge_model_ids) - pass_votes}
            final_verdict = "PASS" if pass_votes == len(judge_model_ids) else "FAIL"
            item["final_verdict"] = final_verdict

            all_items.append(item)
            if final_verdict == "PASS":
                passed_items.append(item)
                if len(passed_items) >= target_pass_count:
                    break

    out_all_path = generated_artifacts_dir / "qa_pairs_all.json"
    out_passed_path = generated_artifacts_dir / "qa_pairs_passed.json"
    with open(out_all_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
    with open(out_passed_path, "w", encoding="utf-8") as f:
        json.dump(passed_items, f, indent=2, ensure_ascii=False)

    _progress(
        progress_log_path,
        f"Wrote qa_pairs_all.json ({len(all_items)} item(s)); qa_pairs_passed.json ({len(passed_items)} PASS item(s)).",
    )
    _progress(progress_log_path, "QA generation finished.")

    config = {
        "run_dir": str(run_dir),
        "curated_artifacts_dir": str(curated_artifacts_dir),
        "generated_artifacts_dir": str(generated_artifacts_dir),
        "paths": {
            "qa_pairs_all": str(out_all_path),
            "qa_pairs_passed": str(out_passed_path),
            "generated_config_json": str(generated_artifacts_dir / "generated_config.json"),
            "generate_config_yaml": str(generated_artifacts_dir / "generate_config.yaml"),
            "progress_log": str(progress_log_path),
            "generation_prompt_response_jsonl": str(gen_jsonl_path),
            "judge_prompt_response_jsonl": {mid: str(p) for mid, p in judge_jsonl_paths.items()},
        },
        "packets_count": len(packets),
        "qa_pairs_all_count": len(all_items),
        "qa_pairs_passed_count": len(passed_items),
        "generation_model_id": generation_model_id,
        "judge_model_ids": judge_model_ids,
        "K_per_bundle": K_per_bundle,
        "max_bundles": max_bundles,
        "K_CONNECTOR": K_CONNECTOR,
        "target_pass_count": target_pass_count,
        "min_connectors_per_bundle": min_connectors_per_bundle,
        "min_action_connectors_per_bundle": min_action_connectors_per_bundle,
        "max_jaccard_overlap": max_jaccard_overlap,
        "judge_max_attempts": judge_max_attempts,
        "errors": errors,
        "status": "completed" if not errors else "completed_with_errors",
    }
    config_path = generated_artifacts_dir / "generated_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    yaml_path = generated_artifacts_dir / "generate_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    return config
