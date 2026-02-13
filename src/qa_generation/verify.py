"""
Call an LLM judge ("verifier") model and return validated verdict objects.

This module is intentionally robust:
- verification runs per-candidate (one prompt per QA item)
- retries on invalid JSON / parse errors
- fail-closed fallback verdicts when a judge cannot be parsed after retries
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.qa_generation.bedrock_client import (
    invoke_bedrock,
    parse_and_validate_verifier_response,
)
from src.qa_generation.generate import load_prompt

logger = logging.getLogger(__name__)


def verify_qa_candidates(
    packet: dict[str, Any],
    qa_candidates: list[dict[str, Any]],
    *,
    model_id: str,
    prompts_dir: Path,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    region: Optional[str] = None,
    prompt_response_jsonl_path: Optional[Path] = None,
    bundle_index: Optional[int] = None,
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    """
    For one bundle's packet and its QA candidates, call the verifier model.

    Returns list of verdict objects. For the granular schema, verdict is computed in code
    from the *_flag fields by the parser.

    If prompt_response_jsonl_path is set, appends one JSON line (prompt + response) to the JSONL file.
    """
    template = load_prompt(prompts_dir, "globalness_verifier_system_prompt.txt")
    community_cards = packet["community_cards"]
    connectors_with_ids = packet["connectors_with_ids"]
    # Index packet evidence for ID->text lookup.
    # NOTE: We intentionally do NOT send community summaries or uncited findings/connectors
    # to the verifier. Each verification call is per-candidate with only cited evidence.
    findings_by_comm: dict[str, dict[str, str]] = {}
    for card in community_cards:
        comm_id = str(card.get("community_id"))
        comm_findings: dict[str, str] = {}
        for f in card.get("top_findings", []) or []:
            finding_id = str(f.get("finding_id"))
            text = (f.get("text") or "").strip()
            snippet = (f.get("snippet") or "").strip()
            comm_findings[finding_id] = (text + (f"\nSnippet: {snippet}" if snippet else "")).strip()
        findings_by_comm[comm_id] = comm_findings

    connectors_by_id: dict[str, str] = {}
    for c in connectors_with_ids:
        connector_id = str(c.get("connector_id"))
        subj = (c.get("subject") or "").strip()
        rel = (c.get("relation_text") or "").strip()
        obj = (c.get("object") or "").strip()
        connector_text = " — ".join([p for p in (subj, rel, obj) if p])
        connectors_by_id[connector_id] = connector_text

    verdicts: list[dict[str, Any]] = []
    for candidate_index, qa in enumerate(qa_candidates):
        # Merge cited IDs + evidence texts into a single, model-friendly QA payload.
        evidence_findings: list[dict[str, Any]] = []
        for sf in qa.get("supporting_findings", []) or []:
            comm_id_raw = sf.get("community_id")
            finding_id_raw = sf.get("finding_id")
            comm_id = str(comm_id_raw)
            finding_id = str(finding_id_raw)
            text = (findings_by_comm.get(comm_id, {}) or {}).get(finding_id, "")
            evidence_findings.append(
                {
                    "community_id": comm_id_raw,
                    "finding_id": finding_id_raw,
                    "text": text,
                }
            )

        evidence_connectors: list[dict[str, Any]] = []
        for sc in qa.get("supporting_connectors", []) or []:
            connector_id_raw = sc
            connector_id = str(connector_id_raw)
            evidence_connectors.append(
                {
                    "connector_id": connector_id_raw,
                    "text": connectors_by_id.get(connector_id, ""),
                }
            )

        qa_for_verifier: dict[str, Any] = {
            # Core QA
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            # Optional helpful metadata (does not add new evidence)
            "intent_pattern": qa.get("intent_pattern", ""),
            "why_multi_community": qa.get("why_multi_community") or qa.get("why_global") or "",
            # Evidence (only what the generator cited, resolved to text here)
            "evidence_findings": evidence_findings,
            "evidence_connectors": evidence_connectors,
        }
        qa_candidate_json = json.dumps(qa_for_verifier, indent=2, ensure_ascii=False)

        # Fill only the per-candidate placeholders. If legacy placeholders exist in the template,
        # replace them with empty structures to avoid leaking generator-level context.
        prompt = template
        prompt = prompt.replace("{QA_CANDIDATE_JSON}", qa_candidate_json)
        prompt = prompt.replace("{QA_JSON}", qa_candidate_json)  # backward compat if older prompt used QA_JSON for a single object
        prompt = prompt.replace("{COMMUNITY_CARDS_JSON}", "[]")
        prompt = prompt.replace("{CONNECTORS_WITH_IDS}", "[]")
        prompt = prompt.replace("{CITED_FINDINGS_JSON}", "[]")
        prompt = prompt.replace("{CITED_CONNECTORS_JSON}", "[]")

        last_error: Optional[str] = None
        parsed_item: Optional[dict[str, Any]] = None
        text = ""
        response: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            try:
                response = invoke_bedrock(
                    prompt,
                    model_id,
                    system=None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    region=region,
                )
                text = response.get("text", "") or ""
                if not text.strip():
                    raise ValueError("Judge returned empty response")

                parsed = parse_and_validate_verifier_response(text)
                if len(parsed) != 1:
                    raise ValueError(f"Judge returned {len(parsed)} item(s) for a single candidate")
                parsed_item = parsed[0]
                last_error = None
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "Verifier parse/invoke failed (model=%s bundle=%s candidate=%s attempt=%s/%s): %s",
                    model_id,
                    bundle_index,
                    candidate_index,
                    attempt,
                    max_attempts,
                    last_error,
                )
                continue

        if prompt_response_jsonl_path is not None:
            log_obj: dict[str, Any] = {
                "step": "verification",
                "model_id": model_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "response": text,
            }
            if bundle_index is not None:
                log_obj["bundle_index"] = bundle_index
            log_obj["candidate_index"] = candidate_index
            if response.get("usage"):
                log_obj["usage"] = response["usage"]
            if last_error:
                log_obj["error"] = last_error
            if parsed_item is not None:
                log_obj["parsed"] = parsed_item
            prompt_response_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_response_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        if parsed_item is None:
            # Fail-closed fallback verdict. Use legacy shape so the downstream pipeline can proceed.
            verdicts.append(
                {
                    "verdict": "FAIL",
                    "verification_reasons": [f"Judge error after {max_attempts} attempt(s): {last_error or 'unknown'}"],
                    "reasons": [f"Judge error after {max_attempts} attempt(s): {last_error or 'unknown'}"],
                }
            )
        else:
            verdicts.append(parsed_item)

    return verdicts
