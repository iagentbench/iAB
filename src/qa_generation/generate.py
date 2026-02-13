"""
Call the QA generator model with a QA packet and return validated QA candidates.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.qa_generation.bedrock_client import (
    invoke_bedrock,
    parse_and_validate_qa_generator_response,
)

logger = logging.getLogger(__name__)


def load_prompt(prompts_dir: Path, name: str = "qa_generator_system_prompt.txt") -> str:
    """Load prompt template from prompts_dir (e.g. prompts/qa_gen_prompts)."""
    path = prompts_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def generate_qa_candidates(
    packet: dict[str, Any],
    *,
    model_id: str,
    K: int = 3,
    prompts_dir: Path,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    region: Optional[str] = None,
    prompt_response_jsonl_path: Optional[Path] = None,
    bundle_index: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    For one QA packet, call the generator model and return a list of validated QA items.

    Uses system prompt from prompts_dir and fills COMMUNITY_CARDS_JSON, CONNECTORS_WITH_IDS, K.
    If prompt_response_jsonl_path is set, appends one JSON line (prompt + response) to the JSONL file.
    """
    template = load_prompt(prompts_dir, "qa_generator_system_prompt.txt")
    community_cards = packet["community_cards"]
    connectors_with_ids = packet["connectors_with_ids"]
    community_cards_json = json.dumps(community_cards, indent=2, ensure_ascii=False)
    connectors_json = json.dumps(connectors_with_ids, indent=2, ensure_ascii=False)

    prompt = template.replace("{COMMUNITY_CARDS_JSON}", community_cards_json)
    prompt = prompt.replace("{CONNECTORS_WITH_IDS}", connectors_json)
    prompt = prompt.replace("{K}", str(K))

    response = invoke_bedrock(
        prompt,
        model_id,
        system=None,
        max_tokens=max_tokens,
        temperature=temperature,
        region=region,
    )
    text = response.get("text", "")
    if not text:
        raise ValueError("Generator returned empty response")

    if prompt_response_jsonl_path is not None:
        log_obj: dict[str, Any] = {
            "step": "generation",
            "model_id": model_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": text,
        }
        if bundle_index is not None:
            log_obj["bundle_index"] = bundle_index
        if response.get("usage"):
            log_obj["usage"] = response["usage"]
        prompt_response_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_response_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

    qa_list = parse_and_validate_qa_generator_response(text)
    return qa_list
