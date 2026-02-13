"""
Bedrock client for QA generation: invoke with retries, throttling handling, and JSON validation.

Supports Claude (Anthropic) and Llama (Meta) via bedrock-runtime invoke_model.
"""

import json
import logging
import re
import time
from typing import Any, Optional, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

AWS_REGION = "us-east-1"
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 2.0
THROTTLE_CODES = ("ThrottlingException", "ServiceQuotaExceededException", "ModelNotReadyException")


def _get_bedrock_client(region: Optional[str] = None):
    region_name = region or AWS_REGION
    return boto3.client("bedrock-runtime", region_name=region_name)


def _get_model_family(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "claude" in model_id_lower or "anthropic" in model_id_lower:
        return "claude"
    if "llama" in model_id_lower or "meta" in model_id_lower:
        return "llama"
    if "gemma" in model_id_lower or "google" in model_id_lower:
        return "gemma"
    return "claude"


def _format_claude_body(
    message: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": message}]}],
    }
    if system:
        body["system"] = [{"type": "text", "text": system}]
    return json.dumps(body)


def _format_llama_body(
    message: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    # Llama 3 instruction format per AWS Bedrock docs (user/assistant only)
    user_content = f"{system}\n\n{message}" if system else message
    prompt = (
        "<<|begin_of_text|>><<|start_header_id|>>user<<|end_header_id|>>\n\n"
        f"{user_content}<<|eot_id|>>\n"
        "<<|start_header_id|>>assistant<<|end_header_id|>>\n\n"
    )
    body = {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
    }
    return json.dumps(body)


def _format_gemma_body(
    message: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> str:
    content = message
    if system:
        content = f"{system}\n\n{message}"
    body = {
        "messages": [{"role": "user", "content": content}],
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    return json.dumps(body)


def _parse_claude_response(response_body: bytes) -> dict[str, Any]:
    return json.loads(response_body)


def _parse_llama_response(response_body: bytes) -> dict[str, Any]:
    parsed = json.loads(response_body)
    text = parsed.get("generation", "") or ""
    return {"content": [{"type": "text", "text": text}], "usage": parsed.get("usage", {})}


def _parse_gemma_response(response_body: bytes) -> dict[str, Any]:
    parsed = json.loads(response_body)
    text = ""
    if "candidates" in parsed and parsed["candidates"]:
        c = parsed["candidates"][0]
        if "content" in c and "parts" in c["content"] and c["content"]["parts"]:
            text = c["content"]["parts"][0].get("text", "")
    return {"content": [{"type": "text", "text": text}], "usage": parsed.get("usage", {})}


def _extract_text(model_response: dict[str, Any]) -> str:
    if "content" in model_response and model_response["content"]:
        return model_response["content"][0].get("text", "")
    return ""

def _extract_text_from_content_blocks(blocks: Any) -> str:
    """
    Extract text from Bedrock content blocks.

    `converse` returns blocks like: [{"text": "..."}] (plus possible non-text blocks).
    Some legacy invoke_model responses return: [{"type": "text", "text": "..."}].
    """
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if isinstance(b.get("text"), str):
            parts.append(b["text"])
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
            continue
    return "".join(parts).strip()


def _converse_request(
    client: Any,
    *,
    model_id: str,
    message: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """
    Preferred multi-provider Bedrock call: bedrock-runtime `converse`.

    This works across providers (Meta/Llama, Google/Gemma, OpenAI GPT-OSS, Anthropic, etc.)
    with a consistent request/response shape.
    """
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": message}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    blocks = (((resp or {}).get("output") or {}).get("message") or {}).get("content")
    text = _extract_text_from_content_blocks(blocks)
    usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
    return {"text": text, "usage": usage, "raw_response": resp if isinstance(resp, dict) else None}


def _strip_json_from_text(text: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace to get raw JSON."""
    text = text.strip()
    # ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text


def parse_and_validate_qa_generator_response(text: str) -> list[dict[str, Any]]:
    """
    Parse response text as JSON array and validate required fields for QA generator output.

    Each item must have: question, answer, required_communities, supporting_findings,
    supporting_connectors, intent_pattern, and either why_multi_community (preferred) or
    why_global (legacy).

    Returns:
        List of validated QA items.
    Raises:
        ValueError: If JSON is invalid or required fields are missing.
    """
    raw = _strip_json_from_text(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model response: {e}") from e
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array; got " + type(data).__name__)
    required_base = {
        "question",
        "answer",
        "required_communities",
        "supporting_findings",
        "supporting_connectors",
        "intent_pattern",
    }
    valid_patterns = {"explainer", "connection", "trigger", "consequence", "stake"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i}: expected object, got {type(item).__name__}")
        missing_base = required_base - set(item.keys())
        if missing_base:
            raise ValueError(f"Item {i}: missing fields {missing_base}")
        has_why_multi = "why_multi_community" in item
        has_why_global = "why_global" in item
        if not (has_why_multi or has_why_global):
            raise ValueError("Item %s: missing 'why_multi_community' (or legacy 'why_global')" % i)
        # Backward compatibility: if only legacy key exists, alias to the new name.
        if has_why_global and not has_why_multi:
            item["why_multi_community"] = item["why_global"]
        if item.get("intent_pattern") not in valid_patterns:
            raise ValueError(f"Item {i}: intent_pattern must be one of {valid_patterns}")
        if not isinstance(item["required_communities"], list) or len(item["required_communities"]) < 2:
            raise ValueError(f"Item {i}: required_communities must be a list of at least 2")
        if not isinstance(item["supporting_connectors"], list) or len(item["supporting_connectors"]) < 1:
            raise ValueError(f"Item {i}: supporting_connectors must be a non-empty list")
    return data


def parse_and_validate_verifier_response(text: str) -> list[dict[str, Any]]:
    """
    Parse response text as JSON array and validate verifier output.

    Verifier output supports:
    - New granular schema: per-criterion *_flag booleans and *_reasoning strings plus verdict.
    - Legacy schema: verdict (PASS/FAIL) + verification_reasons (or reasons) list.

    Returns:
        List of verdict items.
    Raises:
        ValueError: If JSON is invalid or required fields are missing.
    """
    raw = _strip_json_from_text(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model response: {e}") from e

    # Backward/forward compatibility:
    # - Older verifier prompts returned a JSON array of verdict objects.
    # - Newer per-candidate prompts may return a single verdict object.
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array or object; got " + type(data).__name__)

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i}: expected object, got {type(item).__name__}")

        # New granular schema detection.
        if "evidence_only_support_flag" in item:
            required_flags_and_reasoning = [
                ("evidence_only_support_flag", bool),
                ("evidence_only_support_reasoning", str),
                ("multi_community_necessity_flag", bool),
                ("multi_community_necessity_reasoning", str),
                ("connector_necessity_flag", bool),
                ("connector_necessity_reasoning", str),
                ("objective_qa_flag", bool),
                ("objective_qa_reasoning", str),
                ("natural_user_question_flag", bool),
                ("natural_user_question_reasoning", str),
                ("anti_trivia_flag", bool),
                ("anti_trivia_reasoning", str),
                ("evidence_presence_consistency_flag", bool),
                ("evidence_presence_consistency_reasoning", str),
                ("standalone_clarity_flag", bool),
                ("standalone_clarity_reasoning", str),
            ]
            # Fail-closed robustness: if the model omits a required key or returns wrong types,
            # do not crash the run. Instead, coerce to a failing value with an explicit reason.
            for k, t in required_flags_and_reasoning:
                if k not in item:
                    if k.endswith("_flag"):
                        item[k] = False
                    else:
                        item[k] = f"Missing field '{k}' in verifier output."
                    continue
                if not isinstance(item.get(k), t):
                    if k.endswith("_flag"):
                        item[k] = False
                    else:
                        item[k] = f"Invalid type for '{k}' (expected {t.__name__})."

            # Deterministic verdict: PASS iff all flags are True.
            flag_keys = [k for (k, t) in required_flags_and_reasoning if k.endswith("_flag")]
            all_true = all(item.get(k) is True for k in flag_keys)
            item["verdict"] = "PASS" if all_true else "FAIL"

            # Normalize legacy keys for downstream code (optional).
            # Derive verification_reasons from failed criteria to keep prior consumers working.
            failures: list[str] = []
            flag_to_reason = [
                ("evidence_only_support_flag", "evidence_only_support_reasoning"),
                ("multi_community_necessity_flag", "multi_community_necessity_reasoning"),
                ("connector_necessity_flag", "connector_necessity_reasoning"),
                ("objective_qa_flag", "objective_qa_reasoning"),
                ("natural_user_question_flag", "natural_user_question_reasoning"),
                ("anti_trivia_flag", "anti_trivia_reasoning"),
                ("evidence_presence_consistency_flag", "evidence_presence_consistency_reasoning"),
                ("standalone_clarity_flag", "standalone_clarity_reasoning"),
            ]
            for fk, rk in flag_to_reason:
                if item.get(fk) is False:
                    failures.append(f"{fk}: {item.get(rk, '')}".strip())
            item["verification_reasons"] = failures
            item["reasons"] = failures
            continue

        # Legacy schema: accept either key and normalize to both for downstream compatibility.
        if "verdict" not in item:
            raise ValueError(f"Item {i}: missing 'verdict'")
        v = item["verdict"]
        if str(v).upper() not in ("PASS", "FAIL"):
            raise ValueError(f"Item {i}: verdict must be PASS or FAIL, got {v!r}")

        reasons = None
        if "verification_reasons" in item:
            reasons = item.get("verification_reasons")
        elif "reasons" in item:
            reasons = item.get("reasons")
        if reasons is None:
            raise ValueError(f"Item {i}: missing 'verification_reasons' (or legacy 'reasons')")
        if not isinstance(reasons, list):
            raise ValueError(f"Item {i}: verification_reasons must be a list")
        item["verification_reasons"] = reasons
        item["reasons"] = reasons
    return data


def invoke_bedrock(
    prompt: str,
    model_id: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    region: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
) -> dict[str, Any]:
    """
    Invoke Bedrock model with retries for throttling.

    Returns dict with keys: text, usage (optional), raw_response (optional).
    On failure after retries, raises the last ClientError or Exception.
    """
    client = _get_bedrock_client(region)
    family = _get_model_family(model_id)

    # Preferred: Bedrock `converse` API (multi-provider). If it fails (e.g. unsupported model ID),
    # fall back to legacy invoke_model request formatting below.
    try:
        return _converse_request(
            client,
            model_id=model_id,
            message=(f"{system}\n\n{prompt}" if system else prompt),
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        logger.warning("Bedrock converse failed (%s) for model %s; falling back to invoke_model", code, model_id)
    except Exception as e:
        logger.warning("Bedrock converse failed for model %s; falling back to invoke_model (%s)", model_id, e)

    if family == "claude":
        body = _format_claude_body(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        parse_response = _parse_claude_response
    elif family == "llama":
        body = _format_llama_body(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        parse_response = _parse_llama_response
    elif family == "gemma":
        body = _format_gemma_body(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        parse_response = _parse_gemma_response
    else:
        body = _format_claude_body(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        parse_response = _parse_claude_response

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.invoke_model(modelId=model_id, body=body)
            parsed = parse_response(response["body"].read())
            text = _extract_text(parsed)
            return {"text": text, "usage": parsed.get("usage", {}), "raw_response": parsed}
        except ClientError as e:
            last_error = e
            code = e.response.get("Error", {}).get("Code", "")
            if code in THROTTLE_CODES:
                backoff = initial_backoff * (2 ** attempt)
                logger.warning("Bedrock throttled (%s), retry %s/%s in %.1fs", code, attempt + 1, max_retries, backoff)
                time.sleep(backoff)
                continue
            raise
        except Exception as e:
            last_error = e
            raise
    if last_error:
        raise last_error
    raise RuntimeError("invoke_bedrock: max retries exceeded")
