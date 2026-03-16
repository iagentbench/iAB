"""
Bedrock-backed LLM wrapper for the Reflexion HotPotQA runs.

This mirrors `llm.py`'s API (a callable object that takes a single prompt string
and returns a response string), but uses AWS Bedrock instead of OpenAI.

Why this exists:
- The upstream repo uses LangChain's OpenAI wrappers (`ChatOpenAI` / `OpenAI`).
- For your setup, we want Bedrock (no OpenAI credits required).

Drop-in usage:
- Construct it with similar kwargs used in `agents.py` (temperature, max_tokens,
  model_kwargs={"stop": "\n"}, model_name=...).
- Call it with a prompt: `text = llm(prompt)`.

Notes:
- Bedrock model selection is controlled via (in priority order):
  1) explicit `model_id=...` kwarg
  2) env var `BEDROCK_MODEL_ID`
  3) default Claude Haiku model id
- `model_name` is accepted for compatibility but not required for Bedrock.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Sequence, Union


def _ensure_dyn_benchmarking_src_on_path() -> None:
    """
    Make `src/` importable so we can reuse the project's
    Bedrock helper (`benchmarking.bedrock_client.bedrock_generate`).
    Reflexion is now inside src/benchmarking/reflexion/hotpotqa_runs.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # hotpotqa_runs/ -> reflexion/ -> benchmarking/ -> src/
    dyn_src = os.path.abspath(os.path.join(here, "..", "..", ".."))
    if dyn_src not in sys.path:
        sys.path.insert(0, dyn_src)


def _apply_stop(text: str, stop: Optional[Union[str, Sequence[str]]]) -> str:
    if not stop:
        return text
    stops: Sequence[str] = (stop,) if isinstance(stop, str) else list(stop)
    cut = None
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text[:cut] if cut is not None else text


class AnyBedrockLLM:
    """
    Bedrock version of the upstream `AnyOpenAILLM` wrapper.

    It intentionally supports the same init kwargs pattern used throughout the
    Reflexion HotPotQA code: `temperature`, `max_tokens`, `model_name`,
    `model_kwargs={"stop": ...}`, etc.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        # Compatibility: accept but ignore args (upstream passes only kwargs).
        self.temperature: float = float(kwargs.get("temperature", 0))
        self.max_tokens: int = int(kwargs.get("max_tokens", 256))

        model_kwargs = kwargs.get("model_kwargs") or {}
        self.stop = model_kwargs.get("stop")

        # Bedrock selection
        self.model_id: str = (
            kwargs.get("model_id")
            or os.environ.get("BEDROCK_MODEL_ID")
            or "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.region: Optional[str] = kwargs.get("region") or os.environ.get("AWS_DEFAULT_REGION")

        # Accepted for compatibility with upstream code; not required for Bedrock.
        self.model_name: str = kwargs.get("model_name", self.model_id)

        _ensure_dyn_benchmarking_src_on_path()
        try:
            from benchmarking.bedrock_client import bedrock_generate  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import `benchmarking.bedrock_client.bedrock_generate`.\n"
                "Expected src/benchmarking/ to be present (reflexion is inside benchmarking).\n"
                f"Original error: {e}"
            )
        self._bedrock_generate = bedrock_generate

    def __call__(self, prompt: str) -> str:
        text = self._bedrock_generate(
            prompt,
            model_id=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            region=self.region,
        )
        return _apply_stop(text, self.stop)


# Optional alias to make later migrations easier:
# If you switch `agents.py` to `from llm_bedrock import AnyOpenAILLM`,
# everything else can stay identical.
AnyOpenAILLM = AnyBedrockLLM


