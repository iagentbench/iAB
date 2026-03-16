"""
Reflexion-based Agentic Solution (Mode 3) adapter for this benchmarking repo.

Goal
----
Expose a simple callable that the benchmark runner can use *per question*:
    question -> (optionally ground_truth for Reflexion reward) -> answer

We reuse the upstream Reflexion HotPotQA ReAct agent + reflection loop, but:
- swap OpenAI -> Bedrock via `reflexion/hotpotqa_runs/llm_bedrock.py`
- run 1 question at a time (fits our benchmark runner)
- run a small, configurable number of "trials" per question (e.g., 2–3)

Important note (methodology):
----------------------------
Reflexion as implemented in the paper uses access to the correct answer to
decide if an attempt was correct (reward signal). This is *not* a fair "agent"
in a strict test-time setting (it leaks ground truth). If you still want to use
it as Mode 3 (because it's citable / matches the paper), pass `ground_truth`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


def _ensure_reflexion_hotpotqa_on_path() -> str:
    """
    Add `.../benchmarking/reflexion/hotpotqa_runs` to sys.path so we can import
    the Bedrock-backed agent classes. Reflexion is now inside benchmarking.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # Reflexion is inside benchmarking: src/benchmarking/reflexion/hotpotqa_runs
    hotpotqa_runs = os.path.abspath(os.path.join(here, "reflexion", "hotpotqa_runs"))
    if hotpotqa_runs not in sys.path:
        sys.path.insert(0, hotpotqa_runs)
    return hotpotqa_runs


@dataclass
class ReflexionRunResult:
    answer: str
    trials_used: int
    is_correct: Optional[bool]
    halted: bool


def agentic_answer_reflexion(
    question: str,
    ground_truth: str,
    *,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    max_trials: int = 5,
    max_steps: int = 6,
    searxng_url: Optional[str] = None,
    verbose: bool = False,
) -> ReflexionRunResult:
    """
    Answer a single question using Reflexion's ReAct+Reflection loop (Bedrock-backed).

    How it works (high level):
    - Create a `ReactReflectAgent(question, key=ground_truth)`.
    - Run `agent.run(...)` repeatedly for up to `max_trials`.
    - After each trial:
        - If the agent matched ground truth via Reflexion EM -> stop early.
        - Otherwise reflect, then try again (the agent carries reflections forward).

    Returns a small struct with the final answer + metadata (handy for debugging).
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None (pass an empty string if needed)")

    # Configure Bedrock defaults used by `AnyBedrockLLM` (constructor defaults in agents_bedrock).
    os.environ["BEDROCK_MODEL_ID"] = model_id
    os.environ["AWS_DEFAULT_REGION"] = region

    _ensure_reflexion_hotpotqa_on_path()

    # Import after sys.path tweak
    from agents_bedrock import ReactReflectAgent, ReflexionStrategy  # type: ignore

    agent = ReactReflectAgent(
        question=question,
        key=ground_truth,
        max_steps=max_steps,
        searxng_url=searxng_url,
    )

    last_answer = ""
    halted = False
    correct: Optional[bool] = None

    for trial_idx in range(1, max_trials + 1):
        if verbose:
            print(f"[reflexion] trial {trial_idx}/{max_trials}")
        agent.run(reset=True, reflect_strategy=ReflexionStrategy.REFLEXION)

        last_answer = getattr(agent, "answer", "") or ""
        halted = bool(agent.is_halted())
        correct = bool(agent.is_correct())

        if verbose:
            print(f"[reflexion] answer: {last_answer}")
            print(f"[reflexion] correct: {correct}, halted: {halted}")

        if correct:
            return ReflexionRunResult(
                answer=last_answer,
                trials_used=trial_idx,
                is_correct=True,
                halted=halted,
            )

    # Not correct after max_trials: return last attempt.
    return ReflexionRunResult(
        answer=last_answer,
        trials_used=max_trials,
        is_correct=correct,
        halted=halted,
    )


def agentic_answer(
    question: str,
    *,
    ground_truth: str = "",
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    max_trials: int = 5,
    max_steps: int = 6,
    searxng_url: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Convenience wrapper that matches the style of the existing Mode 3 function.

    In `benchmark_runner.py`, we can call:
        agentic_answer(question, ground_truth=ground_truth, model_id=..., region=...)

    Returns just the answer string.
    """
    result = agentic_answer_reflexion(
        question,
        ground_truth,
        model_id=model_id,
        region=region,
        max_trials=max_trials,
        max_steps=max_steps,
        searxng_url=searxng_url,
        verbose=verbose,
    )
    return result.answer


