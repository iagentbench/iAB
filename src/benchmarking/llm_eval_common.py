"""
Shared LLM-based evaluation helper.

This module exposes a single entrypoint:

    llm_eval_metrics(question, answer, response, backend="ollama" | "bedrock", ...)

It implements the canonical grading spec:
  - Labels: "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
  - Criteria: rich Obama / numeric-tolerance guidance used across the repo.

Backends:
  - "ollama": calls a local Ollama HTTP endpoint.
  - "bedrock": calls an AWS Bedrock model using the same prompt.
"""

import os
from typing import Optional

from .bedrock_client import ping_bedrock_model


def _canonical_eval_prompt(question: str, answer: str, response: str) -> str:
    """
    Canonical grading prompt shared across backends.
    """
    return f'''
        Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

        First, I will give examples of each grade, and then you will grade a new example.

        The following are examples of CORRECT predicted answers.

        Question: What are the names of Barack Obama's children?
        Gold target: Malia Obama and Sasha Obama
        Predicted answer 1: sasha and malia obama
        Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
        Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.

        These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

        The following are examples of INCORRECT predicted answers.

        Question: What are the names of Barack Obama's children?
        Gold target: Malia and Sasha
        Predicted answer 1: Malia.
        Predicted answer 2: Malia, Sasha, and Susan.
        Predicted answer 3: Barack Obama does not have any children.
        Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
        Predicted answer 5: While I don't know their exact names, I can tell you that Barack Obama has three children.
        Predicted answer 6: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?

        These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.

        The following are examples of NOT_ATTEMPTED predicted answers.

        Question: What are the names of Barack Obama's children?
        Gold target: Malia and Sasha
        Predicted answer 1: I don't know.
        Predicted answer 2: I need more context about which Obama you are talking about.
        Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
        Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.

        These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.

        Also note the following things:
        - For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
            - Predicted answers "120k", "124k", and "115k" are all CORRECT. 
            - Predicted answers "100k" and "113k" are INCORRECT. 
            - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
        - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
            - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
        - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
            - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
            - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
            - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
            - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
        - Do not punish for typos in people's name if it's clearly the same name. 
            - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

        Here is a new example. Simply reply with either CORRECT, INCORRECT, or NOT_ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

        Question: {question}
        Gold target: {answer}
        Predicted answer: {response}
    '''.format(question=question, answer=answer, response=response)


def _normalize_raw_label(raw: str) -> str:
    """
    Normalize a raw model string into one of the canonical labels, if possible.
    """
    if not raw:
        return raw
    text = raw.upper()
    if "CORRECT" in text and "INCORRECT" not in text:
        return "CORRECT"
    if "INCORRECT" in text:
        return "INCORRECT"
    if "NOT_ATTEMPTED" in text or "NOT ATTEMPTED" in text:
        return "NOT_ATTEMPTED"
    return raw


def _call_ollama(prompt: str, model: Optional[str], base_url: Optional[str]) -> str:
    """
    Call a local Ollama model with the grading prompt.
    """
    import requests

    ollama_url = base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434"
    ollama_model = model or os.getenv("OLLAMA_MODEL") or "gemma3:latest"
    chat_url = f"{ollama_url.rstrip('/')}/api/chat"

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1000,
        },
    }

    http_response = requests.post(chat_url, json=payload, timeout=240)
    http_response.raise_for_status()
    return str(http_response.json()["message"]["content"]).strip()


def _call_bedrock(prompt: str, model_id: Optional[str], region: Optional[str]) -> str:
    """
    Call an AWS Bedrock model with the grading prompt.
    """
    resolved_model_id = model_id or os.getenv("BEDROCK_EVAL_MODEL_ID") or os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0",
    )
    resp = ping_bedrock_model(
        message=prompt,
        model_id=resolved_model_id,
        max_tokens=256,
        temperature=0.1,
        region=region,
        verbose=False,
    )
    text = ""
    try:
        if resp and isinstance(resp, dict):
            content = resp.get("content") or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    text = str(first.get("text", "")).strip()
    except Exception:
        text = ""
    return text


def llm_eval_metrics(
    question: str,
    answer: str,
    response: str,
    *,
    backend: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    bedrock_model_id: Optional[str] = None,
    bedrock_region: Optional[str] = None,
) -> str:
    """
    Evaluate a response against ground truth using an LLM classifier.

    Args:
        question: Original question text (may be empty but recommended).
        answer: Ground truth answer string.
        response: Predicted answer string to grade.
        backend: "ollama" (default) or "bedrock".
        ollama_model: Optional override of Ollama model name.
        ollama_base_url: Optional override of Ollama API base URL.
        bedrock_model_id: Optional override of Bedrock model ID.
        bedrock_region: Optional override of AWS region for Bedrock.

    Returns:
        "CORRECT", "INCORRECT", or "NOT_ATTEMPTED" (or the raw string if parsing fails).
    """
    # Resolve backend: explicit arg → env var → default "ollama"
    resolved_backend = (backend or os.getenv("LLM_EVAL_BACKEND") or "ollama").lower()

    prompt = _canonical_eval_prompt(question=question or "", answer=str(answer), response=str(response))

    try:
        if resolved_backend == "bedrock":
            raw = _call_bedrock(prompt, model_id=bedrock_model_id, region=bedrock_region)
        else:
            # Default: Ollama
            raw = _call_ollama(prompt, model=ollama_model, base_url=ollama_base_url)
    except Exception as e:
        return f"ERROR: {e}"

    return _normalize_raw_label(raw)

