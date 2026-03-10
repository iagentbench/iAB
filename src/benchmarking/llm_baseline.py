"""
LLM Only Baseline mode for QA benchmarking.

This module implements Mode 1: Direct Bedrock API call where a question
is sent directly to the LLM without any retrieval or agentic reasoning.

Workflow: question → LLM → answer
"""

from typing import Optional
from .bedrock_client import bedrock_generate


# Prompt template for concise baseline answers
# Instructs the model to give only the answer, no sentences or explanations
BASELINE_PROMPT_TEMPLATE = """Answer the following question. Provide only the answer itself, no complete sentences, no explanations, no additional context. Just the answer.

Question: {question}

Answer:"""


def llm_baseline_answer(
    question: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens: int = 512,
    temperature: float = 0.1,
    region: Optional[str] = None
) -> str:
    """
    Generate an answer to a question using LLM only (no retrieval, no agentic reasoning).
    
    This is the baseline mode that sends the question directly to the Bedrock LLM
    and returns the generated answer. No external knowledge retrieval or multi-step
    reasoning is performed.
    
    Args:
        question: The question to answer
        model_id: Bedrock model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        max_tokens: Maximum tokens in response (default: 512)
        temperature: Temperature for sampling (default: 0.1 for concise, deterministic responses)
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
    
    Returns:
        Generated answer string from the LLM
    
    Raises:
        Exception: If the Bedrock API request fails
    
    Example:
        >>> answer = llm_baseline_answer("What is the capital of France?")
        >>> print(answer)
        "Paris"
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Format prompt with instruction for concise answer only
    prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
    
    # Direct LLM call - question → LLM → answer
    answer = bedrock_generate(
        prompt=prompt,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        region=region
    )
    
    return answer.strip()


# Test script: Run this file directly to test the LLM baseline
if __name__ == "__main__":
    print("Testing LLM Baseline Mode...")
    print("=" * 60)
    
    # Test 1: Simple factual question
    print("\nTest 1: Simple Factual Question")
    print("-" * 60)
    try:
        question = "What is the capital of France?"
        answer = llm_baseline_answer(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("✓ Test 1 successful!")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test 2: More complex question
    print("\nTest 2: Complex Question")
    print("-" * 60)
    try:
        question = "Who wrote the novel '1984' and when was it published?"
        answer = llm_baseline_answer(question, max_tokens=200)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("✓ Test 2 successful!")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Test 3: Question requiring knowledge
    print("\nTest 3: Knowledge-Based Question")
    print("-" * 60)
    try:
        question = "What is the speed of light in a vacuum?"
        answer = llm_baseline_answer(question, temperature=0.1)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("✓ Test 3 successful!")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    # Test 4: Empty question (should raise error)
    print("\nTest 4: Empty Question Validation")
    print("-" * 60)
    try:
        answer = llm_baseline_answer("")
        print(f"✗ Test 4 failed: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Test 4 successful: Correctly raised ValueError - {e}")
    except Exception as e:
        print(f"✗ Test 4 failed: Unexpected error - {e}")
    
    # Test 5: Different model (if available)
    print("\nTest 5: Different Model")
    print("-" * 60)
    try:
        question = "What is Python?"
        # Try with a different Claude model if available
        answer = llm_baseline_answer(
            question,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=150
        )
        print(f"Question: {question}")
        print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
        print("✓ Test 5 successful!")
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
    
    print("\n" + "=" * 60)
    print("LLM Baseline testing complete!")
    print("\nNote: This is Mode 1 - Direct LLM call without retrieval or agentic reasoning.")