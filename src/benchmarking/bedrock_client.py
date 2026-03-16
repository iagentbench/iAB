"""
Bedrock client wrapper for AWS Bedrock models.

This module provides functionality to interact with AWS Bedrock API for generating
responses. It supports multiple model families (Claude, Gemma, etc.) with their
respective request/response formats.

Based on examples/bedrock_ping.ipynb
"""

import os
import boto3
import json
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError


# AWS configuration (can be overridden via environment variables)
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _get_bedrock_client(region: Optional[str] = None) -> boto3.client:
    """
    Get or create a Bedrock Runtime client.
    
    Args:
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
    
    Returns:
        boto3 bedrock-runtime client
    """
    region_name = region or AWS_REGION
    return boto3.client("bedrock-runtime", region_name=region_name)

def _extract_text_from_content_blocks(blocks: Any) -> str:
    """
    Extract a best-effort text string from Bedrock content blocks.

    `converse` returns: [{"text": "..."}] (plus possible non-text blocks).
    Some legacy invoke_model responses return: [{"type": "text", "text": "..."}].
    """
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        # Converse-style
        if isinstance(b.get("text"), str):
            parts.append(b["text"])
            continue
        # Legacy Claude-style
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
            continue
    return "".join(parts).strip()


def _standardize_response_text(text: str, usage: Optional[Dict[str, Any]] = None, raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Standardize to the shape expected by this repo: response['content'][0]['text'].
    """
    resp: Dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "usage": usage or {},
    }
    if raw is not None:
        resp["raw"] = raw
    return resp


def _converse_request(
    client: boto3.client,
    *,
    model_id: str,
    message: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """
    Call Bedrock Runtime `converse` with a simple user message.

    This is the preferred path for multi-provider Bedrock model IDs (OpenAI, Anthropic,
    DeepSeek, Meta, Mistral, etc.).
    """
    resp = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": message}],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    # Standard response per Bedrock: resp["output"]["message"]["content"] is a list of blocks.
    blocks = (((resp or {}).get("output") or {}).get("message") or {}).get("content")
    text = _extract_text_from_content_blocks(blocks)
    usage = resp.get("usage", {}) if isinstance(resp, dict) else {}
    return _standardize_response_text(text=text, usage=usage, raw=resp if isinstance(resp, dict) else None)


def _format_claude_request(message: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
    """
    Format request for Anthropic Claude models.
    
    Args:
        message: Message to send
        max_tokens: Maximum tokens in response
        temperature: Temperature for sampling
    
    Returns:
        JSON string of the formatted request
    """
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": message}],
            }
        ],
    }
    return json.dumps(native_request)


def _format_gemma_request(message: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
    """
    Format request for Google Gemma models.
    
    Args:
        message: Message to send
        max_tokens: Maximum tokens in response
        temperature: Temperature for sampling
    
    Returns:
        JSON string of the formatted request
    """
    native_request = {
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "maxOutputTokens": max_tokens,
        "temperature": temperature,
    }
    return json.dumps(native_request)


def _parse_claude_response(response_body: bytes) -> Dict[str, Any]:
    """
    Parse response from Anthropic Claude models.
    
    Args:
        response_body: Raw response body bytes
    
    Returns:
        Parsed response dictionary
    """
    return json.loads(response_body)


def _parse_gemma_response(response_body: bytes) -> Dict[str, Any]:
    """
    Parse response from Google Gemma models.
    
    Args:
        response_body: Raw response body bytes
    
    Returns:
        Parsed response dictionary with standardized format
    """
    parsed = json.loads(response_body)
    # Gemma returns choices[0].message.content
    # Standardize to match Claude format for easier use
    text = ""
    if "choices" in parsed and len(parsed["choices"]) > 0:
        choice = parsed["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            text = choice["message"]["content"]
    
    # Return in standardized format
    return {
        "content": [{"type": "text", "text": text}],
        "usage": parsed.get("usage", {})
    }


def _get_model_family(model_id: str) -> str:
    """
    Determine model family from model ID.
    
    Args:
        model_id: Bedrock model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
    
    Returns:
        Model family string: "claude", "gemma", or "unknown"
    """
    model_id_lower = model_id.lower()
    if "claude" in model_id_lower or "anthropic" in model_id_lower:
        return "claude"
    elif "gemma" in model_id_lower or "google" in model_id_lower:
        return "gemma"
    else:
        return "unknown"


def ping_bedrock_model(
    message: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens: int = 512,
    temperature: float = 0.5,
    region: Optional[str] = None,
    verbose: bool = True,
    raise_on_error: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Ping an AWS Bedrock model with a message.
    
    This function sends a message to an AWS Bedrock model and returns the response.
    It automatically detects the model family (Claude, Gemma, etc.) and uses the
    appropriate request/response format.
    
    Args:
        message: Message to send (required)
        model_id: Model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        max_tokens: Maximum tokens in response (default: 512)
        temperature: Temperature for sampling (default: 0.5)
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
        verbose: Whether to print progress messages (default: True)
    
    Returns:
        Response dictionary from the model, or None if error occurred.
        Returns standardized format with:
          - 'content': [{'type': 'text', 'text': <str>}]
          - 'usage': token usage metadata when available
          - 'raw': raw Bedrock response (best-effort; may be omitted)
    
    Example:
        >>> response = ping_bedrock_model("What is the capital of France?")
        >>> print(response["content"][0]["text"])
        "The capital of France is Paris."
    """
    try:
        if verbose:
            print(f"Pinging {model_id}...")
        
        # Get Bedrock client
        client = _get_bedrock_client(region)

        # Preferred: Bedrock `converse` (multi-provider).
        try:
            model_response = _converse_request(
                client,
                model_id=model_id,
                message=message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = model_response["content"][0].get("text", "")
            if verbose:
                print("✓ Success! Model responded.")
                print(f"Response: {response_text}")
            return model_response
        except ClientError as e:
            # Fallback: legacy `invoke_model` path for older/unsupported IDs.
            if raise_on_error:
                raise
            if verbose:
                err_code = e.response.get("Error", {}).get("Code", "Unknown")
                err_msg = e.response.get("Error", {}).get("Message", str(e))
                print(f"⚠ Converse failed [{err_code}]: {err_msg}")
                print("  Falling back to legacy invoke_model format...")

        # Legacy path: determine model family and format request accordingly
        model_family = _get_model_family(model_id)
        if model_family == "claude":
            request_body = _format_claude_request(message, max_tokens, temperature)
            parse_response = _parse_claude_response
        elif model_family == "gemma":
            request_body = _format_gemma_request(message, max_tokens, temperature)
            parse_response = _parse_gemma_response
        else:
            if verbose:
                print(f"Warning: Unknown model family for {model_id}, defaulting to Claude format")
            request_body = _format_claude_request(message, max_tokens, temperature)
            parse_response = _parse_claude_response

        response = client.invoke_model(modelId=model_id, body=request_body)
        legacy = parse_response(response["body"].read())
        response_text = _extract_text_from_content_blocks(legacy.get("content"))
        model_response = _standardize_response_text(text=response_text, usage=legacy.get("usage", {}), raw=legacy if isinstance(legacy, dict) else None)
        
        if verbose:
            print(f"✓ Success! Model responded.")
            print(f"Response: {response_text}")
        
        return model_response
        
    except ClientError as e:
        if raise_on_error:
            raise
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        if verbose:
            print(f"✗ AWS Client Error [{error_code}]: {error_msg}")
            if error_code == 'ValidationException':
                print("  Hint: Check if the model ID is correct and available in your region")
            elif error_code == 'AccessDeniedException':
                print("  Hint: Model access may not be enabled. Request access in AWS Bedrock console")
        return None
    except Exception as e:
        if raise_on_error:
            raise
        error_msg = str(e)
        if verbose:
            print(f"✗ Error: {error_msg}")
        return None


def bedrock_generate(
    prompt: str,
    # model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    # model_id: str = "google.gemma-3-4b-it",
    model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens: int = 1000,
    temperature: float = 0.1,
    region: Optional[str] = None
) -> str:
    """
    Generate a response using AWS Bedrock (simplified interface for RAG pipeline).
    
    This is a simplified wrapper around ping_bedrock_model designed for use in
    RAG pipelines. It returns just the text response string.
    
    Args:
        prompt: Prompt to send to the model
        model_id: Model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        max_tokens: Maximum tokens in response (default: 1000)
        temperature: Temperature for sampling (default: 0.1 for factual responses)
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
    
    Returns:
        Generated response text string
    
    Raises:
        Exception: If the Bedrock API request fails
    
    Example:
        >>> answer = bedrock_generate("What is Python?", model_id="google.gemma-3-4b")
        >>> print(answer)
    """
    response = ping_bedrock_model(
        message=prompt,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        region=region,
        verbose=False
    )
    
    if response is None:
        raise Exception(f"Bedrock API request failed for model {model_id}")
    
    # Extract text from response
    if "content" in response and isinstance(response["content"], list) and len(response["content"]) > 0:
        return str(response["content"][0].get("text", "") or "")
    else:
        raise Exception(f"Unexpected response format from model {model_id}")


# Test script: Run this file directly to test the Bedrock client
# if __name__ == "__main__":
#     print("Testing Bedrock client...")
#     print("=" * 60)
    
#     # Test 1: Claude model (default)
#     print("\nTest 1: Claude Haiku Model")
#     print("-" * 60)
#     try:
#         response = ping_bedrock_model(
#             "Hello, this is a test message.",
#             model_id="anthropic.claude-3-haiku-20240307-v1:0"
#         )
#         if response:
#             print("✓ Claude test successful!")
#         else:
#             print("✗ Claude test failed: No response")
#     except Exception as e:
#         print(f"✗ Claude test failed: {e}")
    
#     # Test 2: Simple question
#     print("\nTest 2: Simple Question")
#     print("-" * 60)
#     try:
#         response = ping_bedrock_model("What is the capital of France?")
#         if response:
#             print("✓ Question test successful!")
#         else:
#             print("✗ Question test failed: No response")
#     except Exception as e:
#         print(f"✗ Question test failed: {e}")
    
#     # Test 3: Gemma model (if available)
#     print("\nTest 3: Gemma Model (if available)")
#     print("-" * 60)
#     try:
        # # Correct Gemma 3 model IDs for AWS Bedrock
        # # Note: Gemini is NOT available on Bedrock, only Gemma models are
        # gemma_models = [
        #     "google.gemma-3-4b-it",      # Gemma 3 4B Instruct (your target model)
        #     "google.gemma-3-12b-it",    # Gemma 3 12B Instruct
        #     "google.gemma-3-27b-it",     # Gemma 3 27B Instruct
        # ]
        
#         gemma_available = False
#         last_error = None
#         for gemma_model in gemma_models:
#             try:
#                 print(f"  Trying {gemma_model}...")
#                 response = ping_bedrock_model(
#                     "What is Python?",
#                     model_id=gemma_model,
#                     verbose=False
#                 )
#                 if response:
#                     print(f"✓ Gemma test successful with {gemma_model}!")
#                     gemma_available = True
#                     break
#                 else:
#                     print(f"  ✗ {gemma_model}: No response returned")
#             except Exception as e:
#                 last_error = str(e)
#                 print(f"  ✗ {gemma_model}: {last_error}")
#                 continue
        
#         if not gemma_available:
#             print("\n⚠ Gemma models not available or not accessible in this account")
#             print("  Possible reasons:")
#             print("  1. Model access not requested in AWS Bedrock console")
#             print("  2. Model not available in your region (us-east-1)")
#             print("  3. Incorrect model ID format")
#             if last_error:
#                 print(f"  Last error: {last_error}")
#     except Exception as e:
#         print(f"✗ Gemma test failed: {e}")
    
#     # Test 4: Simplified interface
#     print("\nTest 4: Simplified Interface (bedrock_generate)")
#     print("-" * 60)
#     try:
#         answer = bedrock_generate(
#             "What is the capital of Japan?",
#             max_tokens=100,
#             temperature=0.1
#         )
#         print(f"Answer: {answer}")
#         print("✓ Simplified interface test successful!")
#     except Exception as e:
#         print(f"✗ Simplified interface test failed: {e}")
    
#     print("\n" + "=" * 60)
#     print("Testing complete!")
#     print("\nTroubleshooting:")
#     print("1. Ensure AWS credentials are configured (~/.aws/credentials or environment variables)")
#     print("2. Verify Bedrock service is enabled in your AWS account")
#     print("3. Check model access in AWS Bedrock console")
#     print(f"4. Verify AWS region (current: {AWS_REGION})")

