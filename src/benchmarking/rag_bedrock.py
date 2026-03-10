"""
RAG Pipeline with Bedrock for QA benchmarking.

This module implements Mode 2: RAG with SearxNG using AWS Bedrock LLM.
It replaces Ollama with Bedrock while keeping the same RAG pipeline structure.

Workflow:
1. Query rewriting (Bedrock LLM)
2. Search retrieval (SearxNG)
3. Context formatting
4. Answer generation (Bedrock LLM with retrieved context)
"""

import os
import sys
from typing import Optional

# Handle imports for both direct execution and module import
try:
    from .bedrock_client import bedrock_generate
    from .rag.retrieval import search_searxng
    from .rag.pipeline import format_search_results_as_context
except ImportError:
    # If running as script, use absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.benchmarking.bedrock_client import bedrock_generate
    from src.benchmarking.rag.retrieval import search_searxng
    from src.benchmarking.rag.pipeline import format_search_results_as_context


# Prompt template for query rewriting using Bedrock
QUERY_REWRITER_PROMPT_TEMPLATE = """You are a query rewriter model in a RAG pipeline. You are given a question and you need to rewrite it into a more effective search query optimized for web search engines.

Your task is to:
1. Extract key search terms and concepts
2. Remove unnecessary words while preserving meaning
3. Optimize for different question types:
   - Factual questions: Focus on key facts and entities
   - How-to questions: Include action verbs and relevant keywords
   - Comparison questions: Include both items being compared
   - Why/What/When/Where questions: Emphasize the core topic
4. Make the query concise and search-engine friendly

Question: {question}

Rewrite the question into a more effective search query. Return only the rewritten query, nothing else."""


# Prompt template for RAG generation using Bedrock
# - Use retrieved context when it helps.
# - If context is missing/irrelevant, still answer the question (do not meta-comment on context).
# - Output only the answer.
RAG_GENERATION_PROMPT_TEMPLATE = """You are a question answering assistant.

You will be given a Question and Context (retrieved web snippets).

Instructions:
- Use the Context as helpful evidence when it contains the answer.
- If the Context is irrelevant, incomplete, or does not contain the answer, still answer the Question using your general knowledge and best judgment.
- Do NOT mention the Context, sources, or retrieval in your response.
- Output only the answer (no preamble, no explanation). Keep it as short as possible.

Question: {query}

Context: {context}"""


def bedrock_query_rewriter(
    question: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: Optional[str] = None
) -> str:
    """
    Rewrite a natural language question into an optimized search query using Bedrock LLM.
    
    Args:
        question: Original natural language question to rewrite
        model_id: Bedrock model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
    
    Returns:
        Rewritten search query string optimized for search engines
    
    Raises:
        Exception: If the Bedrock API request fails
    """
    prompt = QUERY_REWRITER_PROMPT_TEMPLATE.format(question=question)
    
    rewritten_query = bedrock_generate(
        prompt=prompt,
        model_id=model_id,
        max_tokens=200,  # Queries should be concise
        temperature=0.2,  # Balanced creativity and consistency
        region=region
    )
    
    # Clean up the query by removing quotes and whitespace
    return rewritten_query.strip().strip('"').strip("'")


def bedrock_rag_generation(
    query: str,
    context: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: Optional[str] = None
) -> str:
    """
    Generate a response using Bedrock LLM in a RAG pipeline.
    
    Takes a user query and retrieved context, formats them using the RAG prompt template,
    and sends a request to Bedrock API to generate a concise, factual response based on
    the provided context.
    
    Args:
        query: User query/question to answer
        context: Retrieved context from search/retrieval system to use for answering
        model_id: Bedrock model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
    
    Returns:
        Generated answer string from the LLM
    
    Raises:
        Exception: If the Bedrock API request fails
    """
    prompt = RAG_GENERATION_PROMPT_TEMPLATE.format(query=query, context=context)
    
    answer = bedrock_generate(
        prompt=prompt,
        model_id=model_id,
        max_tokens=1000,
        temperature=0.1,  # Low temperature for factual, deterministic responses
        region=region
    )
    
    return answer.strip()


def rag_bedrock_pipeline(
    question: str,
    max_results: int = 5,
    searxng_url: Optional[str] = None,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: Optional[str] = None,
    verbose: bool = True,
    rewrite_query: bool = False,
) -> str:
    """
    RAG pipeline using Bedrock LLM and SearxNG retrieval.
    
    This is Mode 2: RAG with SearxNG where the LLM calls are made via Bedrock
    instead of Ollama. The pipeline:
    1. (Optional) Reformulates question into an effective search query (Bedrock)
    2. Retrieves relevant information from SearxNG
    3. Formats search results as context
    4. Generates answer using Bedrock LLM with retrieved context
    
    Args:
        question: User's original question
        max_results: Maximum number of search results to retrieve (default: 5)
        searxng_url: SearxNG base URL (defaults to SEARXNG_URL env var or http://localhost:8080)
        model_id: Bedrock model ID (default: "anthropic.claude-3-haiku-20240307-v1:0")
        region: AWS region (defaults to AWS_DEFAULT_REGION env var or us-east-1)
        verbose: Whether to print progress messages (default: True)
        rewrite_query: If True, use Bedrock to rewrite the query before SearxNG retrieval.
                      If False (default), search with the original question to save tokens
                      and improve retrieval caching consistency across models.
    
    Returns:
        Generated answer string
    
    Raises:
        ValueError: If question is empty or invalid
        ConnectionError: If unable to connect to SearxNG or Bedrock
        Exception: If any step in the pipeline fails
    """
    # Validate input question
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Load configuration from parameters, environment variables, or defaults
    searxng_base_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
    
    if verbose:
        print("=" * 70)
        print("RAG Pipeline with Bedrock (Mode 2)")
        print("=" * 70)
        print(f"Question: {question}")
        print(f"Configuration:")
        print(f"  SearxNG URL: {searxng_base_url}")
        print(f"  Bedrock Model: {model_id}")
        print(f"  Max Results: {max_results}")
        print("=" * 70)
        print()
    
    try:
        # Step 1: Decide search query
        # Default behavior: use the original question (saves tokens; consistent caching across models).
        search_query = question
        rewritten_query: Optional[str] = None
        if rewrite_query:
            if verbose:
                print("Step 1: Query Reformulation (Bedrock)...")
            try:
                rewritten_query = bedrock_query_rewriter(
                    question,
                    model_id=model_id,
                    region=region
                )
                # Clean up the query by removing quotes and whitespace
                rewritten_query = rewritten_query.strip().strip('"').strip("'")
                search_query = rewritten_query or question
                if verbose:
                    print(f"✓ Rewritten query: {search_query}\n")
            except Exception as e:
                raise ConnectionError(f"Failed to reformulate query with Bedrock: {e}") from e
        
        # Step 2: Search retrieval using SearxNG
        if verbose:
            print("Step 2: Search Retrieval (SearxNG)...")
        try:
            results = search_searxng(
                search_query,
                base_url=searxng_base_url,
                max_results=max_results
            )
            
            # Check for API-level errors in the response
            if 'error' in results:
                raise ConnectionError(f"SearxNG search failed: {results.get('error', 'Unknown error')}")
            
            # Validate that we got results
            num_results = len(results.get('results', []))
            
            # If reformulated query returns no results, try original question as fallback
            if num_results == 0 and rewrite_query and search_query != question:
                if verbose:
                    print(f"⚠ Reformulated query returned 0 results, trying original question...")
                # Fallback to original question
                results = search_searxng(
                    question,  # Use original question instead
                    base_url=searxng_base_url,
                    max_results=max_results
                )
                num_results = len(results.get('results', []))
                if verbose and num_results > 0:
                    print(f"✓ Retrieved {num_results} search results using original question\n")
            
            if num_results == 0:
                if rewrite_query and search_query != question:
                    error_msg = f"No search results found for query: '{search_query}' (also tried original: '{question}')"
                else:
                    error_msg = f"No search results found for query: '{search_query}'"
                if verbose:
                    print(f"✗ {error_msg}")
                raise ValueError(error_msg)
            
            if verbose:
                print(f"✓ Retrieved {num_results} search results\n")
                
        except Exception as e:
            raise ConnectionError(f"Search retrieval failed: {e}") from e
        
        # Step 3: Format context
        if verbose:
            print("Step 3: Formatting Context...")
        context = format_search_results_as_context(results)
        if verbose:
            print(f"✓ Formatted context ({len(context)} characters)\n")
        
        # Step 4: Generate answer using Bedrock
        if verbose:
            print("Step 4: Answer Generation (Bedrock)...")
        try:
            answer = bedrock_rag_generation(
                query=question,  # Use original question (not rewritten query) for answer generation
                context=context,
                model_id=model_id,
                region=region
            )
            if verbose:
                print("✓ Generated answer\n")
        except Exception as e:
            raise ConnectionError(f"Answer generation failed with Bedrock: {e}") from e
        
        return answer
        
    except (ValueError, ConnectionError) as e:
        # Handle expected errors (input validation, connection issues)
        if verbose:
            print(f"✗ Pipeline error: {e}\n")
        raise
    except Exception as e:
        # Handle unexpected errors with more context
        error_msg = f"Unexpected pipeline error: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}\n")
        raise Exception(error_msg) from e


# Test script: Run this file directly to test the RAG Bedrock pipeline
# if __name__ == "__main__":
#     print("Testing RAG Pipeline with Bedrock (Mode 2)...")
#     print("=" * 60)
    
#     # Test 1: Simple factual question
#     print("\nTest 1: Simple Factual Question")
#     print("-" * 60)
#     try:
#         question = "What is the capital of France?"
#         answer = rag_bedrock_pipeline(
#             question,
#             max_results=3,
#             verbose=True
#         )
#         print(f"\nFinal Answer: {answer}")
#         print("✓ Test 1 successful!")
#     except Exception as e:
#         print(f"✗ Test 1 failed: {e}")
    
#     # Test 2: More complex question requiring search
#     print("\n" + "=" * 60)
#     print("Test 2: Complex Question Requiring Search")
#     print("-" * 60)
#     try:
#         question = "Who won the Nobel Prize in Physics in 2023?"
#         answer = rag_bedrock_pipeline(
#             question,
#             max_results=5,
#             verbose=True
#         )
#         print(f"\nFinal Answer: {answer}")
#         print("✓ Test 2 successful!")
#     except Exception as e:
#         print(f"✗ Test 2 failed: {e}")
    
#     # Test 3: Question with specific information
#     print("\n" + "=" * 60)
#     print("Test 3: Specific Information Question")
#     print("-" * 60)
#     try:
#         question = "What is the population of Tokyo?"
#         answer = rag_bedrock_pipeline(
#             question,
#             max_results=5,
#             verbose=False  # Less verbose for cleaner output
#         )
#         print(f"Question: {question}")
#         print(f"Answer: {answer}")
#         print("✓ Test 3 successful!")
#     except Exception as e:
#         print(f"✗ Test 3 failed: {e}")
    
#     # Test 4: Empty question validation
#     print("\n" + "=" * 60)
#     print("Test 4: Empty Question Validation")
#     print("-" * 60)
#     try:
#         answer = rag_bedrock_pipeline("", verbose=False)
#         print("✗ Test 4 failed: Should have raised ValueError")
#     except ValueError as e:
#         print(f"✓ Test 4 successful: Correctly raised ValueError - {e}")
#     except Exception as e:
#         print(f"✗ Test 4 failed: Unexpected error - {e}")
    
#     # Test 5: Query rewriter standalone
#     print("\n" + "=" * 60)
#     print("Test 5: Query Rewriter Standalone")
#     print("-" * 60)
#     try:
#         question = "What is the speed of light in a vacuum?"
#         rewritten = bedrock_query_rewriter(question)
#         print(f"Original: {question}")
#         print(f"Rewritten: {rewritten}")
#         print("✓ Test 5 successful!")
#     except Exception as e:
#         print(f"✗ Test 5 failed: {e}")
    
#     print("\n" + "=" * 60)
#     print("RAG Bedrock testing complete!")
#     print("\nNote: This is Mode 2 - RAG with SearxNG using Bedrock LLM instead of Ollama.")

