"""
Agentic Solution for QA benchmarking using Pydantic AI.

This module implements Mode 3: Agentic solution where an AI agent decides
when to search (SearxNG) and processes results to generate answers.

All LLM calls are made via AWS Bedrock.
"""

import os
import sys
from typing import Optional
from pydantic import BaseModel, Field

# Handle imports for both direct execution and module import
try:
    from .bedrock_client import bedrock_generate
    from .rag.retrieval import search_searxng
except ImportError:
    # If running as script, use absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.benchmarking.bedrock_client import bedrock_generate
    from src.benchmarking.rag.retrieval import search_searxng

# Try to import Pydantic AI, with helpful error if not installed
try:
    from pydantic_ai import Agent, Tool
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    # Create dummy classes for type hints
    class Agent:
        pass
    class BedrockConverseModel:
        pass


class QAAnswer(BaseModel):
    """Structured output for agent answers."""
    answer: str = Field(..., description="The direct, concise answer to the question. NO complete sentences, NO explanations, NO additional context. Just the answer itself. For people: just names (e.g., 'Pierre Agostini, Ferenc Krausz, Anne L'Huillier'). For numbers: just the number. For places: just the place name.")
    searched: bool = Field(default=False, description="Whether the agent used search to find the answer")


def get_bedrock_model(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1"
):
    """
    Get a Bedrock model for Pydantic AI.
    
    Args:
        model_id: Bedrock model ID
        region: AWS region
    
    Returns:
        BedrockConverseModel instance
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
        )
    
    provider = BedrockProvider(region_name=region)
    settings = BedrockModelSettings()
    model = BedrockConverseModel(
        model_name=model_id,
        provider=provider,
        settings=settings
    )
    return model


def create_search_tool(searxng_url: Optional[str] = None, max_results: int = 5):
    """
    Create a search tool function for the agent.
    
    Args:
        searxng_url: SearxNG base URL
        max_results: Maximum number of results to return
    
    Returns:
        Tool function that can be registered with the agent
    """
    base_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
    
    def search_tool(query: str) -> str:
        """
        Search tool for the agent to use when it needs external information.
        
        Args:
            query: Search query string
        
        Returns:
            Formatted search results as a string
        """
        try:
            results = search_searxng(
                query,
                base_url=base_url,
                max_results=max_results
            )
            
            if 'error' in results:
                return f"Search error: {results.get('error', 'Unknown error')}"
            
            if 'results' not in results or not results['results']:
                return "No search results found."
            
            # Format results into a readable string
            formatted_results = []
            for i, result in enumerate(results['results'], 1):
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                content = result.get('content', result.get('snippet', 'No content'))
                formatted_results.append(f"Result {i}: {title}\nURL: {url}\nContent: {content}\n")
            
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    return search_tool


def create_agentic_agent(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    searxng_url: Optional[str] = None,
    max_results: int = 5
) -> Agent:
    """
    Create a Pydantic AI agent configured for QA tasks.
    
    The agent can decide when to search and when to answer directly.
    
    Args:
        model_id: Bedrock model ID
        region: AWS region
        searxng_url: SearxNG base URL
        max_results: Maximum search results
    
    Returns:
        Configured Agent instance
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
        )
    
    model = get_bedrock_model(model_id, region)
    search_tool_func = create_search_tool(searxng_url, max_results)
    
    # Create the search tool
    search_tool = Tool(
        search_tool_func,
        name="search",
        description="Search the web for information. Use this when you need current information or specific facts that you're not certain about."
    )
    
    agent = Agent(
        model,
        system_prompt="""You are a question-answering agent. Your task is to answer questions accurately and concisely.

CRITICAL: When answering:
- Provide ONLY the answer itself - no complete sentences, no explanations, no additional context
- If the answer is a person or people, just put their names (e.g., "John Smith" or "John Smith, Jane Doe")
- If the answer is a number, just the number
- If the answer is a place, just the place name
- NEVER include phrases like "The answer is", "According to", or explanatory text
- Extract ONLY the essential information needed to answer the question

You have access to a search tool. Use it when:
- You need current information (recent events, latest data)
- You need specific facts that you're not certain about
- The question asks about something that requires up-to-date information
- You are not certain about the answer

When you use search, analyze the results carefully and extract ONLY the essential answer - no explanations or full sentences.""",
        output_type=QAAnswer,
        tools=[search_tool]
    )
    
    return agent


def agentic_answer(
    question: str,
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    region: str = "us-east-1",
    searxng_url: Optional[str] = None,
    max_results: int = 5,
    verbose: bool = False
) -> str:
    """
    Generate an answer using the agentic solution (Mode 3).
    
    The agent decides when to search and processes results to generate the answer.
    
    Args:
        question: The question to answer
        model_id: Bedrock model ID
        region: AWS region
        searxng_url: SearxNG base URL
        max_results: Maximum search results
        verbose: Whether to print progress messages
    
    Returns:
        Generated answer string
    
    Raises:
        ImportError: If pydantic-ai is not installed
        ValueError: If question is empty
        Exception: If agent execution fails
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is not installed. Install it with: pip install pydantic-ai"
        )
    
    if verbose:
        print("=" * 70)
        print("Agentic Solution (Mode 3)")
        print("=" * 70)
        print(f"Question: {question}")
        print(f"Model: {model_id}")
        print("=" * 70)
        print()
    
    try:
        agent = create_agentic_agent(
            model_id=model_id,
            region=region,
            searxng_url=searxng_url,
            max_results=max_results
        )
        
        if verbose:
            print("Running agent...")
        
        # Run the agent synchronously
        result = agent.run_sync(question)
        
        if verbose:
            print(f"Agent searched: {result.output.searched}")
            print(f"Answer: {result.output.answer}")
        
        return result.output.answer
        
    except Exception as e:
        error_msg = f"Agentic solution failed: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}")
        raise Exception(error_msg) from e


# Test script: Run this file directly to test the agentic solution
if __name__ == "__main__":
    print("Testing Agentic Solution (Mode 3)...")
    print("=" * 60)
    
    if not PYDANTIC_AI_AVAILABLE:
        print("\n✗ pydantic-ai is not installed!")
        print("Install with: pip install pydantic-ai")
        print("Or install all requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test 1: Simple question (should answer directly)
    print("\nTest 1: Simple Question (Should Answer Directly)")
    print("-" * 60)
    try:
        question = "What is the capital of France?"
        answer = agentic_answer(question, verbose=True)
        print(f"\nFinal Answer: {answer}")
        print("✓ Test 1 successful!")
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test 2: Question requiring search
    print("\n" + "=" * 60)
    print("Test 2: Question Requiring Search")
    print("-" * 60)
    try:
        question = "Who won the Nobel Prize in Physics in 2023?"
        answer = agentic_answer(question, verbose=True)
        print(f"\nFinal Answer: {answer}")
        print("✓ Test 2 successful!")
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Test 3: Current information question
    print("\n" + "=" * 60)
    print("Test 3: Current Information Question")
    print("-" * 60)
    try:
        question = "What is the current population of Tokyo?"
        answer = agentic_answer(question, verbose=False)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("✓ Test 3 successful!")
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    # Test 4: Empty question validation
    print("\n" + "=" * 60)
    print("Test 4: Empty Question Validation")
    print("-" * 60)
    try:
        answer = agentic_answer("", verbose=False)
        print("✗ Test 4 failed: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Test 4 successful: Correctly raised ValueError - {e}")
    except Exception as e:
        print(f"✗ Test 4 failed: Unexpected error - {e}")
    
    print("\n" + "=" * 60)
    print("Agentic Solution testing complete!")
    print("\nNote: This is Mode 3 - Agentic solution where the agent decides when to search.")

