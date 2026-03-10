"""
Dataset loader module for QA benchmarking.

This module provides functionality to load QA datasets from Hugging Face:
1. SimpleQA - Simple question-answering dataset
# 2. TriviaQA - Large-scale reading comprehension dataset (not being used anymore)
3. HotpotQA - Multi-hop question-answering dataset
4. iAgentBench (ISAbench) - Instruction-following QA benchmark

All datasets are loaded in a consistent format for benchmarking.
"""

from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
import json

# Note: A CSV file of SimpleQA is also saved in evaluation/datasets/simpleqa_csv.csv
# This provides a local backup option if Hugging Face is unavailable

def load_simpleqa():
    """
    Load the SimpleQA dataset from Hugging Face.
    
    Loads the SimpleQA dataset using the Hugging Face datasets library.
    The dataset is cached locally after first download for faster subsequent loads.
    
    Returns:
        DatasetDict containing the SimpleQA dataset with the following structure:
        - 'test': Dataset split containing question-answer pairs
        - Each example contains:
          * 'problem': The question/query
          * 'answer': The ground truth answer
          * 'metadata': Additional metadata about the question
    
    Example:
        >>> dataset = load_simpleqa()
        >>> print(f"Test examples: {len(dataset['test'])}")
        >>> first_example = dataset['test'][0]
        >>> print(f"Question: {first_example['problem']}")
        >>> print(f"Answer: {first_example['answer']}")
    
    Note:
        - Requires internet connection for first-time download
        - Dataset is cached locally after first download
        - Dataset source: basicv8vc/SimpleQA on Hugging Face
    """
    return load_dataset("basicv8vc/SimpleQA")


def load_triviaqa():
    """
    Load the TriviaQA dataset from Hugging Face.
    
    Loads the TriviaQA dataset using the Hugging Face datasets library.
    The dataset is cached locally after first download for faster subsequent loads.
    
    The dataset can be loaded from:
    1. Local saved copy (primary method) - saved in evaluation/datasets/trivia_qa
    2. Hugging Face datasets library (backup option)
    
    Returns:
        DatasetDict containing the TriviaQA dataset with the following structure:
        - 'test': Dataset split containing question-answer pairs
        - Each example contains:
          * 'question': The question/query
          * 'answer': The ground truth answer (as a dict with 'value', 'aliases', etc.)
          * Other metadata fields
    
    Example:
        >>> dataset = load_triviaqa()
        >>> print(f"Test examples: {len(dataset['test'])}")
        >>> first_example = dataset['test'][0]
        >>> print(f"Question: {first_example['question']}")
        >>> print(f"Answer: {first_example['answer']}")
    
    Note:
        - Local copy is saved in evaluation/datasets/trivia_qa
        - Falls back to Hugging Face if local copy not found
        - Dataset source: mandarjoshi/trivia_qa on Hugging Face
        - Uses 'rc' subset by default
        - Reference: https://huggingface.co/datasets/mandarjoshi/trivia_qa
    """
    # Try to load from local saved copy first
    local_path = Path(__file__).parent.parent / "evaluation" / "datasets" / "trivia_qa"
    if local_path.exists():
        from datasets import load_from_disk
        return load_from_disk(str(local_path))
    
    # Fallback to Hugging Face
    return load_dataset("mandarjoshi/trivia_qa", "rc")


def load_hotpotqa(subset: str = "distractor"):
    """
    Load the HotpotQA dataset from Hugging Face.
    
    HotpotQA is a dataset with 113k Wikipedia-based question-answer pairs that require
    finding and reasoning over multiple supporting documents to answer. The questions are
    diverse and require multi-hop reasoning.
    
    Args:
        subset: Subset to load - "distractor" (default) or "fullwiki"
            - "distractor": Includes distractor paragraphs (97.9k rows)
            - "fullwiki": Uses full Wikipedia (105k rows)
    
    Returns:
        DatasetDict containing the HotpotQA dataset with the following structure:
        - 'train': Training split
        - 'validation': Validation split (distractor) or 'test' (fullwiki)
        - Each example contains:
          * 'id': Unique identifier
          * 'question': The question/query
          * 'answer': The ground truth answer (string)
          * 'type': Question type ("comparison" or "bridge")
          * 'level': Difficulty level ("easy", "medium", "hard")
          * 'supporting_facts': Dictionary with supporting fact titles and sentence IDs
          * 'context': Dictionary with context paragraphs (titles and sentences)
    
    Example:
        >>> dataset = load_hotpotqa()
        >>> print(f"Validation examples: {len(dataset['validation'])}")
        >>> first_example = dataset['validation'][0]
        >>> print(f"Question: {first_example['question']}")
        >>> print(f"Answer: {first_example['answer']}")
    
    Note:
        - Requires internet connection for first-time download
        - Dataset is cached locally after first download
        - Dataset source: hotpotqa/hotpot_qa on Hugging Face
        - Reference: https://huggingface.co/datasets/hotpotqa/hotpot_qa
        - Paper: https://arxiv.org/abs/1809.09600
    """
    return load_dataset("hotpotqa/hotpot_qa", subset)


def load_iagentbench():
    """
    Load the iAgentBench / ISAbench dataset from Hugging Face.
    
    This is the preferred way to access ISAbench going forward, instead of relying
    on local JSONL copies. The dataset is available at:
    
        https://huggingface.co/datasets/preetam7/iAgentBench
    
    Returns:
        DatasetDict containing the iAgentBench dataset. The exact split names are
        determined by the dataset configuration on Hugging Face (typically a single
        split such as 'train').
    
    Example:
        >>> ds = load_iagentbench()
        >>> print(ds)
    """
    return load_dataset("preetam7/iAgentBench")


def load_simpleqa_local(jsonl_path: str = None):
    """
    Load the SimpleQA dataset from a local JSONL file.
    
    Loads the SimpleQA dataset from a JSONL file in inputs/final/ directory.
    Each line is a JSON object with 'problem' (question) and 'answer' fields.
    
    Args:
        jsonl_path: Path to the JSONL file. If None, uses default path:
            inputs/final/simpleqa_verified_500.jsonl
    
    Returns:
        DatasetDict containing the SimpleQA dataset with the following structure:
        - 'test': Dataset split containing question-answer pairs
        - Each example contains:
          * 'problem': The question/query
          * 'answer': The ground truth answer
          * Other metadata fields from the JSONL file
    
    Example:
        >>> dataset = load_simpleqa_local()
        >>> print(f"Test examples: {len(dataset['test'])}")
        >>> first_example = dataset['test'][0]
        >>> print(f"Question: {first_example['problem']}")
        >>> print(f"Answer: {first_example['answer']}")
    
    Note:
        - Loads from local JSONL file (no internet required)
        - Default path: inputs/final/simpleqa_verified_500.jsonl
    """
    if jsonl_path is None:
        # Default to inputs/final/simpleqa_verified_500.jsonl relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        jsonl_path = repo_root / "inputs" / "final" / "simpleqa_verified_500.jsonl"
    else:
        jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"SimpleQA JSONL file not found: {jsonl_path}")
    
    # Read JSONL file
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list(data)
    
    # Return as DatasetDict with 'test' split (for consistency with HuggingFace format)
    return DatasetDict({'test': dataset})


def load_hotpotqa_local(jsonl_path: str = None):
    """
    Load the HotpotQA dataset from a local JSONL file.
    
    Loads the HotpotQA dataset from a JSONL file in inputs/final/ directory.
    Each line is a JSON object with 'question' and 'answer' fields.
    
    Args:
        jsonl_path: Path to the JSONL file. If None, uses default path:
            inputs/final/hotpotqa_fullwiki_validation_500.jsonl
    
    Returns:
        DatasetDict containing the HotpotQA dataset with the following structure:
        - 'validation': Dataset split containing question-answer pairs
        - Each example contains:
          * 'id': Unique identifier
          * 'question': The question/query
          * 'answer': The ground truth answer (string)
          * 'type': Question type ("comparison" or "bridge")
          * 'level': Difficulty level ("easy", "medium", "hard")
          * 'supporting_facts': Dictionary with supporting fact titles and sentence IDs
          * 'context': Dictionary with context paragraphs (titles and sentences)
    
    Example:
        >>> dataset = load_hotpotqa_local()
        >>> print(f"Validation examples: {len(dataset['validation'])}")
        >>> first_example = dataset['validation'][0]
        >>> print(f"Question: {first_example['question']}")
        >>> print(f"Answer: {first_example['answer']}")
    
    Note:
        - Loads from local JSONL file (no internet required)
        - Default path: inputs/final/hotpotqa_fullwiki_validation_500.jsonl
    """
    if jsonl_path is None:
        # Default to inputs/final/hotpotqa_fullwiki_validation_500.jsonl relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        jsonl_path = repo_root / "inputs" / "final" / "hotpotqa_fullwiki_validation_500.jsonl"
    else:
        jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"HotpotQA JSONL file not found: {jsonl_path}")
    
    # Read JSONL file
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list(data)
    
    # Return as DatasetDict with 'validation' split (for consistency with HuggingFace format)
    return DatasetDict({'validation': dataset})