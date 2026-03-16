# QA Benchmarking with AWS Bedrock

This module provides comprehensive benchmarking capabilities for Question-Answering (QA) datasets using AWS Bedrock endpoints. It evaluates three different approaches to QA: LLM-only baseline, RAG with SearxNG, and agentic solutions.

## Overview

The benchmark evaluates QA performance across three distinct modes on datasets like SimpleQA, HotpotQA, and iAgentBench. Each mode represents a different approach to answering questions:

1. **LLM Only Baseline**: Direct question-to-answer using Bedrock LLM
2. **RAG with SearxNG**: Retrieval-Augmented Generation using web search
3. **Agentic Solution**: Intelligent agent that decides when to search

## Three Modes Explained

### Mode 1: LLM Only Baseline

**What it does:**
- Takes a question and sends it directly to AWS Bedrock LLM
- No external knowledge retrieval
- Pure language model reasoning based on training data

**Implementation:** `llm_baseline.py`
- Uses `bedrock_generate()` from `bedrock_client.py`
- Prompt template enforces concise, direct answers
- Default model: `anthropic.claude-3-haiku-20240307-v1:0`

**Use case:** Baseline performance to compare against more sophisticated approaches

### Mode 2: RAG with SearxNG

**What it does:**
- **Step 1 (optional)**: Bedrock LLM rewrites the question into an effective search query
- **Step 2**: Searches the web using SearxNG
- **Step 3**: Formats search results as context
- **Step 4**: Bedrock LLM generates answer using retrieved context

**Implementation:** `rag_bedrock.py`
- Query rewriting (optional): `bedrock_query_rewriter()` (can be disabled to save tokens)
- Search retrieval: Uses `search_searxng()` from `src/benchmarking/rag/retrieval.py`
- Answer generation: `bedrock_rag_generation()` with context

**Use case:** When questions require current information or specific facts not in training data

### Mode 3: Agentic Solution (Reflexion-based)

**What it does:**
- Uses Reflexion framework with ReAct (Reasoning + Acting) loop
- Intelligent agent with multiple tools: Wikipedia Search, Wikipedia Lookup, and WebSearch
- Agent decides which tool to use and when to search vs. answer directly
- Implements reflection: if answer is incorrect, reflects on mistakes and tries again (up to `max_trials` times)
- All LLM calls go through AWS Bedrock

**Available Tools:**
1. **`Search[entity]`**: Search Wikipedia for an entity/topic
   - Returns Wikipedia page summary or "Could not find" with similar suggestions
   - Fast, structured knowledge source
   - Example: `Search[Luke Skywalker]` → Returns Wikipedia article summary

2. **`Lookup[keyword]`**: Lookup a specific keyword in the last searched Wikipedia page
   - **Requirement**: Must be used after a successful `Search` action
   - If used without a prior Search, or if the last Search failed, returns an error message
   - Extracts specific information from the Wikipedia page
   - Example: After `Search[Star Wars]`, use `Lookup[Mark Hamill]` to find actor info

3. **`WebSearch[query]`**: Search the web using SearxNG
   - Returns 5 search results (default) with titles, URLs, and content snippets
   - **Primary use**: When Wikipedia Search fails or doesn't have the information
   - **Strategic use**: Agent can also choose WebSearch directly for queries better suited to web search
   - More comprehensive than Wikipedia for recent events, quotes, specific facts, or when Wikipedia search returns "Could not find"
   - Example: `WebSearch[Who played Luke Skywalker in 1977 Star Wars]` → Returns web search results

4. **`Finish[answer]`**: Provide the final answer
   - Agent uses this when it has enough information
   - Answer should be concise (just the essential information)

**Reflexion Strategy:**
- After each attempt, the agent evaluates correctness using **LLM-based semantic evaluation** (not exact string matching)
  - Uses `llm_eval_correct()` which judges if the answer semantically matches ground truth
  - More lenient than exact match: "Jerry Rawlings" matches "Jerry John Rawlings", "2" matches "two"
  - Handles embedded answers: "The player was Wout Weghorst" matches "Wout Weghorst"
- If the answer is judged incorrect:
  - Agent reflects on what went wrong
  - Generates a reflection that includes: the question, the previous attempt's full reasoning trace (all thoughts/actions/observations), and lessons learned
  - This reflection is added to context for the next trial
  - Tries again with the reflection in context (up to `max_trials` times, default: 5)
- Early stopping: If answer is judged correct, stops immediately (no reflection needed)
- Reflection helps the agent avoid repeating mistakes and guides it toward the correct approach

**Important Methodology Note:**
- Reflexion as implemented uses access to the correct answer (ground truth) to determine if an attempt was correct
- **Evaluation method**: Uses LLM-based semantic evaluation (`llm_eval_correct`), not exact string matching
  - This is more lenient and realistic than exact match
  - Handles variations like "Jerry Rawlings" vs "Jerry John Rawlings" as correct
  - The LLM classifier judges if the answer semantically contains the ground truth
- This provides a reward signal for the reflection process
- In a strict test-time setting, this leaks ground truth information (the agent knows if it's correct)
- However, this matches the original Reflexion paper methodology and is useful for benchmarking
- The agent still must figure out *how* to get the correct answer, even if it knows *what* the correct answer is

**Implementation:** `agentic_solution_reflexion.py`
- Wraps Reflexion's `ReactReflectAgent` from `reflexion/hotpotqa_runs/agents_bedrock.py`
- Bedrock-backed via `llm_bedrock.py` (swaps OpenAI → Bedrock)
- Configurable: `max_trials` (reflection attempts), `max_steps` (actions per trial)
- Default: 5 trials, 6 steps per trial
- **Understanding steps vs trials**:
  - A **trial** is one complete attempt to answer the question (with reflection if incorrect)
  - A **step** is one action (Search, Lookup, WebSearch, or Finish) within a trial
  - So: up to 6 actions per attempt, up to 5 attempts total (if all fail)
  - Most questions complete in 1-2 trials; complex questions may use all 5 trials

**Tool Usage Strategy:**
- **Wikipedia-first approach**: Tries `Search` first (fast, structured)
- **Escalation**: If Wikipedia fails, uses `WebSearch` (comprehensive)
- **Lookup for details**: Uses `Lookup` to extract specific info from Wikipedia pages
- **Smart decision-making**: Agent uses its internal knowledge (from training) to decide when to search vs. when it already knows the answer
  - If confident in its knowledge, may use `Finish` directly without searching
  - If uncertain or needs specific facts, searches first
  - The agent learns from reflections to make better decisions in subsequent trials

**Use case:** Optimal balance between speed (direct answers when possible) and accuracy (search when needed, reflect and retry when wrong)

## How SearxNG Works

SearxNG is a privacy-respecting meta-search engine that aggregates results from multiple search engines. It's used in **Mode 2 (RAG)** and **Mode 3 (Agentic)** to retrieve web information for answering questions.

### Search Function

**Location:** `src/benchmarking/rag/retrieval.py` → `search_searxng()`

**Default Configuration:**
- Base URL: `http://localhost:8080` (configurable via `SEARXNG_URL` env var)
- Default results: **5 results** (configurable via `max_results` parameter)
- Language: `en-US`
- Timeout: 30 seconds

### What You Get Back

The search function returns a dictionary with this structure:

```python
{
    'results': [
        {
            'title': 'Page/article title',
            'url': 'https://example.com/page',
            'content': 'Snippet description (1-3 sentences summarizing the page)',
            # OR 'snippet' (some engines use this field name)
        },
        # ... more results (default: 5 total)
    ],
    'query': 'Your original search query',
    'number_of_results': 1000,  # ⚠️ Estimated total (often inaccurate)
}
```

### Result Structure

Each result in `results['results']` contains:

- **`title`**: The page/article title
- **`url`**: Full URL to the source page
- **`content`** or **`snippet`**: A **few sentence summary/description** (typically 1-3 sentences) that summarizes what's on that page

**Note:** The `content` field is a short snippet extracted from the page - it's not the full page content, just a summary that search engines provide.

### How Many Results?

- **Default:** 5 results per search
- **Configurable:** Set `max_results` parameter when calling:
  - `rag_bedrock_pipeline(max_results=10)` → gets 10 results
  - `agentic_answer(max_results=3)` → gets 3 results
- **Limiting:** Results are limited both at the API level and client-side for safety

### How Results Are Formatted

**Location:** `src/benchmarking/rag/pipeline.py` → `format_search_results_as_context()`

Search results are formatted into a context string like this:

```
Result 1:
Title: Example Page Title
URL: https://example.com/page1
Content: This is a snippet describing what's on the page...

Result 2:
Title: Another Page Title
URL: https://another.com/page2
Content: More snippet content here...
```

This formatted context is then passed to the LLM (Bedrock) to generate answers.

### Usage in Benchmarking

**Mode 2 (RAG):**
1. Question → Bedrock rewrites into search query
2. SearxNG searches → returns 5 results (default)
3. Results formatted → context string
4. Bedrock generates answer using context

**Mode 3 (Agentic/Reflexion):**
1. Agent decides which tool to use (Wikipedia Search, Lookup, or WebSearch)
2. If Wikipedia Search → returns Wikipedia page summary
3. If WebSearch → SearxNG searches → returns 5 results (default)
4. Agent processes results → generates answer
5. If answer incorrect → reflects on mistakes → tries again (up to `max_trials` times)

### Important Notes

- **`number_of_results` field:** This is an estimated total and often inaccurate. Always use `len(results['results'])` to get the actual count of returned results.
- **Error handling:** If search fails, returns `{'error': '...', 'message': '...'}` dictionary instead of results.
- **Empty results:** If no results found, `results['results']` will be an empty list.

## Architecture

```
src/benchmarking/
├── bedrock_client.py      # AWS Bedrock API wrapper (handles Claude, Gemma, etc.)
├── dataset_loader.py      # Load SimpleQA / HotpotQA / iAgentBench from Hugging Face + local JSONL
├── llm_baseline.py        # Mode 1: Direct LLM answers
├── rag/                   # RAG retrieval and pipeline (SearxNG)
│   ├── retrieval.py       # search_searxng()
│   └── pipeline.py        # format_search_results_as_context()
├── rag_bedrock.py         # Mode 2: RAG pipeline with Bedrock
├── agentic_solution_reflexion.py  # Mode 3: Reflexion-based ReAct agent
├── evaluator.py          # Metrics calculation (exact match, F1)
├── evaluate_llm.py       # LLM-based evaluation (semantic assessment)
├── benchmark_runner.py    # Main orchestration script
└── reflexion/             # Reflexion framework (inside benchmarking)
    └── hotpotqa_runs/
        ├── agents_bedrock.py  # ReAct agent with Bedrock backend
        └── llm_bedrock.py     # Bedrock LLM wrapper (swaps OpenAI → Bedrock)
```

## Code Attribution and Modifications

### Reflexion Framework Integration

**Source:** This project uses code from the [Reflexion repository](https://github.com/noahshinn024/reflexion) (Shinn et al., 2023), specifically the HotpotQA ReAct agent implementation.

**What we did:**
1. **Integrated the Reflexion repository** into `src/benchmarking/reflexion/`
2. **Modified the code** to work with AWS Bedrock instead of OpenAI
3. **Integrated it** into our benchmarking system

**Key Modifications:**

1. **`src/benchmarking/reflexion/hotpotqa_runs/llm_bedrock.py`**:
   - **Original**: Used OpenAI's `ChatOpenAI` and `OpenAI` classes
   - **Modified**: Created `AnyOpenAILLM` wrapper that uses AWS Bedrock instead
   - **Change**: Swaps all OpenAI API calls → Bedrock API calls
   - **Purpose**: Allows Reflexion agents to use Bedrock models (Claude, Gemma, etc.)

2. **`src/benchmarking/reflexion/hotpotqa_runs/agents_bedrock.py`**:
   - **Original**: `agents.py` used OpenAI LLM backend
   - **Modified**: Created `agents_bedrock.py` that imports from `llm_bedrock.py` instead of `llm.py`
   - **Change**: All agent classes (`ReactAgent`, `ReactReflectAgent`, `CoTAgent`) now use Bedrock
   - **Additional**: Added `WebSearch` tool support (uses SearxNG) in addition to Wikipedia Search/Lookup
   - **Purpose**: Enables Reflexion agents to work with Bedrock and have web search capabilities

3. **`src/benchmarking/agentic_solution_reflexion.py`**:
   - **New file**: Wrapper that adapts Reflexion's agent for our benchmarking system
   - **Purpose**: Provides a simple interface (`agentic_answer()`) that matches our other modes
   - **Features**: Configurable trials, steps, model selection, and SearxNG integration

**What stayed the same:**
- Core Reflexion algorithm (ReAct loop, reflection strategy)
- Prompt templates and few-shot examples
- Agent logic and decision-making flow
- Evaluation methodology (LLM-based semantic evaluation)

**Result:**
- Mode 3 (Agentic Solution) uses Reflexion's reflection-based agent framework
- All LLM calls go through AWS Bedrock (instead of OpenAI)
- Agents have access to Wikipedia Search, Lookup, and WebSearch tools
- Fully integrated into our benchmarking pipeline

## How It Works

### 1. Dataset Loading

**SimpleQA:**
- Loaded from Hugging Face: `basicv8vc/SimpleQA`
- Uses `test` split
- Format: `problem` (string), `answer` (string)
- ~4,326 test examples

**HotpotQA:**
- Loaded from Hugging Face: `hotpotqa/hotpot_qa` or local JSONL: `inputs/final/hotpotqa_fullwiki_validation_500.jsonl`
- Supports `distractor` and `fullwiki` subsets
- Uses `validation` split by default
- Format: `question` (string), `answer` (string)

**iAgentBench:**
- Loaded from Hugging Face: `preetam7/iAgentBench`
- Uses `test` split (500 examples)
- Format: `question` (string), `answer` (string)

### 2. Benchmark Execution Flow

```
For each question in dataset:
  ├── Extract ground truth answer
  ├── Mode 1: LLM Baseline → Get answer
  ├── Mode 2: RAG Pipeline → Get answer
  ├── Mode 3: Agentic Solution → Get answer
  └── Calculate metrics (exact match, F1) for each mode
```

### 3. Evaluation Metrics

**Exact Match Accuracy:**
- Normalized string comparison (lowercase, remove punctuation)
- Returns True/False for each question
- Overall accuracy = correct / total

**F1 Score:**
- Token-level precision and recall
- Handles partial matches
- Range: 0.0 to 1.0

**LLM-Based Evaluation (semantic):**
- Semantic evaluation using an LLM classifier.
- Canonical labels: **"CORRECT"**, **"INCORRECT"**, **"NOT_ATTEMPTED"**.
- Grading prompt adapted from the OpenAI SimpleQA evaluation setup (Barack Obama children examples, numeric tolerance, name-variant handling).
- Shared implementation: `llm_eval_common.llm_eval_metrics(...)`.

**Implementation:** `evaluator.py`
- `calculate_metrics()`: Computes all metrics for a set of predictions
- `compare_modes()`: Compares performance across all three modes

### 4. Results Storage

All results saved to `outputs/results/`:

- **CSV files**: `benchmark_results_{dataset}_{split}_{entries}entries.csv`
  - Per-question results with all three modes
  - Includes exact match and F1 scores
  
- **JSON files**: `benchmark_results_{dataset}_{split}_{entries}entries.json`
  - Complete results with metadata
  - Includes summary statistics
  
- **Report files**: `benchmark_report_{dataset}_{split}_{entries}entries.txt`
  - Human-readable comparative report
  - Best performing mode identification

### 5. LLM-Based Evaluation (Ollama & Bedrock)

After running the benchmark, you can perform LLM-based **semantic** evaluation on the results.

**What it does:**
- Uses an LLM classifier to evaluate each mode's answers against the ground truth.
- Classifies responses as **"CORRECT"**, **"INCORRECT"**, or **"NOT_ATTEMPTED"** using a detailed grading spec (Barack-Obama examples, numeric tolerance, name variants, etc.).
- More nuanced than exact match/F1 (handles partial answers, hedging, contradictions).
- Core implementation: `src/benchmarking/llm_eval_common.py::llm_eval_metrics(...)`.

**Backends:**
- **Ollama backend (local)**
  - Controlled by environment variables:
    - `LLM_EVAL_BACKEND=ollama` (default if unset)
    - `OLLAMA_URL` (default: `http://localhost:11434`)
    - `OLLAMA_MODEL` (default: `gemma3:latest`)
  - Used when you want cheap, local semantic grading.

- **AWS Bedrock backend (remote)**
  - Controlled by environment variables:
    - `LLM_EVAL_BACKEND=bedrock`
    - `BEDROCK_EVAL_MODEL_ID` (e.g. an Anthropic Claude model on Bedrock)
    - `BEDROCK_MODEL_ID` (fallback if `BEDROCK_EVAL_MODEL_ID` is not set)
    - `AWS_DEFAULT_REGION` / `AWS_REGION` (e.g. `us-east-1`)
  - Uses the same canonical grading prompt, but sends it via `bedrock_client.ping_bedrock_model(...)`.

**Usage (Ollama or Bedrock):**

```bash
# Example: use Ollama as backend
export LLM_EVAL_BACKEND=ollama
export OLLAMA_MODEL="gemma3:latest"

# Example: use Bedrock as backend
# export LLM_EVAL_BACKEND=bedrock
# export BEDROCK_EVAL_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"

# Evaluate a single benchmark results file
python src/benchmarking/evaluate_llm.py benchmark_results_triviaqa_validation_500entries.csv

# Evaluate multiple files and combine results (recommended)
python src/benchmarking/evaluate_llm.py \
  benchmark_results_triviaqa_validation_500entries.csv \
  benchmark_results_simpleqa_test_500entries.csv
```

**Output:**
- **Individual CSV files**: `benchmark_results_{dataset}_{split}_{entries}entries_llm_eval.csv`
  - Original columns plus: `mode1_llm_eval`, `mode2_llm_eval`, `mode3_llm_eval`
  
- **Combined CSV file** (when multiple files provided): `benchmark_results_combined_llm_eval.csv`
  - All questions from all datasets combined
  - Includes `dataset` column indicating source (triviaqa/simpleqa/hotpotqa)
  - All original columns plus LLM evaluation columns
  
- **Combined Report file** (when multiple files provided): `benchmark_results_combined_llm_eval_report.txt`
  - **Combined Metrics**: Statistics across all datasets together
  - **Separate Metrics**: Statistics for each dataset individually
  - Comparative analysis showing best performing mode for combined and per-dataset

**How it works (high level):**
1. Reads one or more benchmark results CSV files (specify which files to process).
2. For each mode's answer, passes **question + ground truth + predicted answer** into `llm_eval_metrics(...)`.
3. The chosen backend (Ollama or Bedrock) runs the canonical grading prompt and returns one of `CORRECT` / `INCORRECT` / `NOT_ATTEMPTED`.
4. Saves individual results with progress tracking (resumable if interrupted).
5. Optionally combines datasets and writes a human-readable summary report.

**Resumability:**
- If interrupted, re-running will resume from where it left off.
- Saves progress every 50 rows.
- Only recalculates missing or error rows.

## Usage

### Repo layout & assumptions

- Code lives under:
  - `src/benchmarking/` (benchmark runner, evaluators, Bedrock client, LLM-eval)
  - `src/benchmarking/rag/` (SearxNG retrieval and RAG utilities)
  - `src/benchmarking/reflexion/` (agentic Reflexion integration used by Mode 3)
- Datasets:
  - **SimpleQA**:
    - Preferred: `load_simpleqa()` from Hugging Face (`basicv8vc/SimpleQA`).
    - Benchmark runner currently uses a local JSONL backup: `inputs/final/simpleqa_verified_500.jsonl`.
  - **HotpotQA**:
    - `load_hotpotqa()` from Hugging Face (`hotpotqa/hotpot_qa`) is available.
    - Benchmark runner uses a local JSONL subset: `inputs/final/hotpotqa_fullwiki_validation_500.jsonl`.
  - **iAgentBench / ISAbench**:
    - Loaded via `load_iagentbench()` from Hugging Face: `preetam7/iAgentBench`.
    - This is the recommended source for ISAbench going forward.
- Outputs:
  - `outputs/results/` for per-run CSV/JSON + text reports.
  - `outputs/overall_results/` for combined summary tables (e.g., `evaluation_report.txt`).

When you copy this code into a new repo, keep the same relative structure (`src/benchmarking`, `src/benchmarking/rag`, `src/benchmarking/reflexion`) and either:
- Bring the small local JSONL files in `inputs/final/` for SimpleQA/HotpotQA, **or**
- Update `benchmark_runner.py` to call the Hugging Face loaders directly.

### Quick Start

```bash
# Run benchmark on both datasets (100 entries each)
cd /path/to/iAgentBench
python src/benchmarking/benchmark_runner.py
```

### Configuration

Edit the `if __name__ == "__main__"` section in `benchmark_runner.py`:

```python
num_entries = 100  # Number of entries per dataset (0 = all)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
region = "us-east-1"
delay_between_questions = 2.0  # Seconds (reduce for faster, may hit rate limits)
```

### Running in Background (Recommended for Long Runs)

```bash
# Start in screen session (survives disconnects)
screen -S benchmark
cd /path/to/iAgentBench
python src/benchmarking/benchmark_runner.py

# Detach: Press Ctrl+A, then D
# Reconnect: screen -r benchmark
```

### Running Individual Datasets

```python
from src.benchmarking.benchmark_runner import run_benchmark

# Run SimpleQA only
results = run_benchmark(
    dataset_name="simpleqa",
    num_entries=100,
    split="test"
)

# Run HotpotQA only
results = run_benchmark(
    dataset_name="hotpotqa",
    num_entries=100,
    split="validation"
)

# Run iAgentBench only
results = run_benchmark(
    dataset_name="iagentbench",
    num_entries=100,
    split="test"
)
```

## Time Estimates

**Per Question:**
- Mode 1 (LLM Baseline): ~1-3 seconds
- Mode 2 (RAG): ~5-10 seconds (query rewrite + search + generation)
- Mode 3 (Agentic/Reflexion): ~5-30 seconds (varies based on number of tool calls and reflection trials)
  - Wikipedia Search: ~1-2 seconds per action
  - WebSearch: ~2-5 seconds per action
  - Reflection: ~2-5 seconds per trial (only if answer incorrect)
  - **Typical**: 1-2 trials, 2-4 steps = ~5-15 seconds for most questions
  - **Maximum**: up to 5 trials × 6 steps = potentially 30+ seconds for very complex questions requiring multiple searches and reflections
- Delay between questions: 2 seconds (configurable)

**Total Time:**
- 100 entries per dataset: ~1-2 hours
- 500 entries per dataset: ~2-3 hours
- Full datasets (22,270 total): ~4-8 days

## Requirements

### Python Dependencies

```bash
pip install -r src/benchmarking/requirements.txt
```

Key dependencies: `boto3`, `datasets`, `langchain`, `tiktoken`, `wikipedia` (required for Mode 3 Wikipedia Search/Lookup tools).

### AWS Configuration

1. AWS credentials configured (via `~/.aws/credentials` or environment variables)
2. Bedrock access enabled in your AWS account
3. Model access granted (Claude, Gemma, etc.)

### External Services

- **SearxNG**: Required for Mode 2 and Mode 3 (WebSearch tool)
  - Default URL: `http://localhost:8080` or set `SEARXNG_URL` env var
  - Must be running and accessible
  - Used by Mode 3's `WebSearch` tool for comprehensive web search

- **Wikipedia API**: Required for Mode 3 (Search and Lookup tools)
  - Accessed via LangChain's Wikipedia docstore
  - No configuration needed (uses public Wikipedia API)
  - Used by Mode 3's `Search` and `Lookup` tools for structured knowledge

### Environment Variables

```bash
export AWS_DEFAULT_REGION=us-east-1
export SEARXNG_URL=http://localhost:8080  # Optional, defaults to localhost
export SEARXNG_CACHE_DISABLE=1  # Optional, disable local SearxNG JSONL cache
```

## Supported Models

The benchmark supports any AWS Bedrock model. Tested with:

- `anthropic.claude-3-haiku-20240307-v1:0` (default)
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `google.gemma-3-4b-it-v1:0` (Gemma 3 4B)

Model family is auto-detected in `bedrock_client.py` (Claude vs Gemma formats).

### Inference profiles (recommended for large / provisioned models)

Some Bedrock models **do not support on-demand throughput** and must be invoked via an **inference profile** (you’ll otherwise see errors like “on-demand throughput isn’t supported”).

These inference-profile IDs are known to work in this repo’s environment:

- `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- `us.deepseek.r1-v1:0`
- `us.meta.llama4-maverick-17b-instruct-v1:0`
- `mistral.mistral-large-3-675b-instruct`
- `openai.gpt-oss-20b-1:0`

To sanity-check model access quickly, run the notebook `examples/bedrock_model_ping.ipynb`.

## Output Format

### CSV Columns

- `question_id`: Unique identifier
- `question`: The question text
- `ground_truth`: Correct answer
- `mode1_answer`, `mode1_exact_match`, `mode1_f1`, `mode1_error`: LLM Baseline results
- `mode2_answer`, `mode2_exact_match`, `mode2_f1`, `mode2_error`: RAG results
- `mode3_answer`, `mode3_exact_match`, `mode3_f1`, `mode3_error`: Agentic results

### Metrics Summary

```python
{
    'LLM Baseline': {
        'exact_match_accuracy': 0.650,
        'f1_score': 0.720
    },
    'RAG with SearxNG': {
        'exact_match_accuracy': 0.750,
        'f1_score': 0.810
    },
    'Agentic Solution': {
        'exact_match_accuracy': 0.780,
        'f1_score': 0.830
    }
}
```

## Troubleshooting

### Common Issues

**1. Bedrock API Errors**
- Check AWS credentials: `aws sts get-caller-identity`
- Verify model access in Bedrock console
- Check region configuration

**2. SearxNG Connection Errors**
- Ensure SearxNG is running: `curl http://localhost:8080`
- Check `SEARXNG_URL` environment variable
- Mode 2 requires SearxNG
- Mode 3 uses SearxNG for `WebSearch` tool (Wikipedia Search/Lookup don't require it)

**3. Import Errors**
- Ensure you're in the project root
- Check Python path: `python -c "import sys; print(sys.path)"`
- Install dependencies: `pip install -r requirements.txt`

**4. Dataset Loading Issues**
- First load downloads from Hugging Face (requires internet)
- Local JSONL fallbacks for SimpleQA/HotpotQA in `inputs/final/`
- Check disk space for large datasets

**5. Mode 3 "Could not import wikipedia"**
- Mode 3 uses LangChain's Wikipedia for Search/Lookup tools
- Install: `pip install wikipedia`
- Or use full deps: `pip install -r src/benchmarking/requirements.txt`

### Error Handling

The benchmark includes error handling:
- API failures: Logged, empty answer recorded, continues to next question
- Timeouts: Handled gracefully
- Invalid responses: Fallback to empty string

Check the `*_error` columns in CSV for per-question errors.

## References

- **Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **SimpleQA Dataset**: https://huggingface.co/datasets/basicv8vc/SimpleQA
- **HotpotQA Dataset**: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- **Reflexion Paper**: "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023)
- **SearxNG**: https://docs.searxng.org/
- **LangChain Wikipedia**: https://python.langchain.com/docs/integrations/tools/wikipedia
