#!/usr/bin/env python3
"""
Wrapper script with JSON parsing fix for community reports.
This fixes the issue where LLM returns 'findings' as a JSON string instead of a list.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

# Parse arguments
root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
output_dir = None
input_dir = None
verbose = False

# Parse arguments more carefully
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--output" or arg == "-o":
        if i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    elif arg == "--input" or arg == "-i":
        if i + 1 < len(sys.argv):
            input_dir = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    elif arg == "--verbose" or arg == "-v":
        verbose = True
        i += 1
    elif not arg.startswith("-") and i == 1:
        # First positional argument is root_dir
        root_dir = arg
        i += 1
    else:
        i += 1

# Setup logging
log_dir = Path(root_dir) / "logs"
log_dir.mkdir(exist_ok=True)
model_io_log = log_dir / f"bedrock_api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
model_io_log_file = open(model_io_log, 'w')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info(f"Model I/O logging enabled. Log file: {model_io_log}")

# Set AWS environment
os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
# AWS_BEARER_TOKEN_BEDROCK should be set via environment variable or AWS credentials
# Do not hardcode API keys in source code
bedrock_api_key = os.environ.get('AWS_BEARER_TOKEN_BEDROCK')
if bedrock_api_key:
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = bedrock_api_key

# Import and configure LiteLLM
import litellm
litellm.set_verbose = True

# Track API calls
api_call_count = {"chat": 0, "embedding": 0}

def log_api_call(*args, **kwargs) -> None:
    """Log all LiteLLM API calls with proper token extraction."""
    try:
        # LiteLLM callback receives different data structures
        # Try to extract from kwargs first (most common)
        data = kwargs if kwargs else {}
        
        # If args[0] is a dict, use it
        if args and isinstance(args[0], dict):
            data = {**data, **args[0]}
        
        timestamp = datetime.now().isoformat()
        model = data.get("model", data.get("model_name", "unknown"))
        event_type = data.get("event", "api_call")
        messages = data.get("messages", [])
        
        # Extract response and usage - check multiple possible locations
        response = data.get("response") or data.get("response_obj") or {}
        usage = data.get("usage") or {}
        
        # If response is a litellm Response object, extract usage from it
        if hasattr(response, 'usage'):
            usage = response.usage.__dict__ if hasattr(response.usage, '__dict__') else {}
        elif hasattr(response, '__dict__'):
            # Try to get usage from response object attributes
            usage = getattr(response, 'usage', usage)
        
        # Also check if usage is in the main data dict
        if not usage or (isinstance(usage, dict) and all(v == 0 for v in usage.values())):
            usage = data.get("usage", {})
        
        model_str = str(model).lower()
        if "embed" in model_str or "titan-embed" in model_str:
            call_type = "embedding"
        else:
            call_type = "chat"
        
        api_call_count[call_type] += 1
        total_call_number = sum(api_call_count.values())  # Global call number
        
        response_content = ""
        if response:
            if hasattr(response, 'choices'):
                choices = response.choices
            elif isinstance(response, dict):
                choices = response.get("choices", [])
            else:
                choices = []
            
            if choices and len(choices) > 0:
                if hasattr(choices[0], 'message'):
                    message = choices[0].message
                    response_content = getattr(message, 'content', '') or ''
                elif isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    response_content = message.get("content", "") or ""
        
        # Extract token counts - try multiple field names
        input_tokens = (
            usage.get("prompt_tokens") or 
            usage.get("input_tokens") or 
            (usage.get("usage", {}).get("prompt_tokens") if isinstance(usage.get("usage"), dict) else None) or
            0
        )
        output_tokens = (
            usage.get("completion_tokens") or 
            usage.get("output_tokens") or
            (usage.get("usage", {}).get("completion_tokens") if isinstance(usage.get("usage"), dict) else None) or
            0
        )
        total_tokens = (
            usage.get("total_tokens") or
            (input_tokens + output_tokens) or
            0
        )
        
        log_entry = {
            "timestamp": timestamp,
            "call_number": total_call_number,  # Global sequential number
            "call_type_number": api_call_count[call_type],  # Type-specific number
            "call_type": call_type,
            "event_type": event_type,
            "model": model,
            "messages": messages[:3] if len(messages) > 3 else messages,
            "response_content_preview": response_content[:500] if response_content else "",
            "usage": {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(total_tokens)
            },
            "error": str(data.get("error", "")) if data.get("error") else None
        }
        
        model_io_log_file.write(json.dumps(log_entry) + "\n")
        model_io_log_file.flush()
        
        logger.info(
            f"[Bedrock API #{total_call_number}] {call_type.upper()} | "
            f"Model: {model} | Input: {log_entry['usage']['input_tokens']} tokens | "
            f"Output: {log_entry['usage']['output_tokens']} tokens"
        )
    except Exception as e:
        logger.warning(f"Error logging API call: {e}")
        import traceback
        logger.warning(traceback.format_exc())

litellm.callbacks = [log_api_call]

# Now import GraphRAG and apply the fix
# Add src to path so graphrag can be imported (graphrag is in src/graphrag)
# This replicates the workflow from graphrag_test where graphrag is found via sys.path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import GraphRAG modules
from graphrag.language_model.providers.litellm import chat_model as litellm_chat_model
from pydantic import BaseModel
import inspect
from typing import cast

# Store original method
original_achat = litellm_chat_model.LitellmChatModel.achat

async def fixed_achat(self, prompt: str, history=None, **kwargs):
    """Fixed achat that handles JSON string parsing for findings."""
    new_kwargs = self._get_kwargs(**kwargs)
    messages = history or []
    messages.append({"role": "user", "content": prompt})
    
    response = await self.acompletion(messages=messages, stream=False, **new_kwargs)
    
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content or "",
    })
    
    parsed_response = None
    if "response_format" in new_kwargs:
        try:
            parsed_dict = json.loads(response.choices[0].message.content or "{}")
        except:
            parsed_dict = {}
        
        # FIX: If findings is a string, parse it as JSON
        if "findings" in parsed_dict and isinstance(parsed_dict["findings"], str):
            try:
                parsed_dict["findings"] = json.loads(parsed_dict["findings"])
                logger.info("✓ Fixed findings JSON string parsing")
            except Exception as e:
                logger.warning(f"Could not parse findings as JSON: {e}, using empty list")
                parsed_dict["findings"] = []
        
        parsed_response = parsed_dict
        
        if inspect.isclass(new_kwargs["response_format"]) and issubclass(new_kwargs["response_format"], BaseModel):
            model_initializer = cast("type[BaseModel]", new_kwargs["response_format"])
            try:
                parsed_response = model_initializer(**parsed_dict)
            except Exception as e:
                logger.error(f"Error creating model instance: {e}")
                # Try to fix and retry
                if "findings" in parsed_dict:
                    if isinstance(parsed_dict["findings"], str):
                        try:
                            parsed_dict["findings"] = json.loads(parsed_dict["findings"])
                            parsed_response = model_initializer(**parsed_dict)
                            logger.info("✓ Fixed and retried model creation")
                        except:
                            logger.error("Could not fix findings, using empty list")
                            parsed_dict["findings"] = []
                            parsed_response = model_initializer(**parsed_dict)
                    else:
                        raise
                else:
                    raise
    
    from graphrag.language_model.providers.litellm.chat_model import LitellmModelResponse, LitellmModelOutput
    return LitellmModelResponse(
        output=LitellmModelOutput(content=response.choices[0].message.content or ""),
        parsed_response=parsed_response,
        history=messages,
    )

# Apply the monkey patch
litellm_chat_model.LitellmChatModel.achat = fixed_achat
logger.info("✓ Applied JSON parsing fix for community reports")

logger.info("=" * 80)
logger.info("Starting GraphRAG indexing with Bedrock API monitoring")
logger.info(f"Model I/O log: {model_io_log}")
logger.info("=" * 80)

# Now run GraphRAG
try:
    from graphrag.cli.index import _run_index
    from graphrag.config.load_config import load_config
    from graphrag.config.enums import IndexingMethod
    
    # Build CLI overrides to specify input and output directories
    cli_overrides = {}
    if output_dir:
        cli_overrides["output.base_dir"] = str(output_dir)
        cli_overrides["reporting.base_dir"] = str(output_dir)
        cli_overrides["update_index_output.base_dir"] = str(output_dir)
    if input_dir:
        # Override input directory to only read from the specified folder
        # This ensures GraphRAG only reads from the keyword-specific folder
        cli_overrides["input.storage.base_dir"] = str(input_dir)
        logger.info(f"✓ Overriding input directory to: {input_dir}")
    
    # Load config with overrides
    config = load_config(
        root_dir=Path(root_dir),
        config_filepath=None,
        cli_overrides=cli_overrides
    )
    
    # Run the index with the modified config
    _run_index(
        config=config,
        method=IndexingMethod.Standard,
        is_update_run=False,
        verbose=verbose,
        memprofile=False,
        cache=True,
        dry_run=False,
        skip_validation=False,
    )
    
except Exception as e:
    logger.exception(f"Error during indexing: {e}")
    raise
finally:
    model_io_log_file.close()
    logger.info("=" * 80)
    logger.info(f"Indexing complete. Total API calls: Chat={api_call_count['chat']}, Embedding={api_call_count['embedding']}")
    logger.info(f"Model I/O log saved to: {model_io_log}")
    logger.info("=" * 80)
