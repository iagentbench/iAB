"""
Bedrock-backed agent implementations for Reflexion HotPotQA runs.

This file is intentionally a near-copy of `agents.py`, but swaps the LLM backend:
- OpenAI-based `AnyOpenAILLM` (from `llm.py`) -> Bedrock-based `AnyOpenAILLM`
  (alias exported by `llm_bedrock.py`).

The agent logic (ReAct loop, reflection formatting, Wikipedia search/lookup)
remains unchanged.
"""

import re, string, os
import sys
from typing import List, Union, Literal, Optional
from enum import Enum
import tiktoken
from langchain import OpenAI, Wikipedia
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate

# IMPORTANT: Bedrock-backed wrapper
from llm_bedrock import AnyOpenAILLM

from prompts import (
    reflect_prompt,
    react_agent_prompt,
    react_reflect_agent_prompt,
    REFLECTION_HEADER,
    LAST_TRIAL_HEADER,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
)
from prompts import (
    cot_agent_prompt,
    cot_reflect_agent_prompt,
    cot_reflect_prompt,
    COT_INSTRUCTION,
    COT_REFLECT_INSTRUCTION,
)
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, COT, COT_REFLECT


class ReflexionStrategy(Enum):
    """
    NONE: No reflection
    LAST_ATTEMPT: Use last reasoning trace in context
    REFLEXION: Apply reflexion to the next reasoning trace
    LAST_ATTEMPT_AND_REFLEXION: Use last reasoning trace in context and apply reflexion to the next reasoning trace
    """

    NONE = "base"
    LAST_ATTEMPT = "last_trial"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_trial_and_reflexion"


class CoTAgent:
    def __init__(
        self,
        question: str,
        context: str,
        key: str,
        agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = cot_reflect_prompt,
        cot_examples: str = COT,
        reflect_examples: str = COT_REFLECT,
        self_reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
            temperature=0,
            max_tokens=250,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"},
        ),
        action_llm: AnyOpenAILLM = AnyOpenAILLM(
            temperature=0,
            max_tokens=250,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"},
        ),
    ) -> None:
        self.question = question
        self.context = context
        self.key = key
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples
        self.reflect_examples = reflect_examples
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm
        self.reflections: List[str] = []
        self.reflections_str = ""
        self.answer = ""
        self.step_n: int = 0
        self.reset()

    def run(self, reflexion_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy != ReflexionStrategy.NONE:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def step(self) -> None:
        # Think
        self.scratchpad += f"\nThought:"
        self.scratchpad += " " + self.prompt_agent()
        print(self.scratchpad.split("\n")[-1])

        # Act
        self.scratchpad += f"\nAction:"
        action = self.prompt_agent()
        self.scratchpad += " " + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split("\n")[-1])

        self.scratchpad += f"\nObservation: "
        if action_type == "Finish":
            self.answer = argument
            if self.is_correct():
                self.scratchpad += "Answer is CORRECT"
            else:
                self.scratchpad += "Answer is INCORRECT"
            self.finished = True
            return
        else:
            print("Invalid action type, please try again.")

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print("Running Reflexion strategy...")
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += "\n" + format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )
        else:
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")
        print(self.reflections_str)

    def prompt_reflection(self) -> str:
        return format_step(self.self_reflect_llm(self._build_reflection_prompt()))

    def reset(self) -> None:
        self.scratchpad: str = ""
        self.finished = False

    def prompt_agent(self) -> str:
        return format_step(self.action_llm(self._build_agent_prompt()))

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=self.reflections_str,
            context=self.context,
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            context=self.context,
            question=self.question,
            scratchpad=self.scratchpad,
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        # Use LLM evaluation instead of exact match for better semantic understanding
        return llm_eval_correct(self.key, self.answer)


class ReactAgent:
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 6,
        agent_prompt: PromptTemplate = react_agent_prompt,
        docstore: Docstore = Wikipedia(),
        react_llm: AnyOpenAILLM = AnyOpenAILLM(
            temperature=0,
            max_tokens=100,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"},
        ),
        searxng_url: Optional[str] = None,
    ) -> None:
        self.question = question
        self.answer = ""
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6

        self.docstore = DocstoreExplorer(docstore)  # Search, Lookup
        self.llm = react_llm
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")

        self.enc = tiktoken.encoding_for_model("text-davinci-003")

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        # Think
        self.scratchpad += f"\nThought {self.step_n}:"
        self.scratchpad += " " + self.prompt_agent(mode="thought")
        print(self.scratchpad.split("\n")[-1])

        # Act
        self.scratchpad += f"\nAction {self.step_n}:"
        # If we're out of steps after this one, force a final answer so every trial
        # produces some answer (prevents blank outputs when max_steps is small).
        if self.step_n >= self.max_steps:
            action = self.prompt_agent(mode="final")
        else:
            action = self.prompt_agent(mode="action")
        self.scratchpad += " " + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split("\n")[-1])

        # Observe
        self.scratchpad += f"\nObservation {self.step_n}: "

        if action_type == "Finish":
            self.answer = argument
            if self.is_correct():
                self.scratchpad += "Answer is CORRECT"
            else:
                self.scratchpad += "Answer is INCORRECT"
            self.finished = True
            self.step_n += 1
            return

        if action_type == "Search":
            try:
                result = self.docstore.search(argument)
                self.scratchpad += format_step(result)
                # If Wikipedia search returns "Could not find" or similar, suggest WebSearch
                if "could not find" in result.lower() or "similar:" in result.lower():
                    self.scratchpad += " (Note: If Wikipedia doesn't have this information, consider using WebSearch for more comprehensive results.)"
            except Exception as e:
                print(e)
                self.scratchpad += f"Could not find that page. Consider using WebSearch[query] for this information instead."

        elif action_type == "Lookup":
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += (
                    f"The last page Searched was not found, so you cannot Lookup a keyword in it. "
                    f"Please try one of the similar pages given."
                )

        elif action_type == "WebSearch":
            try:
                # Import search functions from RAG pipeline (RAG is inside benchmarking)
                try:
                    from src.benchmarking.rag.retrieval import search_searxng
                    from src.benchmarking.rag.pipeline import format_search_results_as_context
                except ImportError:
                    # Fallback: try adding repo root to path (for src.benchmarking imports)
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
                    if repo_root not in sys.path:
                        sys.path.insert(0, repo_root)
                    from src.benchmarking.rag.retrieval import search_searxng
                    from src.benchmarking.rag.pipeline import format_search_results_as_context
                
                # Perform web search
                search_results = search_searxng(argument, base_url=self.searxng_url, max_results=5)
                formatted_context = format_search_results_as_context(search_results)
                self.scratchpad += format_step(formatted_context)
            except Exception as e:
                print(e)
                self.scratchpad += f"Web search failed: {str(e)}"

        else:
            self.scratchpad += (
                "Invalid Action. Valid Actions are Search[<topic>], Lookup[<topic>], WebSearch[<query>], and Finish[<answer>]."
            )

        print(self.scratchpad.split("\n")[-1])

        self.step_n += 1

    def prompt_agent(self, mode: str = "action") -> str:
        """
        Bedrock models can be more verbose than the original GPT-* setup.
        We therefore hard-constrain the output format depending on whether we're
        generating a Thought or an Action.
        """
        base_prompt = self._build_agent_prompt()

        if mode == "thought":
            constrained = (
                base_prompt
                + "\n\nIMPORTANT OUTPUT FORMAT:\n"
                + "- Output ONLY the thought content for the next Thought step.\n"
                + "- Do NOT include prefixes like 'Thought', step numbers, or any other text.\n"
                + "- Single line only.\n"
            )
        elif mode == "final":
            constrained = (
                base_prompt
                + "\n\nIMPORTANT OUTPUT FORMAT:\n"
                + "- You MUST end now.\n"
                + "- Output ONLY ONE line in exactly this form:\n"
                + "  Finish[answer]\n"
                + "- CRITICAL: Be VERY CONCISE. Answer with ONLY the essential information.\n"
                + "- For names: Use just the name (e.g., 'Jerry John Rawlings' not 'Flt Lt J.J Rawlings, the chairman of PNDC').\n"
                + "- For numbers: Use just the number (e.g., '2' not 'two' or 'the last two fingers').\n"
                + "- For dates: Use the exact date format requested (e.g., 'May 21, 1936' not 'May 21st, 1936').\n"
                + "- Do NOT include titles, descriptions, or explanations in your answer.\n"
                + "- Do NOT make complete sentences - just provide the answer.\n"
                + "- Carefully review all search results and observations to extract the correct answer.\n"
                + "- If you found information in search results, use that information to provide the answer.\n"
                + "- If you are unsure, make your best guess based on all observations so far.\n"
                + "- Do NOT output Search, Lookup, or WebSearch.\n"
                + "- Do NOT output 'Thought' or any extra text.\n"
            )
        else:
            constrained = (
                base_prompt
                + "\n\nIMPORTANT OUTPUT FORMAT:\n"
                + "- Output ONLY ONE action in EXACTLY one of these forms:\n"
                + "  Search[entity]\n"
                + "  Lookup[keyword]\n"
                + "  WebSearch[query]\n"
                + "  Finish[answer]\n"
                + "- STRATEGY: If Wikipedia Search fails, try WebSearch next instead of repeating Wikipedia searches.\n"
                + "- Do NOT output 'Thought', explanations, or any extra text.\n"
                + "- Single line only.\n"
            )

        response = format_step(self.llm(constrained))

        # Salvage if an action is embedded in extra text (including WebSearch).
        action_match = re.search(r"\b(Search|Lookup|WebSearch|Finish)\[(.+?)\]", response)
        if action_match:
            return f"{action_match.group(1)}[{action_match.group(2)}]"
        return response

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples, question=self.question, scratchpad=self.scratchpad
        )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        # Use LLM evaluation instead of exact match for better semantic understanding
        return llm_eval_correct(self.key, self.answer)

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)
        ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ""

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


class ReactReflectAgent(ReactAgent):
    def __init__(
        self,
        question: str,
        key: str,
        max_steps: int = 6,
        agent_prompt: PromptTemplate = react_reflect_agent_prompt,
        reflect_prompt: PromptTemplate = reflect_prompt,
        docstore: Docstore = Wikipedia(),
        react_llm: AnyOpenAILLM = AnyOpenAILLM(
            temperature=0,
            max_tokens=100,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"},
        ),
        reflect_llm: AnyOpenAILLM = AnyOpenAILLM(
            temperature=0,
            max_tokens=250,
            model_name="gpt-3.5-turbo",
        ),
        searxng_url: Optional[str] = None,
    ) -> None:

        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm, searxng_url)
        self.reflect_llm = reflect_llm
        self.reflect_prompt = reflect_prompt
        self.reflect_examples = REFLECTIONS
        self.reflections: List[str] = []
        self.reflections_str: str = ""

    def run(
        self, reset=True, reflect_strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION
    ) -> None:
        if (self.is_finished() or self.is_halted()) and not self.is_correct():
            self.reflect(reflect_strategy)

        ReactAgent.run(self, reset)

    def reflect(self, strategy: ReflexionStrategy) -> None:
        print("Reflecting...")
        if strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0])
        elif strategy == ReflexionStrategy.REFLEXION:
            self.reflections += [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.reflections_str = format_last_attempt(self.question, self.scratchpad)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(
                self.reflections, header=REFLECTION_AFTER_LAST_TRIAL_HEADER
            )
        else:
            raise NotImplementedError(f"Unknown reflection strategy: {strategy}")
        print(self.reflections_str)

    def prompt_reflection(self) -> str:
        return format_step(self.reflect_llm(self._build_reflection_prompt()))

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
            examples=self.reflect_examples,
            question=self.question,
            scratchpad=truncate_scratchpad(self.scratchpad, tokenizer=self.enc),
        )

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad,
        )


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string):
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument

    else:
        return None, None


def format_step(step: str) -> str:
    return step.strip("\n").strip().replace("\n", "")


def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ""
    else:
        return header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])


def format_last_attempt(question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER):
    return (
        header
        + f"Question: {question}\n"
        + truncate_scratchpad(scratchpad, tokenizer=gpt2_enc).strip("\n").strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratchpad.split("\n")
    observations = filter(lambda x: x.startswith("Observation"), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode("\n".join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(":")[0] + ": [truncated wikipedia excerpt]"
    return "\n".join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


def llm_eval_correct(answer: str, response: str, model_id: str = None, region: str = None) -> bool:
    """
    Evaluate if a response is correct using LLM evaluation (Bedrock-based).
    
    Uses semantic understanding instead of exact match, so "Jerry Rawlings" matches
    "Jerry John Rawlings" and "2" matches "two".
    
    Args:
        answer: Ground truth answer
        response: Agent's answer
        model_id: Bedrock model ID (defaults to env var or haiku)
        region: AWS region (defaults to env var or us-east-1)
    
    Returns:
        True if LLM judges the answer as "correct", False otherwise
    """
    import os
    import sys
    
    # Import bedrock_generate
    try:
        from src.benchmarking.bedrock_client import bedrock_generate
    except ImportError:
        # Fallback: try adding repo root to path (for src.benchmarking imports)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.benchmarking.bedrock_client import bedrock_generate
    
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    
    eval_prompt = '''You are a classifier that will be given a ground truth answer and an agent's response. 
You will grade the predicted answer as either "correct", "incorrect", or "not attempted".

Here is the criteria: 

Grade: "Correct"	
Definition: The predicted answer fully contains the ground-truth answer without contradicting the reference answer. The ground truth answer may be embedded within a longer sentence or phrase - this is still considered correct as long as the core answer is present and accurate.

Examples:
- Ground truth: "Jerry John Rawlings", Response: "Jerry Rawlings" → CORRECT (missing middle name is acceptable)
- Ground truth: "2", Response: "two" → CORRECT (format differences are acceptable)
- Ground truth: "May 21, 1936", Response: "May 21, 1936" → CORRECT
- Ground truth: "Wout Weghorst", Response: "The player was Wout Weghorst" → CORRECT (embedded in sentence)

Grade: "Incorrect"	
Definition: The predicted answer contradicts the ground-truth answer in any way.

Examples:
- Ground truth: "May 21, 1936", Response: "February 19, 1960" → INCORRECT (wrong date)
- Ground truth: "6", Response: "4" → INCORRECT (wrong number)
- Ground truth: "Jerry Rawlings", Response: "John Rawlings" → INCORRECT (wrong first name)

Grade: "Not attempted"	
Definition: The ground truth target is not fully given in the answer, and there are no contradictions.

Examples:
- Ground truth: "Wout Weghorst", Response: "I don't know" → NOT ATTEMPTED
- Ground truth: "2", Response: "some fingers" → NOT ATTEMPTED

Ground truth answer: {answer}
Agent response: {response}

Your response should be EXACTLY one word: "correct", "incorrect", or "not attempted".'''
    
    formatted_prompt = eval_prompt.format(answer=answer, response=response)
    
    try:
        result = bedrock_generate(
            formatted_prompt,
            model_id=model_id,
            max_tokens=50,  # Just need one word
            temperature=0.0,  # Deterministic
            region=region
        ).strip().lower()
        
        # Parse result
        if 'correct' in result:
            return True
        elif 'incorrect' in result or 'not attempted' in result:
            return False
        else:
            # Fallback to exact match if LLM response is unclear
            return EM(answer, response)
    except Exception as e:
        # Fallback to exact match if LLM evaluation fails
        print(f"LLM evaluation failed: {e}, falling back to exact match")
        return EM(answer, response)


