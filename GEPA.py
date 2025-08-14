## Author: Emmanuel G. Maminta (egmaminta@up.edu.ph)
## GEPA-Lite: A lightweight implementation of the GEPA (Genetic-Pareto) large language model prompt optimization method

from ollama import AsyncClient
from pydantic import BaseModel, Field
from pydantic._internal._model_construction import ModelMetaclass
import ast
import random
from difflib import SequenceMatcher
from colored import Fore, Style, Back
from collections import Counter
import numpy as np
import sys
import time
import threading
import asyncio
from typing import Any, Union
import time
import yaml
import json
import textwrap
from google import genai


# === GLOBAL VARIABLES ===
STOP_SPINNER = False

with open("GEPA_cfg.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

TARGET_MODEL = cfg.get('TARGET_MODEL', 'gemma3n:e4b')
REFLECTION_MODEL = cfg.get('REFLECTION_MODEL', 'gemma3n:e4b')
USE_GEMINI_API = cfg.get('USE_GEMINI_API', False)
OLLAMA_HOST = cfg.get('OLLAMA_HOST', 'http://localhost:11434')

if USE_GEMINI_API:
    VERTEXAI = cfg.get('VERTEXAI', False)
    PROJECT = cfg.get('PROJECT', '')
    LOCATION = cfg.get('LOCATION', '')

    if (not VERTEXAI) or (not PROJECT) or (not LOCATION):
        raise ValueError("Missing required configuration for Gemini API.")

P = cfg.get('EXPLOIT_PROB', 0.95)
Q = cfg.get('MERGE_PROB', 0.90)
BUDGET = cfg.get('BUDGET', 50)
MINI_BATCH_SIZE = cfg.get('MINI_BATCH_SIZE', 2)
NUM_INITIAL_CANDIDATE_PROMPTS = cfg.get('NUM_INITIAL_CANDIDATE_PROMPTS', 6)

# === PROMPTS ===
SEED_PROMPT = """You are Grammarly. Your job is to improve writing while preserving the author's voice and intent. Focus on these specific areas:
1. CORRECTNESS: Fix grammar, spelling, punctuation, and syntax errors.
2. CLARITY: Rewrite unclear or confusing sentences. Remove wordiness and redundancy.
3. ENGAGEMENT: Improve word choice and sentence variety. Replace weak or vague language.
4. DELIVERY: Ensure appropriate tone and formality for the context.
Always preserve the author's meaning, style, and voice.
Make minimal changes - only edit what needs improvement.
Do not add new information or change the core message.
Return ONLY the corrected text with no explanations, comments, or markup."""

META_PROMPT = ("[[ ## context ## ]]\n"
               "You are the reflection model, an expert at improving large language model prompts for a specific task. "
               "You will be given:\n"
               "1. The current prompt (used by the target model for the task).\n"
               "2. A set of feedback notes summarizing where the system underperformed.\n\n"
               "[[ ## current prompt ## ]]\n"
               "{candidate}\n\n"
               "The following are examples of different task inputs provided to the assistant along "
               "with the assistant's response for each of them, and feedback on how the assistant's "
               "response could be better:\n"
               "[[ ## inputs, outputs, feedback ## ]]\n"
               "{inputs_outputs_feedback}\n\n"
               "[[ ## goal ## ]]\n"
               "Propose at most 1 new prompt that keeps what works, fixes weaknesses, improves clarity, "
               "completeness, and correctness, and is concise. Output ONLY the new improved prompt text. "
               "Do NOT generate examples from the task.")

INITIAL_CANDIDATES_GENERATION_PROMPT = ("[[ ## context ## ]]\n"
                                        "You are an expert at generating potentially optimal large language model prompts or instructions "
                                        "for a specific task. You will be given a seed prompt and sampled items from a dataset related to "
                                        "the task to help you diagnose what the task is about.\n"
                                        "[[ ## seed prompt ## ]]\n"
                                        "{seed_prompt}\n\n"
                                        "[[ ## inputs, outputs ## ]]\n"
                                        "{inputs}\n\n"
                                        "[[ ## goal ## ]]\n"
                                        "Generate a set of diverse and high-quality candidate prompts or instructions for the task "
                                        "using the seed prompt as your reference. Provide at most {num_new_prompts} new candidate prompts. "
                                        "Output ONLY the synthesized candidate prompts. "
                                        "Do NOT generate examples from the task.")

MERGING_PROMPT = ("[[ ## context ## ]]\n"
                  "You are an expert at merging text prompts. You will be given candidate prompts to synthesize:\n"
                  "[[ ## candidate prompts ## ]]\n"
                  "{candidates}\n\n"
                  "[[ ## goal ## ]]\n"
                  "Synthesize the candidate prompts into a single, coherent prompt that captures the essence of all the candidates. "
                  "The synthesis must be done such that the weaknesses of the individual candidates are addressed. Output ONLY the "
                  "synthesized candidate prompts. Output ONLY the merged candidate prompt. Do NOT generate examples from the task.")

# === BASEMODELS/MODELMETACLASSES ===
class CandidatePrompt(BaseModel):
    candidate_prompt: str = Field(..., description="The candidate prompt to be evaluated.")

class ListCandidatePrompts(BaseModel):
    list_candidate_prompts: list[CandidatePrompt] = Field(..., description="A list of candidate prompts generated by the reflection model.")

class Response(BaseModel):
    response: str = Field(..., description="The final response provided by the assistant.")


def rich_display(text: str, style: str = "info") -> str:
    styles = {
        "info": f"{Fore.CYAN}ℹ️ ",
        "success": f"{Fore.GREEN}✅",
        "warning": f"{Fore.YELLOW}⚠️ ",
        "error": f"{Fore.RED}❌",
        "scoring": f"{Fore.MAGENTA}📊",
        "comparing": f"{Fore.BLUE}🔍",
        "evaluating": f"{Fore.VIOLET}🧪",
        "rollout": f"{Back.CYAN}🔄",
    }
    print(f"{Style.BOLD}{styles.get(style, Fore.WHITE)} {text}{Style.reset}")

def eval_metric(pred: str, gt: str) -> float:
    """This function measures the quality of text editing by comparing similarity to ground truth."""
    # For text editing, we use a combination of string similarity and length preservation
    ratio = SequenceMatcher(None, pred.strip(), gt.strip()).ratio()

    # Bonus for preserving similar length (indicates minimal changes)
    len_diff = abs(len(pred) - len(gt)) / max(len(pred), len(gt), 1)
    length_bonus = max(0, 1 - len_diff)

    # Weighted score: 70% similarity, 30% length preservation
    final_score = round(0.7 * ratio + 0.3 * length_bonus, 3)
    return min(1.0, final_score)

def eval_feedback(ratio: float) -> str:
    """This function returns feedback based on the measured evaluation metric for text editing."""
    if ratio >= 0.9:
        return "Excellent editing - maintains meaning while improving clarity and correctness."
    elif ratio >= 0.7:
        return "Good editing but may have minor issues with clarity, grammar, or preservation of original style."
    elif ratio >= 0.5:
        return "Moderate editing quality. The response may have issues with:\n- Over-editing or under-editing\n- Changing the author's voice or meaning\n- Inadequate grammar/clarity improvements\n- Poor tone or formality matching"
    else:
        return "Poor editing quality. The response likely has significant issues:\n- Major changes to meaning or voice\n- Insufficient improvements to grammar/clarity\n- Inappropriate tone or style changes\n- Over-verbose explanations or markup"

async def generate_content(model: str, prompt: str, format: BaseModel) -> str:
    if isinstance(format, BaseModel) or isinstance(format, ModelMetaclass):
        format = format.model_json_schema()
    else:
        raise ValueError("Invalid format type. Format must be a Pydantic BaseModel/ModelMetaclass.")

    response = await AsyncClient(host=OLLAMA_HOST).generate(
        model=model,
        prompt=prompt,
        format=format)

    return response

if not USE_GEMINI_API:
    async def extract_response_from_target(contents: str, format: Union[BaseModel, ModelMetaclass]) -> str:
        global TARGET_MODEL
        response = await generate_content(model=TARGET_MODEL, prompt=contents, format=format)

        parsed_response_str = response.response or ""
        # Clean common wrappers / fences and stray indentation
        parsed_response_str = parsed_response_str.strip().strip("\t")
        if parsed_response_str.startswith("```"):
            # Remove possible code fences like ```json ... ```
            lines = parsed_response_str.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            parsed_response_str = "\n".join(lines).strip()

        # Try JSON first, then literal_eval; handle indentation/syntax issues
        data = None
        try:
            data = json.loads(parsed_response_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(parsed_response_str)
            except (ValueError, SyntaxError, IndentationError):
                rich_display("Failed to parse target model response; returning raw string.", "warning")
                return parsed_response_str.strip()

        if isinstance(data, dict):
            value = data.get('response', '')
            if isinstance(value, str):
                return value.strip()
        return ""

    async def extract_response_from_reflection(contents: str, format: Union[BaseModel, ModelMetaclass]) -> list[str]:
        global REFLECTION_MODEL
        response = await generate_content(model=REFLECTION_MODEL, prompt=contents, format=format)

        parsed_response_str = (response.response or "").strip()

        # Remove markdown code fences if present
        if parsed_response_str.startswith("```"):
            lines = parsed_response_str.splitlines()
            # drop first fence line
            lines = lines[1:]
            # drop last fence line if it's a fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            parsed_response_str = "\n".join(lines).strip()

        # Extra cleanup: remove leading tabs/spaces uniformly
        parsed_response_str = textwrap.dedent(parsed_response_str).strip()

        data = None
        # Try JSON first
        if parsed_response_str:
            try:
                data = json.loads(parsed_response_str)
            except json.JSONDecodeError:
                try:
                    # Fallback to ast.literal_eval; catch indentation issues and retry after another dedent
                    try:
                        data = ast.literal_eval(parsed_response_str)
                    except IndentationError:
                        cleaned = textwrap.dedent(parsed_response_str)
                        data = ast.literal_eval(cleaned)
                except (SyntaxError, ValueError, IndentationError):
                    rich_display("Reflection model response parsing failed; returning empty list.", "warning")
                    return []
        else:
            return []

        if isinstance(data, dict):
            parsed_list = data.get('list_candidate_prompts', [])
            if isinstance(parsed_list, list):
                return [item.get('candidate_prompt', '').strip()
                        for item in parsed_list
                        if isinstance(item, dict) and item.get('candidate_prompt', '') and item.get('candidate_prompt', '').strip()]
        return []

    async def extract_merged_prompt(
        for_merging_candidates: str,
        merging_prompt: str,
        format: Union[BaseModel, ModelMetaclass]
    ) -> str:
        """
        Generate a merged prompt and robustly parse the reflection model output.
        Handles:
        - Stripping code fences
        - Dedenting / removing tabs & stray spaces
        - JSON parsing first
        - Fallback to ast.literal_eval with graceful failure
        Returns an empty string if parsing fails or no candidate is found.
        """
        global REFLECTION_MODEL
        contents = merging_prompt.format(candidates=for_merging_candidates)
        response = await generate_content(model=REFLECTION_MODEL, prompt=contents, format=format)

        raw = (getattr(response, "response", "") or "").strip()

        # Remove markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            # drop first fence
            lines = lines[1:]
            # drop last fence if it's a fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines)

        # Normalize whitespace / tabs
        raw = raw.replace("\r", "")
        raw = raw.strip("\t ").strip()
        raw = textwrap.dedent(raw).strip()

        data = None
        if not raw:
            rich_display("Merged prompt response empty.", "warning")
            return ""

        # Try JSON first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback to literal_eval with extra dedent if needed
            try:
                try:
                    data = ast.literal_eval(raw)
                except IndentationError:
                    cleaned = textwrap.dedent(raw)
                    data = ast.literal_eval(cleaned)
            except (SyntaxError, ValueError, IndentationError):
                rich_display("Failed to parse merged prompt response; returning empty string.", "warning")
                return ""

        # Expect structure like {'list_candidate_prompts': [ {'candidate_prompt': '...'} ]}
        merged_prompt_list = []
        if isinstance(data, dict):
            maybe_list = data.get('list_candidate_prompts', [])
            if isinstance(maybe_list, list):
                merged_prompt_list = [
                    (item.get('candidate_prompt', '') if isinstance(item, dict) else '').strip()
                    for item in maybe_list
                ]
                merged_prompt_list = [p for p in merged_prompt_list if p]
        else:
            rich_display("Parsed merged prompt response is not a dict; returning empty string.", "warning")
            return ""

        return merged_prompt_list[0] if merged_prompt_list else ""
else:
    async def extract_response_from_target(contents: str, format: Union[BaseModel, ModelMetaclass]) -> str:
        global TARGET_MODEL, VERTEXAI, PROJECT, LOCATION
        client = genai.Client(vertexai=VERTEXAI, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=TARGET_MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": format
            }
        )
        parsed_response = response.parsed.response
        return parsed_response.strip() if parsed_response else ""

    async def extract_response_from_reflection(contents: str, format: Union[BaseModel, ModelMetaclass]) -> list[str]:
        global REFLECTION_MODEL, VERTEXAI, PROJECT, LOCATION
        client = genai.Client(vertexai=VERTEXAI, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=REFLECTION_MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": format
            }
        )
        parsed_response = response.parsed.list_candidate_prompts
        return [item.candidate_prompt.strip() for item in parsed_response if item.candidate_prompt.strip()]

    async def extract_merged_prompt(
        for_merging_candidates: str,
        merging_prompt: str,
        format: Union[BaseModel, ModelMetaclass]
    ) -> str:
        global REFLECTION_MODEL, VERTEXAI, PROJECT, LOCATION
        contents = merging_prompt.format(candidates=for_merging_candidates)
        client = genai.Client(vertexai=VERTEXAI, project=PROJECT, location=LOCATION)
        response = client.models.generate_content(
            model=REFLECTION_MODEL,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": format
            }
        )

        parsed_response = response.parsed.list_candidate_prompts
        return parsed_response[0].candidate_prompt.strip() if parsed_response else ""

async def run_with_spinner(task: callable, message: str = "Processing...") -> None:
    global STOP_SPINNER
    STOP_SPINNER = False

    def spinning_cursor():
        global STOP_SPINNER
        while not STOP_SPINNER:
            for cursor in '|/-\\':
                sys.stdout.write(f"{message} {cursor}\r")
                sys.stdout.flush()
                time.sleep(0.1)
        sys.stdout.write(' ' * (len(message) + 5) + '\r')
        sys.stdout.flush()

    spinner_thread = threading.Thread(target=spinning_cursor)
    spinner_thread.start()

    result = None
    try:
        result = await task()
    finally:
        STOP_SPINNER = True
        spinner_thread.join()

    return result

async def generate_initial_candidates_from_seed(
    seed_prompt: str,
    initial_candidates_generation_prompt: str,
    Dpareto: list[dict[str, str]],
    num_new_prompts: int = 3,
) -> list[str]:
    """
    Concurrently generate initial candidate prompts from a seed prompt.
    Instead of one request returning many, we launch num_new_prompts parallel
    single-prompt generations (each asking for at most 1 prompt) to leverage concurrency
    and improve diversity. Order is preserved (by task index) and duplicates removed.
    """
    rich_display(f"Seed prompt >> {seed_prompt}", "evaluating")
    rich_display("Generating initial candidates from seed prompt...", "rollout")

    # Build inputs string (sync, cheap)
    inputs_parts = []
    for sample in Dpareto:
        xi, gt = sample.get("question", ""), sample.get("answer", "")
        if not xi or not gt:
            rich_display("Sample is missing question or answer.", "error")
            raise ValueError("Sample is missing question or answer.")
        inputs_parts.append(f"Query: {xi}\nAnswer: {gt}\n")
    inputs = "\n".join(inputs_parts) + "\n"

    # Optional: throttle concurrent generations if needed (e.g., model rate limits)
    semaphore = asyncio.Semaphore(min(num_new_prompts, 5))

    async def fetch_one() -> str:
        async with semaphore:
            # Ask for only 1 prompt this call
            contents = initial_candidates_generation_prompt.format(
                seed_prompt=seed_prompt,
                inputs=inputs,
                num_new_prompts=1
            )
            resp = await extract_response_from_reflection(
                contents=contents,
                format=ListCandidatePrompts
            )
            # resp is a list (0 or 1 element). Return first if present.
            return resp[0] if resp else ""

    async def gather_all():
        tasks = [asyncio.create_task(fetch_one()) for _ in range(num_new_prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for r in results:
            r_stripped = r.strip()
            if r_stripped and r_stripped not in seen:
                seen.add(r_stripped)
                unique.append(r_stripped)
        return unique

    initial_candidates = await run_with_spinner(
        gather_all,
        message="Waiting to finish generating candidate prompts..."
    )

    if not initial_candidates:
        rich_display("No initial candidates were generated.", "warning")
    else:
        for i, candidate in enumerate(initial_candidates):
            rich_display(f"Candidate {i + 1} >> {candidate}", "evaluating")

    return initial_candidates


def select_candidate(P: list[str], S: np.ndarray) -> tuple[list[int], list[float]]:
    """P=prompt candidate pool; S=score matrix on Dpareto"""

    num_tasks = S.shape[1]
    num_candidates = S.shape[0]

    rich_display(f"There are {num_candidates} candidates in the pool.", "info")

    if len(P) == 1:
        return [0], [1.0]

    # Build instance-wise Pareto sets
    s_star = np.max(S, axis=0)  # Best score for each task
    P_star = [set() for _ in range(num_tasks)]

    for i in range(num_tasks):
        for j in range(num_candidates):
            if S[j, i] == s_star[i]:
                P_star[i].add(j)    # Store the index of the candidate

    # Unique candidates in union for P_star[i]
    C = set()
    for p_set in P_star:
        C.update(p_set)

    # Filter dominated candidates
    D = set()
    C_list = list(C)

    while True:
        dominated_found = False
        for idx1 in range(len(C_list)):
            if C_list[idx1] in D:
                continue
            is_dominated = False
            for idx2 in range(len(C_list)):
                if idx1 == idx2 or C_list[idx2] in D:
                    continue
                # Check if C_list[idx1] is dominated by C_list[idx2]
                # Dominated: for all tasks, S[idx1][i] <= S[idx2][i] and strict for at least one
                all_leq = all(S[C_list[idx1]][i] <= S[C_list[idx2]][i] for i in range(num_tasks))
                any_lt = any(S[C_list[idx1]][i] < S[C_list[idx2]][i] for i in range(num_tasks))
                if all_leq and any_lt:
                    is_dominated = True
                    break
            if is_dominated:
                D.add(C_list[idx1])
                dominated_found = True
        if not dominated_found:
            break

    # Remove D from each P_star[i] to get hat{P_star}[i]
    hat_P_star = [p_set - D for p_set in P_star]
    f = Counter()
    for p_set in hat_P_star:
        for k in p_set:
            f[k] += 1

    # Sample from hat_C (unique in hat_P_star) with prob proportional to f[k]
    hat_C = list(set(k for p_set in hat_P_star for k in p_set))
    if not hat_C:
        raise ValueError("No candidates available after filtering dominated candidates.")
    probs = [f[k] for k in hat_C]
    total = sum(probs)
    probs = [p / total for p in probs]

    rich_display(f"Selected {len(hat_C)} candidates from the pool.", "info")
    rich_display(f"Candidate IDs: {hat_C}", "info")
    rich_display(f"Selection probabilities: {probs}", "info")
    return hat_C, probs

async def GEPA(
    Dpareto: list[dict[str, str]],
    Dfeedback: list[dict[str, str]],
    meta_prompt: str,
    merging_prompt: str,
    eval_metric: callable,
    eval_feedback: callable,
    initial_candidates: list[str],
    budget: int = 50,
    mini_batch_size: int = 2
) -> None:
    """Returns the optimal prompt for a given task."""
    print("-" * 50)
    rich_display("Starting GEPA...", "info")
    rich_display(f"Target model: {TARGET_MODEL}\n"
                 f"Reflection model: {REFLECTION_MODEL}\n"
                 f"No. of Pareto samples: {len(Dpareto)}\n"
                 f"No. of feedback samples: {len(Dfeedback)}\n"
                 f"Mini-batch size: {mini_batch_size}\n"
                 f"Budget: {budget}", "info")
    print("-" * 50)

    candidate_prompts = []
    for ith, candidate_prompt_txt in enumerate(initial_candidates):
        candidate_prompts.append({"id": ith, "parent_id": -1, "prompt": candidate_prompt_txt,
                                  "scores": [0] * len(Dpareto), "mean_score": 0.0})

    # Initialize S-matrix
    S = np.zeros((len(candidate_prompts), len(Dpareto)), dtype=float)

    rich_display("Evaluating initial candidates...", "info")

    # Concurrent evaluation of initial candidates
    async def _eval_candidate(j: int, candidate: dict[str, Any]) -> int:
        candidate_prompt = candidate["prompt"]
        rich_display(f"Evaluating candidate (id={candidate['id']}) >> {candidate_prompt}", "evaluating")
        local_budget_used = 0
        for h, xi_gt in enumerate(Dpareto):
            xi, gt = xi_gt.get("question", ""), xi_gt.get("answer", "")
            if not xi or not gt:
                rich_display("Sample is missing question or answer.", "error")
                raise ValueError("Sample is missing question or answer.")

            contents = f"{candidate_prompt}\n\nQuery: {xi}\n"
            pred = await extract_response_from_target(contents=contents, format=Response)

            score = eval_metric(pred, gt)
            S[j, h] = score
            candidate["scores"][h] = score

            if score == 1:
                rich_display(f"Candidate (id={candidate['id']}) got sample {h + 1} right! [prediction={pred}; ground truth={gt}]", "success")
            else:
                rich_display(f"Candidate (id={candidate['id']}) got sample {h + 1} wrong. [prediction={pred}; ground truth={gt}]", "error")
            local_budget_used += 1
        else:
            candidate["mean_score"] = float(np.mean(candidate["scores"]))
            rich_display(f"Candidate (id={candidate['id']}) mean score: {candidate['mean_score']}", "scoring")

        return local_budget_used

    # Launch all candidate evaluations concurrently
    budget_used_list = await asyncio.gather(
        *[ _eval_candidate(j, candidate) for j, candidate in enumerate(candidate_prompts) ]
    )

    budget -= sum(budget_used_list)
    rich_display(f"Budget status: {budget}", "info")

    rich_display(f"Starting GEPA optimization loop with budget = {budget} left...", "info")
    while budget > 0:
        # Sample a mini-batch of feedback samples
        mini_batch_indices = random.sample(range(len(Dfeedback)), mini_batch_size)
        Dminibatch = [Dfeedback[i] for i in mini_batch_indices]

        # Exploration (1-P) vs. exploitation (P)
        # P probability to pick the best
        # 1-P probability to pick random sample
        global P
        global Q

        if random.random() <= P:
            # Choose the best based on strategy
            selected_candidate_ids, probs = select_candidate(candidate_prompts, S)
            if len(selected_candidate_ids) > 1:
                # 1-Q probability to perform candidate prompt-merge
                # Q probability to choose one based on probs
                if random.random() <= Q:
                    rich_display("[Exploit-Normal] Selecting one candidate based on probabilities...", "comparing")
                    selected_candidate_id = random.choices(selected_candidate_ids, weights=probs, k=1)[0]
                    selected_candidate = candidate_prompts[selected_candidate_id]
                else:
                    # Perform candidate prompt-merge
                    rich_display("[Exploit-Merge] Performing candidate prompt-merge on selected candidates...", "comparing")
                    candidate_parents = ()
                    for_merging_candidates = ""
                    for selected_candidate_id in selected_candidate_ids:
                        candidate_p = candidate_prompts[selected_candidate_id]
                        candidate_parents += (candidate_p["id"],)
                        for_merging_candidates += f"Candidate {candidate_p['id']} >> {candidate_p['prompt']}\n"

                    merged_prompt = await extract_merged_prompt(
                        for_merging_candidates=for_merging_candidates,
                        merging_prompt=merging_prompt,
                        format=ListCandidatePrompts
                    )
                    selected_candidate = {
                        "id": len(candidate_prompts),
                        "parent_id": candidate_parents,
                        "prompt": merged_prompt,
                        "scores": [0] * len(Dpareto),
                        "mean_score": 0.0
                    }
                    candidate_prompts.append(selected_candidate)
            else:
                rich_display("[Exploit-Max] Selecting the best candidate based on probabilities...", "comparing")
                selected_candidate = candidate_prompts[selected_candidate_ids[0]]
        else:
            # Choose a random candidate from the pool
            rich_display("[Explore] Selecting a random candidate from the pool...", "comparing")
            selected_candidate_id = random.choice(range(len(candidate_prompts)))
            selected_candidate = candidate_prompts[selected_candidate_id]

        rich_display(f"Selected candidate (id={selected_candidate['id']}) >> {selected_candidate['prompt']}", "evaluating")

        # Generate feedback for the selected candidate
        overall_feedback = ""
        selected_candidate_feedback = []
        S_feedback = []
        # --- Concurrent evaluation of selected_candidate on Dminibatch ---
        async def _eval_minibatch_sample(xi_gt):
            xi, gt = xi_gt.get("question", ""), xi_gt.get("answer", "")
            if not xi or not gt:
                raise ValueError("Sample is missing question or answer.")
            contents = f"{selected_candidate['prompt']}\n\nQuery: {xi}\n"
            pred = await extract_response_from_target(contents=contents, format=Response)
            score = eval_metric(pred, gt)
            feedback = eval_feedback(score)
            return {
                "query": xi,
                "prediction": pred,
                "ground_truth": gt,
                "score": score,
                "feedback": feedback
            }

        rich_display(f"Evaluating selected candidate (id={selected_candidate['id']}) on Dminibatch...", "info")
        minibatch_tasks = [asyncio.create_task(_eval_minibatch_sample(xi_gt)) for xi_gt in Dminibatch]
        selected_candidate_feedback = await asyncio.gather(*minibatch_tasks)

        # Budget consumed equals number of samples evaluated
        budget -= len(selected_candidate_feedback)

        # Logging per-sample results
        for i, fobj in enumerate(selected_candidate_feedback):
            if fobj["score"] == 1:
                rich_display(f"Candidate (id={selected_candidate['id']}) got sample {i + 1} right! [prediction={fobj['prediction']}; ground truth={fobj['ground_truth']}]", "success")
            else:
                rich_display(f"Candidate (id={selected_candidate['id']}) got sample {i + 1} wrong. [prediction={fobj['prediction']}; ground truth={fobj['ground_truth']}]", "error")

        S_feedback = [fobj["score"] for fobj in selected_candidate_feedback]
        mean_feedback_score = float(np.mean(S_feedback))
        rich_display(f"Candidate (id={selected_candidate['id']}) mean feedback score: {mean_feedback_score}", "scoring")

        overall_feedback = "\n".join(
            f"Query: {f['query']}\n"
            f"Assistant's Response: {f['prediction']}\n"
            f"Correct Response: {f['ground_truth']}\n"
            f"Score: {f['score']}\n"
            f"Feedback: {f['feedback']}\n"
            for f in selected_candidate_feedback
        )
        rich_display(f"Overall feedback for candidate (id={selected_candidate['id']}) generated.", "success")

        # Reflection / Mutation
        rich_display("Evolving selected prompt via reflection...", "info")
        meta_contents = meta_prompt.format(
            candidate=selected_candidate['prompt'],
            inputs_outputs_feedback=overall_feedback
        )
        reflection_candidates = await extract_response_from_reflection(contents=meta_contents, format=ListCandidatePrompts)
        if not reflection_candidates:
            rich_display("Reflection model returned no candidates. Skipping mutation...", "warning")
            continue
        new_candidate_prompt = reflection_candidates[0]
        rich_display(f"New candidate prompt generated >> {new_candidate_prompt}", "evaluating")

        # Concurrent evaluation of new candidate on Dminibatch
        async def _eval_new_minibatch_sample(xi_gt):
            xi, gt = xi_gt.get("question", ""), xi_gt.get("answer", "")
            if not xi or not gt:
                raise ValueError("Sample is missing question or answer.")
            contents = f"{new_candidate_prompt}\n\nQuery: {xi}\n"
            pred = await extract_response_from_target(contents=contents, format=Response)
            score = eval_metric(pred, gt)
            return score

        rich_display("Evaluating new candidate on Dminibatch...", "info")
        new_minibatch_tasks = [asyncio.create_task(_eval_new_minibatch_sample(xi_gt)) for xi_gt in Dminibatch]
        new_candidate_scores = await asyncio.gather(*new_minibatch_tasks)
        budget -= len(new_candidate_scores)

        for i, score in enumerate(new_candidate_scores):
            if score == 1:
                rich_display(f"New candidate got sample {i + 1} right! [prediction={score}; ground truth={Dminibatch[i]['answer']}]", "success")
            else:
                rich_display(f"New candidate got sample {i + 1} wrong. [prediction={score}; ground truth={Dminibatch[i]['answer']}]", "error")

        mean_new_candidate_score = float(np.mean(new_candidate_scores))
        rich_display(f"New candidate mean score on Dminibatch: {mean_new_candidate_score}", "scoring")

        if mean_new_candidate_score >= mean_feedback_score:
            # Concurrent evaluation on full Dpareto
            rich_display("Score improvement detected. Evaluating on Dpareto...", "info")

            async def _eval_pareto_sample(xi_gt):
                xi, gt = xi_gt.get("question", ""), xi_gt.get("answer", "")
                if not xi or not gt:
                    raise ValueError("Sample is missing question or answer.")
                contents = f"{new_candidate_prompt}\n\nQuery: {xi}\n"
                pred = await extract_response_from_target(contents=contents, format=Response)
                score = eval_metric(pred, gt)
                return score, pred

            pareto_tasks = [asyncio.create_task(_eval_pareto_sample(xi_gt)) for xi_gt in Dpareto]
            pareto_results = await asyncio.gather(*pareto_tasks)
            budget -= len(pareto_results)

            new_candidate_scores_Dpareto = []
            for j, (score, pred) in enumerate(pareto_results):
                new_candidate_scores_Dpareto.append(score)
                if score == 1:
                    rich_display(f"New candidate prompt got sample {j + 1} right! [prediction={pred}; ground truth={Dpareto[j]['answer']}]", "success")
                else:
                    rich_display(f"New candidate prompt got sample {j + 1} wrong. [prediction={pred}; ground truth={Dpareto[j]['answer']}]", "error")

            # Add new candidate prompt to pool
            new_candidate_dict = {
                "id": len(candidate_prompts),
                "parent_id": selected_candidate['id'],
                "prompt": new_candidate_prompt,
                "scores": new_candidate_scores_Dpareto,
                "mean_score": float(np.mean(new_candidate_scores_Dpareto))
            }
            candidate_prompts.append(new_candidate_dict)

            # Update S-matrix
            S = np.vstack((S, new_candidate_scores_Dpareto))
            rich_display(f"New candidate prompt (id={new_candidate_dict['id']}) added to candidate prompt pool.", "success")
            rich_display(f"New candidate prompt info :: 'parent_id': {new_candidate_dict['parent_id']}, "
                         f"'mean_score': {new_candidate_dict['mean_score']}", "info")
        else:
            rich_display("New candidate prompt did not improve over the selected candidate prompt. Skipping addition to candidate prompt pool...", "warning")

    # Display the best candidate prompt
    print("-" * 50)
    rich_display("GEPA optimization completed.", "success")
    best_candidate_id = np.argmax([c["mean_score"] for c in candidate_prompts])
    best_candidate = candidate_prompts[best_candidate_id]
    rich_display(f"Best candidate prompt (id={best_candidate['id']}) >> {best_candidate['prompt']}", "success")
    rich_display(f"Best candidate prompt info :: 'parent_id': {best_candidate['parent_id']}, "
                 f"'mean_score': {best_candidate['mean_score']}", "info")


# Wrap GEPA inside a spinner
async def run_gepa(
    Dpareto: list[dict[str, str]],
    Dfeedback: list[dict[str, str]],
    meta_prompt: str,
    merging_prompt: str,
    eval_metric: callable,
    eval_feedback: callable,
    initial_candidates: list[str],
    budget: int = 50,
    mini_batch_size: int =2
) -> None:
    async def _gepa():
        await GEPA(
            Dpareto=Dpareto,
            Dfeedback=Dfeedback,
            meta_prompt=meta_prompt,
            merging_prompt=merging_prompt,
            eval_metric=eval_metric,
            eval_feedback=eval_feedback,
            initial_candidates=initial_candidates,
            budget=budget,
            mini_batch_size=mini_batch_size
        )

    start_time = time.time()

    await run_with_spinner(
        task=_gepa,
        message="Waiting to finish GEPA optimization..."
    )

    elapsed_time = time.time() - start_time

    rich_display(f"GEPA optimization completed in {elapsed_time:.2f} seconds.", "info")


async def main():
    global META_PROMPT, MERGING_PROMPT, BUDGET, MINI_BATCH_SIZE, NUM_INITIAL_CANDIDATE_PROMPTS

    with open("Dpareto.json", "r") as fPareto:
        Dpareto = json.load(fPareto)

    with open("Dfeedback.json", "r") as fFeedback:
        Dfeedback = json.load(fFeedback)

    initial_candidates = await generate_initial_candidates_from_seed(
        seed_prompt=SEED_PROMPT,
        initial_candidates_generation_prompt=INITIAL_CANDIDATES_GENERATION_PROMPT,
        Dpareto=Dpareto,
        num_new_prompts=NUM_INITIAL_CANDIDATE_PROMPTS
    )

    await run_gepa(
        Dpareto=Dpareto,
        Dfeedback=Dfeedback,
        meta_prompt=META_PROMPT,
        merging_prompt=MERGING_PROMPT,
        eval_metric=eval_metric,
        eval_feedback=eval_feedback,
        initial_candidates=initial_candidates,
        budget=BUDGET,
        mini_batch_size=MINI_BATCH_SIZE
    )

if __name__ == "__main__":
    asyncio.run(main())
