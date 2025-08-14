# GEPA-Lite: Let the LLM reflect and optimize its own prompts.
`#ForTheLoveOfCode`

**GEPA-Lite** is a lightweight implementation based on the proposed [GEPA prompt optimization method](https://arxiv.org/pdf/2507.19457) that is custom fit for single-task applications. It's built on the **core principle of LLM self-reflection, self-improvement, streamlined**.

This implementation is configured for **Grammarly-style text editing optimization**, demonstrating how GEPA can be used to evolve and improve writing assistant prompts through iterative refinement.

Developed in the spirit of open-source initiatives like `Google Summer of Code 2025` and `For the Love of Code 2025`, this project leverages **Gemma** (`ollama::gemma3n:e4b`) as its core model. The project also offers optional support for the **Gemini API**, allowing access to powerful models like `gemini-2.5-flash-lite`, `gemini-2.5-flash`, and `gemini-2.5-pro`.

Created by: *Emmanuel G. Maminta<sup>*</sup>* (`GitHub: egmaminta`, [`LinkedIn: egmaminta`](https://ph.linkedin.com/in/egmaminta), [`Personal Blog: egmaminta.github.io`](https://egmaminta.github.io/), `Email: egmaminta@up.edu.ph`)

<sup>*</sup>University of the Philippines

#### Demonstration
<p align="center" width="100%">
   <img src="demo.gif" alt="demo" />
   <p align="justify" width="100%">
      GEPA-Lite demonstration on text editing optimization. <b>Initial prompt</b>: <i>"You are Grammarly. Your job is to improve writing while preserving the author's voice and intent..."</i> The system iteratively evolves this prompt to maximize text editing quality while maintaining the author's original style and meaning.
   </p>
</p>

#### Four strategies implemented
| Strategy | Trigger condition | Action taken | Goal |
| -------- | ----------------- | ------------ | ---- |
| Exploit-Max | Only one best candidate remains | Select it directly | Focus on clear winner |
| Exploit-Normal | Multiple best, Q branch | Sample one by frequency as "best" | Exploit robust, multi-task candidates
| Exploit-Merge | Multiple best, (1-Q) branch | Merge (mutate) all bests into a new candidate | Synthesize new, possibly better prompt
| Explore | With probability (1-P) | Pick a random candidate from the pool | Maintain diversity, avoid local optima

Note: `P`: Exploit probability, `Q`: Probability to not merge (or not mutate) all bests

Disclaimer: This implementation contains minor deviations from the reference paper. I, the creator of this repository, am held accountable for any part that is implemented incorrectly. <ins>However, the idea remains the same.</ins> Feel free to contribute!

## Reference paper
Title: [**GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**](https://arxiv.org/pdf/2507.19457)

Authors: *Lakshya A Agrawal<sup>1</sup>*, *Shangyin Tan<sup>1</sup>*, *Dilara Soylu<sup>2</sup>*, *Noah Ziems<sup>4</sup>*, *Rishi Khare<sup>1</sup>*, *Krista Opsahl-Ong<sup>5</sup>*, *Arnav Singhvi<sup>2,5</sup>*, *Herumb Shandilya<sup>2</sup>*, *Michael J Ryan<sup>2</sup>*, *Meng Jiang<sup>4</sup>*, *Christopher Potts<sup>2</sup>*, *Koushik Sen<sup>1</sup>*, *Alexandros G. Dimakis<sup>1,3</sup>*, *Ion Stoica<sup>1</sup>*, *Dan Klein<sup>1</sup>*, *Matei Zaharia<sup>1,5</sup>*, *Omar Khattab<sup>6</sup>*

<sup>1</sup>UC Berkeley, <sup>2</sup>Stanford University, <sup>3</sup>BespokeLabs.ai, <sup>4</sup>Notre Dame, <sup>5</sup>Databricks, <sup>6</sup>MIT

TL;DR: GEPA is introduced, which utilizes natural language reflection to learn from trial and error. GEPA outperforms the reinforcement learning method GRPO (Group Relative Policy Optimization) by an average of 10% and up to 20%, while using up to 35 times fewer rollouts. It also surpasses the leading prompt optimizer, MIPROv2, by over 10%. Additionally, GEPA shows potential as an effective inference-time search strategy for code optimization.

## About the implementation
This implementation performs iterative prompt evolution using concurrent generation, evaluation, Pareto-style selection, reflection-based mutation, and optional merging. It is fully asynchronous around model I/O to maximize throughput under a fixed evaluation budget.

#### High-level flow
1. Load config (`GEPA_cfg.yaml`) and datasets (`Dpareto.json`, `Dfeedback.json`).
2. Generate an initial pool of diverse candidate prompts concurrently from a seed prompt.
3. Evaluate each candidate over the full Pareto set (`Dpareto`) to build a score matrix S (candidates × tasks).
4. Enter the optimization loop (budget-limited):
   - Sample a mini-batch of feedback examples (`Dfeedback`).
   - Choose a candidate via exploitation (probability P) or random exploration (1−P).
   - Under exploitation:
       - Use Pareto-based filtering + dominance removal to get a set of non-dominated "best-on-something" candidates.
       - With probability Q: sample one candidate proportional to its frequency of being per-task best.
       - With probability (1−Q): merge multiple top candidates into a synthesized prompt.
   - Evaluate the chosen prompt on the mini-batch, collect structured feedback text.
   - Reflect (mutation step): ask the reflection model to propose an improved prompt.
   - Evaluate the new prompt on the same mini-batch; if it meets or exceeds the parent’s mean mini-batch score, evaluate it on the full Pareto set and append to the pool.
5. After budget exhaustion, report the prompt with highest mean Pareto score.

#### Data roles
- `Dpareto`: Ground-truth QA pairs used for stable, comparable scoring across candidates.
- `Dfeedback`: Additional sampled items for rapid, lower-cost iterative refinement.
- `Scores`: String similarity via `difflib.SequenceMatcher` (ratio ∈ [0,1]); exact matches earn 1. (You have the option to implement your own `eval_metric` and `eval_feedback` functions).

#### Target and reflection model
- Target model is the worker that actually answers the user's questions. It runs candidate prompts and produces task outputs that get scored.
- Reflection model is the coach that looks at the worker's prompt plus feedback and suggests an improved prompt. It never answers the task directly; it just proposes new or merged prompts.

#### Candidate selection (Pareto filtering)
- For each task (column), collect candidates achieving the column max.
- Union these sets → initial elite pool.
- Remove dominated candidates: a candidate dominated if another is ≥ on all tasks and > on at least one.
- Count how often each survivor is task-best; convert frequencies to a sampling distribution.
- This concentrates probability mass on genuinely distinct trade-offs and eliminates strictly inferior prompts.

#### Reflection & mutation
- A meta prompt (`META_PROMPT`) injects the current prompt and structured per-sample feedback (query, model answer, ground truth, score, textual feedback).
- The reflection model returns at most one improved prompt (keeps loop tight, reduces drift).
- Only promotes a mutant to global evaluation if it does not regress on the mini-batch.

#### Merging
When multiple strong candidates exist, a merge step requests a synthesized prompt that preserves strengths and addresses weaknesses (controlled by probability branch).

#### Four strategies used in the GEPA optimization loop
1. `Exploit-Max`
    - When: Only one candidate survives Pareto filtering (i.e., it is the clear best).
    - What happens: The algorithm selects this single best candidate deterministically for further evaluation and mutation.
    - Purpose: Quickly exploit a clear winner, focusing resources on refining the top performer.
2. `Exploit-Normal`
    - When: Multiple non-dominated candidates survive, and a random draw (with probability Q) chooses not to merge.
    - What happens: One candidate is sampled from the survivors, with probability proportional to how often each is "best" across tasks.
    - Purpose: Exploit strong candidates while maintaining diversity, allowing the process to focus on robust prompts that perform well on multiple tasks.
3. `Exploit-Merge`
    - When: Multiple non-dominated candidates survive, and a random draw (with probability 1-Q) chooses to merge.
    - What happens: The algorithm synthesizes a new prompt by merging the surviving candidates, aiming to combine their strengths and address their weaknesses.
    - Purpose: Encourage innovation and avoid stagnation by creating new, potentially superior prompts from the best available options.
4. `Explore`
    - When: With probability (1-P), the algorithm chooses to explore rather than exploit.
    - What happens: A random candidate is selected from the entire pool, regardless of its current performance.
    - Purpose: Maintain diversity, prevent premature convergence, and allow the discovery of overlooked or novel solutions.

#### Concurrency
- Async tasks evaluate prompts across samples in parallel (per candidate and per mini-batch mutation stage).
- A semaphore throttles concurrent initial generations to respect model rate limits.

#### Key functions
- `generate_initial_candidates_from_seed`: Concurrent single-prompt generations for diversity + deduplication.
- `select_candidate`: Pareto elite extraction, dominance pruning, probability assignment.
- `extract_response_from_target` / `extract_response_from_reflection`: Structured JSON-like parsing via `Pydantic` schema.
- `GEPA`: Orchestrates the evolutionary loop with budget accounting.
- `extract_merged_prompt`: Synthesizes multiple prompts into one.
- `eval_metric` / `eval_feedback`: Scoring + diagnostic feedback synthesis.

#### Budget control
Every model query against a sample decrements the global budget. Full-set evaluations are only triggered when a mutation shows promise on the mini-batch, amortizing expensive scoring.

#### Design rationale
- Frequency weighting of per-task best appearances yields a simple proxy for multi-task robustness without solving a weighted scalarization.
- Using an explicit merge path combats stagnation when multiple partial specialists exist.
- Mini-batch gating reduces variance while avoiding full-set overfitting each iteration.
- Limiting reflection to one improvement per cycle simplifies acceptance logic and traceability.

#### Limitations / future improvements
- Similarity metric is surface-level; could plug in semantic or task-specific evaluators.
- Needs epsilon-based comparisons for floating scores to avoid brittle equality.
- Dominance check is O(k^2 * tasks); could be optimized for large pools.
- No archival mechanism (e.g., hall-of-fame) or diversity penalty to prevent convergence to local optima.
- Reflection may benefit from multiple candidates with bandit selection.
- Logging could be made structured (JSON) for analytics.

#### Extensibility points
- Swap `eval_metric` for domain-specific scoring.
- Add alternative selection policies (e.g., Thompson sampling on mean + variance).
- Introduce early stopping when no improvement in N consecutive accepted mutations.
- Integrate caching for repeated model queries on unchanged (prompt, query) pairs.

## General instructions
1. Ensure you have `miniconda` / `anaconda` installed. Create a virtual environment with Python ver. 3.11.13.
```
conda create -n "GEPA-Lite" python=3.11.13 -y
```
2. Activate the virtual environment.
```
conda activate GEPA-Lite
```
3. Install the required libraries.
```
pip install -r requirements.txt
```

## Configuration instructions
1. Open `GEPA_cfg.yml`.
2. **(Gemini API)** Replace placeholder `PROJECT` and `LOCATION` if `VERTEXAI: true`; else optionally set `VERTEXAI: false`. **(Ollama API)** Replace `TARGET_MODEL` and `REFLECTION_MODEL` to Ollama-provided `gemma3n:e4b` model.
3. Adjust `BUDGET` ensuring it's ≥ `(NUM_INITIAL_CANDIDATE_PROMPTS * |Dpareto|) + (MINI_BATCH_SIZE * 5 * |Dpareto|) + safety margin (~30%)`.
4. Tune `EXPLOIT_PROB` and `MERGE_PROB` based on desired exploration.
5. Save and re-run the main script (`GEPA.py`).

## Dpareto and Dfeedback instructions
1. Purpose
    - `Dpareto.json`: Stable evaluation set. Used to compute full score vectors per candidate prompt (deterministic performance basis).
    - `Dfeedback.json`: Fast adaptation pool. Small random mini-batches sampled each iteration to generate feedback and mutate prompts.
3. Expected schema. Required keys: `question`, `answer` (non-empty strings). Avoid extra keys unless you extend code to read them.
```
[
  {
    "question": "string input or query",
    "answer": "ground truth string answer"
  },
  ...
]
```
3. Choosing content
    - `Dpareto.json`: Must be representative, cover all subtopics / formats / difficulty tiers. It must vary in length, phrasing, edge cases (numbers, units, multi-step reasoning). Do not modify mid-run.
    - `Dfeedback.json`: Larger pool than `MINI_BATCH_SIZE` (≥ twice the size of `Dpareto`). Higher variability, includes tricky and failure-prone cases. Can include some overlap with `Dpareto`, but prefer mostly distinct to broaden learning signals.
4. Sizing guidelines
    - `Dpareto` size: Small tasks (QA-like) 15-40; moderate 40-100. Larger increases cost linearly.
    - `Dfeedback` size: 2–5× `Dpareto` or at least 50 if you want variety; `minimum = MINI_BATCH_SIZE * 2 * |Dpareto|`.
    - Budget sanity: `Initial cost = NUM_INITIAL_CANDIDATE_PROMPTS * |Dpareto|`. Each accepted new candidate costs another `|Dpareto|`. Ensure `BUDGET ≥ initial + (expected improvements * |Dpareto|) + iteration minibatch costs`.
5. Difficulty / Distribution Strategy
    - Create a difficulty split: easy / medium / hard (e.g., 40/40/20). Hard items expose weaknesses; easy items stabilize averages.
    - Ensure each sub-domain appears at least twice in `Dpareto` to reduce variance.
    - Put rare but critical edge cases (formatting, units, rounding) in `Dpareto` so they can influence selection pressure.
6. Workflow to build `Dpareto`
    1. Gather raw candidate pairs.
    2. Deduplicate (case-insensitive question hash).
    3. Remove ambiguous items.
    4. Normalize answers.
    5. Tag coverage (spread topics).
    6. Shuffle with fixed seed; pick top N.
    7. Save to `Dpareto.json` and freeze (commit with version tag).
7. Workflow to build `Dfeedback`
    1. Start with remaining pool not used in Dpareto.
    2. Inject a few "stress" / adversarial items (format traps, long inputs).
    3. Ensure at least `MINI_BATCH_SIZE * 2 * |Dpareto|` total.
    4. Optionally rotate / refresh between runs (version the file).

## About this Grammarly Implementation
This configuration optimizes prompts for text editing tasks using the real **Grammarly CoEdit dataset** from Hugging Face, specifically:
- **Grammar and spelling correction**
- **Clarity improvement**
- **Style preservation**
- **Conciseness enhancement**

#### Data source
The implementation automatically downloads and processes the [Grammarly CoEdit dataset](https://huggingface.co/datasets/grammarly/coedit) (69K instances) from Hugging Face. The dataset contains real text editing examples with:
- Source text with grammatical errors or unclear phrasing
- Target text with professional edits maintaining original meaning

#### Data roles for text editing
- `Dpareto.json`: Contains 20 representative text editing pairs from CoEdit for stable evaluation
- `Dfeedback.json`: Contains 40 additional CoEdit examples for iterative refinement during optimization
- **Automatic loading**: If these files don't exist, the system automatically downloads and processes the CoEdit dataset

#### Evaluation metrics
- **String similarity**: Measures how closely the AI's edits match expert edits from the CoEdit dataset
- **Length preservation**: Rewards minimal, targeted changes over extensive rewrites
- **Combined score**: 70% similarity + 30% length preservation bonus

#### Getting started with CoEdit data
1. Install dependencies: `pip install -r requirements.txt` (includes `datasets` library)
2. Run the optimizer: `python GEPA.py`
3. The system will automatically download CoEdit data on first run
4. Or manually prepare data: `python load_coedit_data.py`

## Acknowledgments
I'd like to acknowledge the authors of GEPA paper for their awesome work; the team (developers, engineers, and scientists) behind [Gemma 3n](https://deepmind.google/models/gemma/gemma-3n/), [Gemini](https://deepmind.google/models/gemini/) for their powerful models; and [Google Summer of Code 2025](https://summerofcode.withgoogle.com/) and [For the Love of Code 2025](https://github.blog/open-source/for-the-love-of-code-2025/) for inspiring me to continue contributing to open-source. `#ForTheLoveOfCode`
