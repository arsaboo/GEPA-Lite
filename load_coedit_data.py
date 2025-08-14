"""
Script to load and process the Grammarly CoEdit dataset from Hugging Face.
Creates Dpareto.json and Dfeedback.json files for GEPA optimization.
"""

import json
import random
from datasets import load_dataset
from colored import Fore, Style


def extract_input_text(src: str) -> str:
    """Extract the input text from the formatted source string."""
    # Remove the instruction prefix (e.g., "Improve the grammaticality: ")
    if ": " in src:
        return src.split(": ", 1)[1]
    return src


def load_and_process_coedit(
    pareto_size: int = 50,      # Increased from 20
    feedback_size: int = 100,   # Increased from 50
    seed: int = 42,
    max_samples: int = 5000     # Increased from 1000
) -> tuple[list[dict], list[dict]]:
    """
    Load CoEdit dataset and create Dpareto and Dfeedback datasets.

    Args:
        pareto_size: Number of samples for Dpareto.json
        feedback_size: Number of samples for Dfeedback.json
        seed: Random seed for reproducibility
        max_samples: Maximum samples to load from full dataset (for speed)

    Returns:
        Tuple of (Dpareto_data, Dfeedback_data)
    """
    print(f"{Fore.cyan}📚 Loading CoEdit dataset from Hugging Face...{Style.reset}")

    # Load the dataset
    dataset = load_dataset("grammarly/coedit", split="train")

    # Convert to list and sample if dataset is large
    data_list = list(dataset)
    if len(data_list) > max_samples:
        random.seed(seed)
        data_list = random.sample(data_list, max_samples)

    print(f"{Fore.green}✅ Loaded {len(data_list)} samples from CoEdit dataset{Style.reset}")

    # Process the data with more diverse filtering
    processed_data = []
    task_distribution = {"gec": [], "fluency": [], "clarity": [], "coherence": [], "other": []}

    for item in data_list:
        src = item.get("src", "")
        tgt = item.get("tgt", "")
        task = item.get("task", "other")

        if src and tgt:
            # Extract the actual input text (remove instruction prefix)
            input_text = extract_input_text(src)

            # Filter out very long texts but allow more variety
            if len(input_text) <= 800 and len(tgt) <= 800:  # Increased from 500
                processed_item = {
                    "question": input_text,
                    "answer": tgt,
                    "task": task
                }
                processed_data.append(processed_item)

                # Categorize by task for balanced sampling
                if task in task_distribution:
                    task_distribution[task].append(processed_item)
                else:
                    task_distribution["other"].append(processed_item)

    print(f"{Fore.yellow}⚙️ Processed {len(processed_data)} valid samples{Style.reset}")
    print(f"{Fore.cyan}📊 Task distribution:{Style.reset}")
    for task, items in task_distribution.items():
        print(f"  {task}: {len(items)} samples")

    # Ensure we have enough data
    total_needed = pareto_size + feedback_size
    if len(processed_data) < total_needed:
        print(f"{Fore.red}❌ Not enough data! Need {total_needed}, got {len(processed_data)}{Style.reset}")
        ratio = len(processed_data) / total_needed
        pareto_size = max(10, int(pareto_size * ratio))
        feedback_size = max(20, int(feedback_size * ratio))
        print(f"{Fore.yellow}⚠️ Adjusted sizes: Dpareto={pareto_size}, Dfeedback={feedback_size}{Style.reset}")

    # Stratified sampling for better task coverage
    random.seed(seed)

    # Calculate samples per task for Dpareto (balanced representation)
    active_tasks = [k for k, v in task_distribution.items() if v]
    if not active_tasks:
        # Fallback if no tasks found
        dpareto_data = processed_data[:pareto_size]
        dfeedback_data = processed_data[pareto_size:pareto_size + feedback_size]
    else:
        pareto_per_task = max(1, pareto_size // len(active_tasks))
        feedback_per_task = max(2, feedback_size // len(active_tasks))

        dpareto_data = []
        dfeedback_data = []

        # Sample from each task category
        for task in active_tasks:
            items = task_distribution[task]
            if items:
                random.shuffle(items)
                # Take samples for Dpareto
                task_pareto = items[:min(pareto_per_task, len(items))]
                dpareto_data.extend(task_pareto)

                # Take samples for Dfeedback (from remaining)
                remaining = items[len(task_pareto):]
                task_feedback = remaining[:min(feedback_per_task, len(remaining))]
                dfeedback_data.extend(task_feedback)

        # If we need more samples, fill from remaining data
        all_used = set(item["question"] for item in dpareto_data + dfeedback_data)
        remaining_data = [item for item in processed_data if item["question"] not in all_used]

        if len(dpareto_data) < pareto_size and remaining_data:
            additional_pareto = remaining_data[:pareto_size - len(dpareto_data)]
            dpareto_data.extend(additional_pareto)
            remaining_data = remaining_data[len(additional_pareto):]

        if len(dfeedback_data) < feedback_size and remaining_data:
            additional_feedback = remaining_data[:feedback_size - len(dfeedback_data)]
            dfeedback_data.extend(additional_feedback)

    # Final shuffle
    random.shuffle(dpareto_data)
    random.shuffle(dfeedback_data)

    # Remove task metadata for final output
    dpareto_data = [{"question": item["question"], "answer": item["answer"]} for item in dpareto_data]
    dfeedback_data = [{"question": item["question"], "answer": item["answer"]} for item in dfeedback_data]

    # Ensure no overlap
    dpareto_questions = set(item["question"] for item in dpareto_data)
    dfeedback_questions = set(item["question"] for item in dfeedback_data)
    overlap = dpareto_questions & dfeedback_questions
    if overlap:
        print(f"{Fore.yellow}⚠️ Removing {len(overlap)} overlapping samples{Style.reset}")
        dfeedback_data = [item for item in dfeedback_data if item["question"] not in overlap]

    print(f"{Fore.green}✅ Created Dpareto with {len(dpareto_data)} samples{Style.reset}")
    print(f"{Fore.green}✅ Created Dfeedback with {len(dfeedback_data)} samples{Style.reset}")

    return dpareto_data, dfeedback_data


def save_datasets(dpareto_data: list[dict], dfeedback_data: list[dict]) -> None:
    """Save the datasets to JSON files."""

    # Save Dpareto.json
    with open("Dpareto.json", "w", encoding="utf-8") as f:
        json.dump(dpareto_data, f, indent=2, ensure_ascii=False)
    print(f"{Fore.green}✅ Saved Dpareto.json{Style.reset}")

    # Save Dfeedback.json
    with open("Dfeedback.json", "w", encoding="utf-8") as f:
        json.dump(dfeedback_data, f, indent=2, ensure_ascii=False)
    print(f"{Fore.green}✅ Saved Dfeedback.json{Style.reset}")


def preview_samples(dpareto_data: list[dict], dfeedback_data: list[dict], num_samples: int = 3) -> None:
    """Preview some samples from the datasets."""
    print(f"\n{Fore.magenta}📋 Preview of Dpareto samples:{Style.reset}")
    for i, item in enumerate(dpareto_data[:num_samples]):
        print(f"{Fore.cyan}Sample {i+1}:{Style.reset}")
        print(f"  Input:  {item['question'][:100]}...")
        print(f"  Output: {item['answer'][:100]}...")
        print()

    print(f"\n{Fore.magenta}📋 Preview of Dfeedback samples:{Style.reset}")
    for i, item in enumerate(dfeedback_data[:num_samples]):
        print(f"{Fore.cyan}Sample {i+1}:{Style.reset}")
        print(f"  Input:  {item['question'][:100]}...")
        print(f"  Output: {item['answer'][:100]}...")
        print()


if __name__ == "__main__":
    print(f"{Style.bold}{Fore.blue}🚀 CoEdit Dataset Loader for GEPA-Lite{Style.reset}")
    print("=" * 50)

    # Load and process the data with enhanced parameters
    dpareto_data, dfeedback_data = load_and_process_coedit(
        pareto_size=50,     # Increased for better coverage
        feedback_size=100,  # Increased for more diverse feedback
        seed=42,
        max_samples=5000    # Much larger sample pool
    )

    # Preview samples
    preview_samples(dpareto_data, dfeedback_data)

    # Save the datasets
    save_datasets(dpareto_data, dfeedback_data)

    print(f"\n{Style.bold}{Fore.green}🎉 Dataset preparation complete!{Style.reset}")
    print("You can now run: python GEPA.py")
