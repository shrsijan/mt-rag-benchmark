#!/usr/bin/env python3
"""
Task B Evaluation Runner: Generation with Reference Passages

Task B uses GOLD/REFERENCE passages provided by the organizers (not retrieved).
The task is to generate an answer based on these reference passages.

Usage:
    HF_API_KEY=your_key python -m task_B.run_eval_task_b --input reference_taskB.jsonl --output task_B/submission_eval.jsonl
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_B.model import TaskBGenerator


def load_tasks(input_file: str):
    """Load tasks from JSONL file."""
    tasks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def save_predictions(predictions: list, output_file: str):
    """Save predictions to JSONL file."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')


def run_task_b_pipeline(
    input_file: str,
    output_file: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    use_local: bool = False
):
    """
    Run Task B: Generation with Reference Passages
    """
    print("=" * 60)
    print("Task B: Generation with Reference Passages")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Model: {model_id}")
    print(f"Local mode: {use_local}")
    print("=" * 60)

    # Load tasks
    print(f"\nLoading tasks from {input_file}...")
    tasks = load_tasks(input_file)
    print(f"Loaded {len(tasks)} tasks")

    # Check input structure
    if tasks:
        sample = tasks[0]
        print(f"\nInput fields: {list(sample.keys())}")
        print(f"Has contexts: {'contexts' in sample}")
        print(f"Contexts count: {len(sample.get('contexts', []))}")

    # Initialize generator
    print(f"\nInitializing generator with {model_id}...")
    generator = TaskBGenerator(
        model_id=model_id,
        use_local=use_local
    )

    # Process tasks
    predictions = []
    print(f"\nGenerating answers for {len(tasks)} tasks...")

    for task in tqdm(tasks, desc="Task B Generation"):
        task_id = task.get('task_id', 'unknown')

        try:
            # Generate answer using provided gold contexts
            answer = generator.generate(task)

            # Build output record - preserve required fields for format checker
            # Task B requires: task_id, input, contexts, predictions
            # Ensure contexts have 'score' field (required by format checker)
            # Limit to max 10 contexts as required by format checker
            contexts_with_scores = []
            for i, ctx in enumerate(task.get('contexts', [])[:10]):  # Limit to 10
                ctx_copy = dict(ctx)
                if 'score' not in ctx_copy:
                    ctx_copy['score'] = 1.0 - (i * 0.1)  # Default decreasing scores
                contexts_with_scores.append(ctx_copy)

            output_record = {
                'task_id': task_id,
                'input': task.get('input', []),
                'contexts': contexts_with_scores,
                'predictions': [{'text': answer}]
            }
            predictions.append(output_record)

        except Exception as e:
            print(f"\nError processing task {task_id}: {e}")
            # Ensure contexts have scores even in error case
            # Limit to max 10 contexts as required by format checker
            contexts_with_scores = []
            for i, ctx in enumerate(task.get('contexts', [])[:10]):  # Limit to 10
                ctx_copy = dict(ctx)
                if 'score' not in ctx_copy:
                    ctx_copy['score'] = 1.0 - (i * 0.1)
                contexts_with_scores.append(ctx_copy)

            output_record = {
                'task_id': task_id,
                'input': task.get('input', []),
                'contexts': contexts_with_scores,
                'predictions': [{'text': "I'm sorry, but I don't have specific information to answer this question."}]
            }
            predictions.append(output_record)

    # Save predictions
    print(f"\nSaving {len(predictions)} predictions to {output_file}...")
    save_predictions(predictions, output_file)

    # Verify output
    print(f"\nVerification:")
    print(f"  - Total predictions: {len(predictions)}")
    if predictions:
        sample_out = predictions[0]
        print(f"  - Output fields: {list(sample_out.keys())}")
        print(f"  - Has predictions: {'predictions' in sample_out}")

    print("\nDone!")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Task B: Generation with Reference Passages")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with tasks and gold contexts")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for predictions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--local", action="store_true",
                        help="Run model locally instead of via API")

    args = parser.parse_args()

    run_task_b_pipeline(
        input_file=args.input,
        output_file=args.output,
        model_id=args.model,
        use_local=args.local
    )


if __name__ == "__main__":
    main()
