#!/usr/bin/env python3
"""
Run generation only on existing predictions with retrieved contexts.
Uses an LLM to generate better answers based on the retrieved passages.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_C.src.generator import RAGGenerator


def load_predictions(input_file: str):
    """Load predictions from JSONL file."""
    predictions = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def save_predictions(predictions: list, output_file: str):
    """Save predictions to JSONL file."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')


def run_generation_only(
    input_file: str,
    output_file: str,
    generator_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    use_local: bool = False
):
    """
    Run LLM generation on existing predictions with retrieved contexts.
    """
    print("=" * 60)
    print("Running Generation on Existing Retrieval")
    print("=" * 60)

    # Load predictions with contexts
    print(f"\nLoading predictions from {input_file}...")
    predictions = load_predictions(input_file)
    print(f"Loaded {len(predictions)} predictions")

    # Initialize generator
    print(f"\nInitializing generator with model: {generator_model}")
    generator = RAGGenerator(
        model_id=generator_model,
        use_local=use_local
    )

    # Process each prediction
    print(f"\nGenerating answers for {len(predictions)} tasks...")
    updated_predictions = []

    for pred in tqdm(predictions, desc="Generation"):
        task_id = pred.get('task_id', 'unknown')

        try:
            # Create task dict for generator
            task = {
                'input': pred.get('input', [])
            }
            contexts = pred.get('contexts', [])

            # Generate new answer
            answer = generator.generate(task, contexts)

            # Update prediction with new answer
            updated_pred = {
                'conversation_id': pred.get('conversation_id'),
                'task_id': task_id,
                'Collection': pred.get('Collection'),
                'input': pred.get('input', []),
                'contexts': contexts,
                'predictions': [{'text': answer}]
            }
            updated_predictions.append(updated_pred)

        except Exception as e:
            print(f"\nError generating for task {task_id}: {e}")
            # Keep original prediction
            updated_predictions.append(pred)

    # Save updated predictions
    print(f"\nSaving {len(updated_predictions)} predictions to {output_file}...")
    save_predictions(updated_predictions, output_file)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Run generation on existing predictions")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with predictions")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--generator_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--local", action="store_true", help="Run generator locally")

    args = parser.parse_args()

    run_generation_only(
        input_file=args.input,
        output_file=args.output,
        generator_model=args.generator_model,
        use_local=args.local
    )


if __name__ == "__main__":
    main()
