#!/usr/bin/env python3
"""
Task C: Full RAG Pipeline
This script runs the complete RAG pipeline: retrieval + generation

Usage:
    python -m task_C.run_rag --input <input_file> --output <output_file>

Example:
    python -m task_C.run_rag --input human/mtrageval/sample_data/retrieval_taskac_input.jsonl --output task_C/predictions.jsonl
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_C.src.retriever import DenseRetriever, COLLECTION_TO_DOMAIN
from task_C.src.generator import RAGGenerator, SimpleGenerator


def load_tasks(input_file: str):
    """Load tasks from JSONL file."""
    tasks = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def save_predictions(predictions: list, output_file: str):
    """Save predictions to JSONL file."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')


def run_rag_pipeline(
    input_file: str,
    output_file: str,
    data_dir: str = 'task_A/data',  # Use pre-computed embeddings from task_A
    top_k: int = 5,
    use_reranker: bool = True,
    use_query_rewrite: bool = True,
    generator_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_local_generator: bool = False,
    use_simple_generator: bool = False,
    skip_existing: bool = False
):
    """
    Run the full RAG pipeline.

    Args:
        input_file: Path to input JSONL file with tasks
        output_file: Path to output JSONL file for predictions
        data_dir: Directory for cached embeddings
        top_k: Number of passages to retrieve (default: 5)
        use_reranker: Whether to use cross-encoder reranking
        use_query_rewrite: Whether to use query rewriting for multi-turn
        generator_model: HuggingFace model ID for generation
        use_local_generator: Run generator locally instead of API
        use_simple_generator: Use simple extractive generator (no LLM)
        skip_existing: Skip tasks that already have predictions
    """
    print("=" * 60)
    print("Task C: Full RAG Pipeline")
    print("=" * 60)

    # Load tasks
    print(f"\nLoading tasks from {input_file}...")
    tasks = load_tasks(input_file)
    print(f"Loaded {len(tasks)} tasks")

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = DenseRetriever(
        data_dir=data_dir,
        use_reranker=use_reranker,
        use_query_rewrite=use_query_rewrite
    )

    # Initialize generator
    print("\nInitializing generator...")
    if use_simple_generator:
        generator = SimpleGenerator()
        print("Using simple extractive generator")
    else:
        generator = RAGGenerator(
            model_id=generator_model,
            use_local=use_local_generator
        )

    # Process tasks
    predictions = []
    print(f"\nProcessing {len(tasks)} tasks...")

    for task in tqdm(tasks, desc="RAG Pipeline"):
        task_id = task.get('task_id', 'unknown')

        try:
            # Step 1: Retrieve passages
            contexts = retriever.retrieve(task, top_k_initial=50, top_k_final=top_k)

            # Step 2: Generate answer
            answer = generator.generate(task, contexts)

            # Build output record
            output_record = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task_id,
                'Collection': task.get('Collection'),
                'input': task.get('input', []),
                'contexts': contexts,
                'predictions': [{'text': answer}]
            }
            predictions.append(output_record)

        except Exception as e:
            print(f"\nError processing task {task_id}: {e}")
            # Add error record with empty results
            output_record = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task_id,
                'Collection': task.get('Collection'),
                'input': task.get('input', []),
                'contexts': [],
                'predictions': [{'text': 'I do not have specific information to answer this question.'}]
            }
            predictions.append(output_record)

    # Save predictions
    print(f"\nSaving {len(predictions)} predictions to {output_file}...")
    save_predictions(predictions, output_file)
    print("Done!")

    return predictions


def create_input_from_rag_file(rag_file: str, output_file: str):
    """
    Create clean input file from RAG.jsonl by removing targets, enrichments, etc.
    """
    print(f"Creating input file from {rag_file}...")
    count = 0
    with open(rag_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            task = json.loads(line)
            clean_task = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task.get('task_id'),
                'task_type': task.get('task_type', 'rag'),
                'turn': task.get('turn'),
                'Collection': task.get('Collection'),
                'dataset': task.get('dataset'),
                'input': task.get('input', [])
            }
            f_out.write(json.dumps(clean_task) + '\n')
            count += 1
    print(f"Created input file with {count} tasks: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Task C: Full RAG Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with tasks")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for predictions")
    parser.add_argument("--data_dir", type=str, default="task_A/data", help="Directory for pre-computed embeddings")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--no_reranker", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--no_query_rewrite", action="store_true", help="Disable query rewriting")
    parser.add_argument("--generator_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--local_generator", action="store_true", help="Run generator locally")
    parser.add_argument("--simple_generator", action="store_true", help="Use simple extractive generator")
    parser.add_argument("--create_input", type=str, help="Create input file from RAG.jsonl path")

    args = parser.parse_args()

    # If --create_input is specified, create input file and exit
    if args.create_input:
        create_input_from_rag_file(args.create_input, args.input)
        return

    # Run the RAG pipeline
    run_rag_pipeline(
        input_file=args.input,
        output_file=args.output,
        data_dir=args.data_dir,
        top_k=args.top_k,
        use_reranker=not args.no_reranker,
        use_query_rewrite=not args.no_query_rewrite,
        generator_model=args.generator_model,
        use_local_generator=args.local_generator,
        use_simple_generator=args.simple_generator
    )


if __name__ == "__main__":
    main()
