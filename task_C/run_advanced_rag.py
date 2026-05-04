#!/usr/bin/env python3
"""
Task C: Advanced RAG Pipeline with state-of-the-art methods
- Hybrid Search (BM25 + Dense) with RRF fusion
- HyDE (Hypothetical Document Embeddings)
- Query Decomposition
- Strong cross-encoder reranking

Usage:
    python -m task_C.run_advanced_rag --input <input_file> --output <output_file>
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_C.src.advanced_retriever import AdvancedRetriever, COLLECTION_TO_DOMAIN
from task_C.src.generator import RAGGenerator, SimpleGenerator

# Full Collection names for output (evaluation expects these)
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'ibmcloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'govt': 'mt-rag-govt-elser-512-100-20240611'
}


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


def run_advanced_rag_pipeline(
    input_file: str,
    output_file: str,
    data_dir: str = 'task_A/data',
    top_k: int = 5,
    use_hyde: bool = True,
    use_query_decomposition: bool = True,
    use_hybrid: bool = True,
    generator_model: str = "meta-llama/Llama-3.3-70B-Instruct",
    use_local_generator: bool = False,
    use_simple_generator: bool = False,
    retriever_llm: str = "meta-llama/Llama-3.3-70B-Instruct"
):
    """
    Run the advanced RAG pipeline with state-of-the-art methods.
    """
    print("=" * 60)
    print("Task C: Advanced RAG Pipeline")
    print("=" * 60)
    print(f"Features enabled:")
    print(f"  - Hybrid Search (BM25 + Dense): {use_hybrid}")
    print(f"  - HyDE: {use_hyde}")
    print(f"  - Query Decomposition: {use_query_decomposition}")
    print("=" * 60)

    # Load tasks
    print(f"\nLoading tasks from {input_file}...")
    tasks = load_tasks(input_file)
    print(f"Loaded {len(tasks)} tasks")

    # Initialize advanced retriever
    print("\nInitializing advanced retriever...")
    retriever = AdvancedRetriever(
        data_dir=data_dir,
        use_reranker=True,
        use_hyde=use_hyde,
        use_query_decomposition=use_query_decomposition,
        use_hybrid=use_hybrid,
        llm_model=retriever_llm
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

    for task in tqdm(tasks, desc="Advanced RAG Pipeline"):
        task_id = task.get('task_id', 'unknown')

        try:
            # Step 1: Retrieve passages with advanced methods
            contexts = retriever.retrieve(task, top_k_initial=100, top_k_final=top_k)

            # Step 2: Generate answer
            answer = generator.generate(task, contexts)

            # Build output record
            collection = task.get('Collection', '')
            output_record = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task_id,
                'Collection': COLLECTION_NAMES.get(collection, collection),  # Map to full name
                'input': task.get('input', []),
                'contexts': contexts,
                'predictions': [{'text': answer}]
            }
            predictions.append(output_record)

        except Exception as e:
            print(f"\nError processing task {task_id}: {e}")
            collection = task.get('Collection', '')
            output_record = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task_id,
                'Collection': COLLECTION_NAMES.get(collection, collection),  # Map to full name
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


def main():
    parser = argparse.ArgumentParser(description="Task C: Advanced RAG Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with tasks")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for predictions")
    parser.add_argument("--data_dir", type=str, default="task_A/data", help="Directory for pre-computed embeddings")
    parser.add_argument("--top_k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--no_hyde", action="store_true", help="Disable HyDE")
    parser.add_argument("--no_decomposition", action="store_true", help="Disable query decomposition")
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid search")
    parser.add_argument("--generator_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID for generation")
    parser.add_argument("--retriever_llm", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="LLM for retrieval augmentation (HyDE, decomposition)")
    parser.add_argument("--local_generator", action="store_true", help="Run generator locally")
    parser.add_argument("--simple_generator", action="store_true", help="Use simple extractive generator")

    args = parser.parse_args()

    run_advanced_rag_pipeline(
        input_file=args.input,
        output_file=args.output,
        data_dir=args.data_dir,
        top_k=args.top_k,
        use_hyde=not args.no_hyde,
        use_query_decomposition=not args.no_decomposition,
        use_hybrid=not args.no_hybrid,
        generator_model=args.generator_model,
        use_local_generator=args.local_generator,
        use_simple_generator=args.simple_generator,
        retriever_llm=args.retriever_llm
    )


if __name__ == "__main__":
    main()
