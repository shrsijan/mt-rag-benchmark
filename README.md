# MTRAG: Multi-Turn RAG Benchmark - Experiments

This repository contains the evaluation and experiment pipeline for the **MTRAG Benchmark**. 
It focuses solely on the code required to run the custom Tasks A, B, and C. 

## Original Data & References

For the dataset, corpora, and the original paper references, please visit the main repository:
**[IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)**

You will need to download the corpora and human evaluation data from the link above and place them in the required directories (e.g., `task_A/data/`) to run the scripts contained here.

## Benchmark Tasks Overview

This repository includes custom scripts for running experiments across the benchmark tasks. You can run these using either local models (e.g., Mac MPS, CUDA) or Hugging Face APIs.

### Task A: Retrieval (Hybrid Search & Reranking)
- **Directory**: `task_A/`
- **Main Script**: `task_A/run_eval_task_a.py`
- **Description**: Evaluates retrieval by running a hybrid search (Dense embedding with `BAAI/bge-base-en-v1.5` + BM25) followed by CrossEncoder reranking (`BAAI/bge-reranker-large`).
- **Execution**: Automatically detects and runs on MPS (Mac), CUDA, or CPU depending on your hardware using local Hugging Face sentence-transformers. 
- **Usage Example**:
  ```bash
  python task_A/run_eval_task_a.py --input <path_to_queries.jsonl> --output task_A/submission.jsonl --data_dir task_A/data
  ```
  *(Ensure you have the downloaded corpus embeddings and ID maps in the `task_A/data` folder)*

### Task B: Generation from Context
- **Directory**: `task_B/`
- **Main Script**: `task_B/run_generation.py`
- **Description**: Generates answers from provided reference contexts.
- **Execution**: Can be run using the Hugging Face Inference API or locally on MPS/GPU by passing the `--local` flag to load the model locally instead of making remote API calls.
- **Usage Example**:
  ```bash
  python -m task_B.run_generation --input <path_to_input.jsonl> --output task_B/submission_eval.jsonl --local
  ```

### Task C: End-to-End RAG Pipeline
- **Directory**: `task_C/`
- **Main Script**: `task_C/run.sh`
- **Description**: An end-to-end pipeline that chains dense retrieval and LLM generation. 
- **Execution**: The shell script supports passing `--local_generator` to run generation using a local MPS/GPU model, or uses the Hugging Face API by default.
- **Usage Example**:
  ```bash
  bash task_C/run.sh <input_file.jsonl> <output_file.jsonl> --local_generator
  ```

## Getting Started & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/shrsijan/mt-rag-benchmark.git
   cd mt-rag-benchmark
   ```
2. **Install requirements:**
   ```bash
   pip install -r temp_requirements.txt
   ```
3. **Download Data:**
   Visit the [original repository](https://github.com/IBM/mt-rag-benchmark) to download the multi-turn conversations, queries, and passage-level corpora required to run these scripts.
