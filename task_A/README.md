# Task A: Retrieval

This directory contains the optimized retrieval pipeline for Task A of the MT-RAG Benchmark.
The pipeline uses a two-stage approach:
1.  **Dense Retrieval**: BGE-Base (`BAAI/bge-base-en-v1.5`)
2.  **Reranking**: BGE-Reranker (`BAAI/bge-reranker-base`)

## Structure
-   `run_optimized.sh`: Main script to run the full pipeline.
-   `src/`: Source code for encoding, searching, reranking, and evaluation.
-   `data/`: Directory for storing embeddings and indices (approx 5-10GB required).
-   `predictions.jsonl`: Final output in the required format.

## Usage

To run the full pipeline (approx 3-4 hours on M-series Mac):
```bash
./run_optimized.sh
```

To run a fast debug verification (on 1000 docs only):
```bash
./run_optimized.sh --debug
```

## Requirements
See `requirements.txt`.
