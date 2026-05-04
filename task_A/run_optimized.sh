#!/bin/bash

# Configuration
export KMP_DUPLICATE_LIB_OK=TRUE

# Ensure clean state
echo "Cleaning up previous outputs..."
rm -f task_A/intermediates_top50.jsonl
rm -f task_A/predictions.jsonl

# Step 1: Encode Corpora and Queries (Top-level embeddings + Text Maps)
echo "Step 1: Encoding Corpora and Queries (Base Model)..."
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step1_encode.py --domain clapnq "$@"
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step1_encode.py --domain fiqa "$@"
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step1_encode.py --domain govt "$@"
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step1_encode.py --domain cloud "$@"

# Step 2: Search (Top-50)
echo "Step 2: Searching Candidates (Top-50)..."
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step2_search.py --domain clapnq
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step2_search.py --domain fiqa
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step2_search.py --domain govt
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step2_search.py --domain cloud

# Step 3: Rerank (Top-10)
echo "Step 3: Reranking..."
/opt/homebrew/opt/python@3.11/bin/python3.11 task_A/src/step3_rerank.py

# Step 4: Evaluate
echo "Step 4: Evaluating..."
/opt/homebrew/opt/python@3.11/bin/python3.11 scripts/evaluation/run_retrieval_eval.py --input_file task_A/predictions.jsonl --output_file task_A/eval_results.json

echo "Done! Check task_A/eval_results_aggregate.csv"
