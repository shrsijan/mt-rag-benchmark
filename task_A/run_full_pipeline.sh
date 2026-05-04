#!/bin/bash

# Configuration
export KMP_DUPLICATE_LIB_OK=TRUE

# Ensure clean state
echo "Cleaning up previous predictions..."
rm -f task_A/predictions.jsonl

# Step 1: Encode Corpora and Queries
# This will take approximately 3-4 hours on Mac M1/M2/M3
echo "Step 1: Encoding Corpora and Queries..."
/opt/anaconda3/bin/python task_A/src/step1_encode.py --domain clapnq
/opt/anaconda3/bin/python task_A/src/step1_encode.py --domain fiqa
/opt/anaconda3/bin/python task_A/src/step1_encode.py --domain govt
/opt/anaconda3/bin/python task_A/src/step1_encode.py --domain cloud

# Step 2: Search and Generate Predictions
echo "Step 2: Searching..."
/opt/anaconda3/bin/python task_A/src/step2_search.py --domain clapnq
/opt/anaconda3/bin/python task_A/src/step2_search.py --domain fiqa
/opt/anaconda3/bin/python task_A/src/step2_search.py --domain govt
/opt/anaconda3/bin/python task_A/src/step2_search.py --domain cloud

# Step 3: Evaluate
echo "Step 3: Evaluating..."
/opt/anaconda3/bin/python task_A/src/run_retrieval_eval_patched.py --input_file task_A/predictions.jsonl --output_file task_A/eval_results.json

echo "Done! Check task_A/eval_results.json and task_A/eval_results_aggregate.csv"
