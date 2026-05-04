#!/bin/bash
# Task C: Full RAG Pipeline Runner
# This script runs the complete RAG pipeline (retrieval + generation)

set -e

# Configuration
INPUT_FILE="${INPUT_FILE:-human/mtrageval/sample_data/retrieval_taskac_input.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-task_C/predictions.jsonl}"
DATA_DIR="${DATA_DIR:-task_C/data}"
TOP_K="${TOP_K:-5}"
GENERATOR_MODEL="${GENERATOR_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            # Run on full dataset
            INPUT_FILE="task_C/input_full.jsonl"
            OUTPUT_FILE="task_C/predictions_full.jsonl"
            CREATE_INPUT="human/generation_tasks/RAG.jsonl"
            shift
            ;;
        --sample)
            # Run on sample data (default)
            INPUT_FILE="human/mtrageval/sample_data/retrieval_taskac_input.jsonl"
            OUTPUT_FILE="task_C/predictions_sample.jsonl"
            shift
            ;;
        --simple)

            SIMPLE_GEN="--simple_generator"
            shift
            ;;
        --local)
            # Run generator locally
            LOCAL_GEN="--local_generator"
            shift
            ;;
        --no-rerank)
            # Disable reranking
            NO_RERANK="--no_reranker"
            shift
            ;;
        --debug)
            # Debug mode - fewer samples
            DEBUG_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run.sh [--full|--sample|--simple|--local|--no-rerank|--debug]"
            exit 1
            ;;
    esac
done

echo ""
echo "Task C: Full RAG Pipeline"
echo ""
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Data directory: $DATA_DIR"
echo "Top-K passages: $TOP_K"
echo "Generator model: $GENERATOR_MODEL"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# If running full dataset, create input file first
if [[ -n "$CREATE_INPUT" ]]; then
    echo "Creating input file from RAG.jsonl..."
    python -m task_C.run_rag --create_input "$CREATE_INPUT" --input "$INPUT_FILE" --output "$OUTPUT_FILE"
fi

# Run the RAG pipeline
echo "Running RAG pipeline..."
python -m task_C.run_rag \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --data_dir "$DATA_DIR" \
    --top_k "$TOP_K" \
    --generator_model "$GENERATOR_MODEL" \
    $SIMPLE_GEN $LOCAL_GEN $NO_RERANK

echo ""
echo ""
echo "RAG Pipeline Complete!"
echo ""
echo "Predictions saved to: $OUTPUT_FILE"

# Run format checker
echo ""
echo "Running format checker..."
python scripts/evaluation/format_checker.py \
    --input_file "$INPUT_FILE" \
    --prediction_file "$OUTPUT_FILE" \
    --mode rag_taskc

echo ""
echo ""
echo "To run evaluation, use:"
echo ""
echo ""
echo "# For retrieval evaluation:"
echo "python scripts/evaluation/run_retrieval_eval.py \\"
echo "    --input_file $OUTPUT_FILE \\"
echo "    --output_file task_C/retrieval_eval_results.jsonl"
echo ""
echo "# For generation evaluation (requires API key):"
echo "python scripts/evaluation/run_generation_eval.py \\"
echo "    -i $OUTPUT_FILE \\"
echo "    -o task_C/generation_eval_results.jsonl \\"
echo "    -e scripts/evaluation/config.yaml \\"
echo "    --provider hf \\"
echo "    --judge_model ibm-granite/granite-3.3-8b-instruct"
echo ""
