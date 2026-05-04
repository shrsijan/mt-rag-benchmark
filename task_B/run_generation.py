import argparse
import json
import os
from tqdm import tqdm
from .model import TaskBGenerator

def load_data(input_file):
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--local", action="store_true", help="Run locally using MPS/GPU instead of API")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID")
    args = parser.parse_args()

    # Load data
    tasks = load_data(args.input)
    
    # Initialize Generator
    generator = TaskBGenerator(model_id=args.model, use_local=args.local)
    
    output_records = []
    
    print(f"Generating predictions for {len(tasks)} tasks...")
    for task in tqdm(tasks):
        # Generate prediction
        try:
            response_text = generator.generate(task)
        except Exception as e:
            print(f"Error processing task {task.get('task_id')}: {e}")
            response_text = "Error generating response."

        # Create output record preserving original fields + predictions
        # The evaluation script expects the original task object with a 'predictions' list added.
        record = task.copy()
        record['predictions'] = [{"text": response_text}]
        output_records.append(record)

    # Save
    save_predictions(output_records, args.output)
    print(f"Saved {len(output_records)} predictions to {args.output}")

if __name__ == "__main__":
    main()
