import os
import json
import torch
from sentence_transformers import CrossEncoder
import argparse
from tqdm import tqdm

RERANKER_MODEL = 'BAAI/bge-reranker-large'  # Upgraded from base for better performance
DATA_DIR = 'task_A/data'
TOP_K_FINAL = 10

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_text_maps(domain, data_dir):
    # Load corpus text map
    with open(f"{data_dir}/{domain}_text_map.json", 'r') as f:
        corpus_texts = json.load(f)
    # Load query text map
    with open(f"{data_dir}/{domain}_qtext_map.json", 'r') as f:
        query_texts = json.load(f)
    return corpus_texts, query_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="task_A/intermediates_top50.jsonl")
    parser.add_argument("--output", type=str, default="task_A/predictions.jsonl")
    parser.add_argument("--data_dir", type=str, default="task_A/data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("Loading Reranker Model...")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = CrossEncoder(RERANKER_MODEL, device=device)

    print(f"Loading retrieval results from {args.input}...")
    retrieval_results = load_jsonl(args.input)
    


    COLL_TO_DOMAIN = {
        'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
        'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud',
        'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
        'mt-rag-govt-elser-512-100-20240611': 'govt'
    }
    
    # Cache text maps to avoid reloading
    domain_text_maps = {} 
    domain_query_maps = {}

    print(f"Reranking {len(retrieval_results)} queries...")
    
    final_results = []
    
    for item in tqdm(retrieval_results):
        coll = item['Collection']
        domain = COLL_TO_DOMAIN.get(coll)
        if not domain:
            print(f"Unknown collection: {coll}")
            continue
            
        if domain not in domain_text_maps:
            print(f"Loading text maps for {domain}...")
            c_map, q_map = load_text_maps(domain, args.data_dir)
            domain_text_maps[domain] = c_map
            domain_query_maps[domain] = q_map
            
        task_id = item['task_id']
        # The query text. Note: task_id matches _id in query file
        query_text = domain_query_maps[domain].get(task_id)
        if not query_text:
            print(f"Query text not found for {task_id}")
            continue
            

        pairs = []
        doc_ids = []
        
        for ctx in item['contexts']:
            doc_id = ctx['document_id']
            doc_text = domain_text_maps[domain].get(doc_id, "")
            pairs.append([query_text, doc_text])
            doc_ids.append(ctx) # Keep original context obj to preserve id
            
        if not pairs:
            final_results.append(item) # Should not happen if intermediate has results
            continue
            
        # Predict scores
        scores = model.predict(pairs)
        
        # Attach new scores
        scored_contexts = []
        for i, score in enumerate(scores):
            ctx = doc_ids[i].copy()
            ctx['score'] = float(score)
            scored_contexts.append(ctx)
            
        # Sort by new score
        scored_contexts.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep top K final
        scored_contexts = scored_contexts[:TOP_K_FINAL]
        
        item['contexts'] = scored_contexts
        final_results.append(item)

    print(f"Saving final predictions to {args.output}")
    with open(args.output, 'w') as f:
        for res in final_results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
