import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

# Configuration
DOMAINS = {
    'clapnq': {
        'corpus': 'corpora/passage_level/clapnq.jsonl',
        'queries': 'human/retrieval_tasks/clapnq/clapnq_rewrite.jsonl',
    },
    'cloud': {
        'corpus': 'corpora/passage_level/cloud.jsonl',
        'queries': 'human/retrieval_tasks/cloud/cloud_rewrite.jsonl',
    },
    'fiqa': {
        'corpus': 'corpora/passage_level/fiqa.jsonl',
        'queries': 'human/retrieval_tasks/fiqa/fiqa_rewrite.jsonl',
    },
    'govt': {
        'corpus': 'corpora/passage_level/govt.jsonl',
        'queries': 'human/retrieval_tasks/govt/govt_rewrite.jsonl',
    }
}

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
BATCH_SIZE = 32

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def encode_domain(domain, model, output_dir, debug=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # CORPUS
    if os.path.exists(f"{output_dir}/{domain}_corpus.npy") and not debug:
        print(f"[{domain}] Corpus embeddings already exist, skipping.")
    else:
        print(f"[{domain}] Loading corpus...")
        corpus_path = DOMAINS[domain]['corpus']
        documents = load_jsonl(corpus_path)
        if debug: documents = documents[:1000]
        
        texts = []
        ids = []
        # Save text mapping for reranker
        text_map = {}
        
        for doc in documents:
            did = doc['_id']
            ids.append(did)
            title = doc.get('title', '')
            content = doc.get('text', '')
            full_text = f"{title}\n{content}" if title else content
            texts.append(full_text)
            text_map[did] = full_text
            
        print(f"[{domain}] Encoding {len(texts)} docs...")
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
        
        np.save(f"{output_dir}/{domain}_corpus.npy", embeddings)
        with open(f"{output_dir}/{domain}_ids.json", 'w') as f:
            json.dump(ids, f)
            
        # Save text map for step 3
        # Use simple json for now, or pickle if too slow/large. JSON is safer for compatibility.
        # Check size: 100k docs * ~1kb = 100MB. JSON is fine.
        with open(f"{output_dir}/{domain}_text_map.json", 'w') as f:
            json.dump(text_map, f)

    # QUERIES
    print(f"[{domain}] Loading queries...")
    queries_path = DOMAINS[domain]['queries']
    queries = load_jsonl(queries_path)
    if debug: queries = queries[:10]
    
    q_texts = []
    q_ids = []
    q_text_map = {}
    
    for q in queries:
        qid = q['_id']
        txt = q['text']
        q_ids.append(qid)
        q_text_map[qid] = txt
        q_texts.append(f"Represent this sentence for searching relevant passages: {txt}")
    
    print(f"[{domain}] Encoding {len(q_texts)} queries...")
    q_embeddings = model.encode(q_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    
    np.save(f"{output_dir}/{domain}_queries.npy", q_embeddings)
    with open(f"{output_dir}/{domain}_qids.json", 'w') as f:
        json.dump(q_ids, f)
    with open(f"{output_dir}/{domain}_qtext_map.json", 'w') as f:
        json.dump(q_text_map, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()))
    parser.add_argument("--output_dir", type=str, default="task_A/data")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    domains = [args.domain] if args.domain else DOMAINS.keys()
    
    for d in domains:
        encode_domain(d, model, args.output_dir, args.debug)

if __name__ == "__main__":
    main()
