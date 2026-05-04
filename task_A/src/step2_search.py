import os
import json
import numpy as np
import faiss
import argparse

# Configuration
DOMAINS = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'govt': 'mt-rag-govt-elser-512-100-20240611'
}
TOP_K = 50

def search_domain(domain, data_dir):
    print(f"[{domain}] Loading data...")
    corpus_emb = np.load(f"{data_dir}/{domain}_corpus.npy")
    query_emb = np.load(f"{data_dir}/{domain}_queries.npy")
    
    with open(f"{data_dir}/{domain}_ids.json", 'r') as f:
        doc_ids = json.load(f)
    with open(f"{data_dir}/{domain}_qids.json", 'r') as f:
        q_ids = json.load(f)
        
    print(f"[{domain}] Building search index (FlatIP)...")
    d = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus_emb)
    
    print(f"[{domain}] Searching...")
    scores, indices = index.search(query_emb, TOP_K)
    
    results = []
    collection_name = DOMAINS[domain]
    
    for i, q_id in enumerate(q_ids):
        contexts = []
        for j in range(TOP_K):
            idx = indices[i][j]
            score = float(scores[i][j])
            if idx < len(doc_ids):
                contexts.append({
                    "document_id": doc_ids[idx],
                    "score": score
                })
        
        results.append({
            "task_id": q_id,
            "Collection": collection_name,
            "contexts": contexts
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=list(DOMAINS.keys()))
    parser.add_argument("--data_dir", type=str, default="task_A/data")
    parser.add_argument("--output", type=str, default="task_A/intermediates_top50.jsonl")
    args = parser.parse_args()
    
    domains = [args.domain] if args.domain else DOMAINS.keys()
    
    all_results = []
    for d in domains:
        res = search_domain(d, args.data_dir)
        all_results.extend(res)
        
    print(f"Saving {len(all_results)} results to {args.output}")

    # We'll just append.
    with open(args.output, 'a') as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
