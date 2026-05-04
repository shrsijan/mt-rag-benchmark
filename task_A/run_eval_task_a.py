#!/usr/bin/env python3
"""
Task A Evaluation Runner: Retrieval for evaluation file
Handles the official MT-RAG evaluation file format.

Usage:
    python task_A/run_eval_task_a.py --input rag_taskAC.jsonl --output task_A/submission.jsonl
"""

import argparse
import json
import os
import sys
import numpy as np
import faiss
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder


DOMAIN_MAP = {
    'clapnq': 'clapnq',
    'ibmcloud': 'cloud',
    'cloud': 'cloud',
    'fiqa': 'fiqa',
    'govt': 'govt'
}

# Full Collection names for output
COLLECTION_NAMES = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'ibmcloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'govt': 'mt-rag-govt-elser-512-100-20240611'
}

TOP_K_INITIAL = 100  # Initial retrieval
TOP_K_FINAL = 10     # Return 10 contexts as required
RRF_K = 60           # RRF parameter

ENCODER_MODEL = 'BAAI/bge-base-en-v1.5'
RERANKER_MODEL = 'BAAI/bge-reranker-large'


class BM25Retriever:
    """Simple BM25 implementation for hybrid search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lens = {}
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.inverted_index = defaultdict(list)
        self.doc_ids = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def index(self, doc_ids: List[str], texts: Dict[str, str]):
        self.doc_ids = doc_ids
        self.corpus_size = len(doc_ids)
        total_len = 0

        for doc_id in doc_ids:
            text = texts.get(doc_id, "")
            tokens = self._tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for token in set(tokens):
                self.inverted_index[token].append((doc_id, term_freq[token]))

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

        for token, posting_list in self.inverted_index.items():
            self.doc_freqs[token] = len(posting_list)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            df = self.doc_freqs[token]
            idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self.inverted_index[token]:
                doc_len = self.doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[doc_id] += idf * numerator / denominator

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(results_list: List[List[Tuple[str, float]]], k: int = 60):
    """Fuse multiple result lists using RRF."""
    rrf_scores = defaultdict(float)
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def extract_query(task: Dict) -> str:
    """Extract query text from conversation."""
    conversation = task.get('input', [])

    # Get last user turn
    for turn in reversed(conversation):
        if turn.get('speaker') == 'user':
            return turn.get('text', '')

    return ''


def main():
    parser = argparse.ArgumentParser(description="Task A: Retrieval Evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input evaluation JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output submission JSONL file")
    parser.add_argument("--data_dir", type=str, default="task_A/data", help="Data directory with embeddings")
    args = parser.parse_args()

    # Determine device
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load models
    print(f"Loading encoder: {ENCODER_MODEL}")
    encoder = SentenceTransformer(ENCODER_MODEL, device=device)

    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    # Load tasks
    print(f"Loading tasks from {args.input}")
    tasks = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks")


    tasks_by_domain = defaultdict(list)
    for task in tasks:
        collection = task.get('Collection', '')
        domain = DOMAIN_MAP.get(collection, collection)
        tasks_by_domain[domain].append(task)

    print(f"Tasks per domain: {dict((k, len(v)) for k, v in tasks_by_domain.items())}")


    all_results = []

    for domain, domain_tasks in tasks_by_domain.items():
        print(f"\n{'='*60}")
        print(f"Processing {domain}: {len(domain_tasks)} tasks")
        print(f"{'='*60}")

        # Load corpus data
        emb_path = f"{args.data_dir}/{domain}_corpus.npy"
        ids_path = f"{args.data_dir}/{domain}_ids.json"
        texts_path = f"{args.data_dir}/{domain}_text_map.json"

        if not os.path.exists(emb_path):
            print(f"Warning: Embeddings not found for {domain}, skipping")
            continue

        print(f"Loading corpus embeddings...")
        corpus_emb = np.load(emb_path)
        with open(ids_path, 'r') as f:
            doc_ids = json.load(f)
        with open(texts_path, 'r') as f:
            corpus_texts = json.load(f)


        print(f"Building FAISS index ({corpus_emb.shape[0]} docs)...")
        d = corpus_emb.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(corpus_emb)

        # Build BM25 index
        print(f"Building BM25 index...")
        bm25 = BM25Retriever()
        bm25.index(doc_ids, corpus_texts)

        # Process tasks
        print(f"Processing {len(domain_tasks)} queries...")
        for task in tqdm(domain_tasks, desc=f"{domain}"):
            task_id = task.get('task_id', '')
            collection = task.get('Collection', '')

            # Extract query
            query_text = extract_query(task)
            if not query_text:
                print(f"Warning: Empty query for {task_id}")
                continue

            # Dense search
            query_with_prefix = f"Represent this sentence for searching relevant passages: {query_text}"
            query_emb = encoder.encode([query_with_prefix], normalize_embeddings=True)
            scores, indices = faiss_index.search(query_emb, TOP_K_INITIAL)
            dense_results = [(doc_ids[idx], float(scores[0][j])) for j, idx in enumerate(indices[0]) if idx < len(doc_ids)]

            # BM25 search
            bm25_results = bm25.search(query_text, TOP_K_INITIAL)

            # RRF fusion
            fused_results = reciprocal_rank_fusion([dense_results, bm25_results], k=RRF_K)

            # Build candidates for reranking
            candidates = []
            for doc_id, rrf_score in fused_results[:TOP_K_INITIAL]:
                doc_text = corpus_texts.get(doc_id, "")
                candidates.append({
                    'document_id': doc_id,
                    'text': doc_text,
                    'score': rrf_score
                })

            # Rerank
            if candidates:
                pairs = [[query_text, c['text']] for c in candidates[:75]]
                rerank_scores = reranker.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    if i < len(candidates):
                        candidates[i]['score'] = float(score)
                candidates.sort(key=lambda x: x['score'], reverse=True)

            # Build output (top 10 contexts)
            contexts = []
            for c in candidates[:TOP_K_FINAL]:
                contexts.append({
                    'document_id': c['document_id'],
                    'score': c['score']
                })

            all_results.append({
                'task_id': task_id,
                'Collection': COLLECTION_NAMES.get(collection, collection),  # Map to full collection name for evaluation
                'contexts': contexts
            })

    # Save results
    print(f"\nSaving {len(all_results)} results to {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    print("Done!")


if __name__ == "__main__":
    main()
