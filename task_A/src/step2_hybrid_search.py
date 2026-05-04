"""
Hybrid Search for Task A: BM25 + Dense retrieval with RRF fusion.
"""

import os
import json
import numpy as np
import faiss
import argparse
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# Configuration
DOMAINS = {
    'clapnq': 'mt-rag-clapnq-elser-512-100-20240503',
    'cloud': 'mt-rag-ibmcloud-elser-512-100-20240502',
    'fiqa': 'mt-rag-fiqa-beir-elser-512-100-20240501',
    'govt': 'mt-rag-govt-elser-512-100-20240611'
}
TOP_K = 50  # Initial retrieval
RRF_K = 60  # RRF parameter


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
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index(self, doc_ids: List[str], texts: Dict[str, str]):
        """Build BM25 index."""
        self.doc_ids = doc_ids
        self.corpus_size = len(doc_ids)

        # Calculate document frequencies and build inverted index
        doc_term_freqs = {}
        total_len = 0

        print(f"  Building BM25 index for {len(doc_ids)} documents...")
        for i, doc_id in enumerate(doc_ids):
            text = texts.get(doc_id, "")
            tokens = self._tokenize(text)
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)

            # Term frequencies for this doc
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            doc_term_freqs[doc_id] = term_freq

            # Update inverted index
            for token in set(tokens):
                self.inverted_index[token].append((doc_id, term_freq[token]))

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate IDF
        for token, posting_list in self.inverted_index.items():
            self.doc_freqs[token] = len(posting_list)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Search using BM25."""
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

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


def reciprocal_rank_fusion(results_list: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) for each list where document appears
    """
    rrf_scores = defaultdict(float)

    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def search_domain_hybrid(domain: str, data_dir: str):
    """Perform hybrid search (BM25 + Dense) for a domain."""
    print(f"[{domain}] Loading data...")

    # Load dense embeddings
    corpus_emb = np.load(f"{data_dir}/{domain}_corpus.npy")
    query_emb = np.load(f"{data_dir}/{domain}_queries.npy")

    with open(f"{data_dir}/{domain}_ids.json", 'r') as f:
        doc_ids = json.load(f)
    with open(f"{data_dir}/{domain}_qids.json", 'r') as f:
        q_ids = json.load(f)

    # Load text maps for BM25
    with open(f"{data_dir}/{domain}_text_map.json", 'r') as f:
        corpus_texts = json.load(f)
    with open(f"{data_dir}/{domain}_qtext_map.json", 'r') as f:
        query_texts = json.load(f)


    print(f"[{domain}] Building FAISS index...")
    d = corpus_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(corpus_emb)

    # Build BM25 index
    print(f"[{domain}] Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.index(doc_ids, corpus_texts)

    # Perform hybrid search for each query
    print(f"[{domain}] Performing hybrid search for {len(q_ids)} queries...")
    results = []
    collection_name = DOMAINS[domain]

    for i, q_id in enumerate(tqdm(q_ids, desc=f"[{domain}] Hybrid Search")):
        # Dense search
        scores, indices = faiss_index.search(query_emb[i:i+1], TOP_K)
        dense_results = [(doc_ids[idx], float(scores[0][j])) for j, idx in enumerate(indices[0]) if idx < len(doc_ids)]

        # BM25 search
        query_text = query_texts.get(q_id, "")
        bm25_results = bm25.search(query_text, TOP_K)

        # RRF fusion
        fused_results = reciprocal_rank_fusion([dense_results, bm25_results], k=RRF_K)

        # Build contexts list
        contexts = []
        for doc_id, rrf_score in fused_results[:TOP_K]:
            contexts.append({
                "document_id": doc_id,
                "score": float(rrf_score)
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
    parser.add_argument("--output", type=str, default="task_A/intermediates_hybrid.jsonl")
    args = parser.parse_args()

    domains = [args.domain] if args.domain else DOMAINS.keys()

    all_results = []
    for d in domains:
        res = search_domain_hybrid(d, args.data_dir)
        all_results.extend(res)

    print(f"Saving {len(all_results)} results to {args.output}")
    with open(args.output, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
