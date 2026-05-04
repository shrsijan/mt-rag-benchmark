"""
Advanced Retriever for Task C: Full RAG Pipeline
Implements state-of-the-art retrieval methods:
- Hybrid Search (BM25 + Dense) with RRF fusion
- HyDE (Hypothetical Document Embeddings)
- Query Decomposition for complex questions
- Strong cross-encoder reranking
"""

import os
import json
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient


COLLECTION_TO_DOMAIN = {
    # Full collection names
    'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
    'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud',
    'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
    'mt-rag-govt-elser-512-100-20240611': 'govt',
    # Short names (from evaluation file)
    'clapnq': 'clapnq',
    'ibmcloud': 'cloud',
    'cloud': 'cloud',
    'fiqa': 'fiqa',
    'govt': 'govt'
}

DOMAIN_TO_COLLECTION = {v: k for k, v in COLLECTION_TO_DOMAIN.items()}

CORPUS_PATHS = {
    'clapnq': 'corpora/passage_level/clapnq.jsonl',
    'cloud': 'corpora/passage_level/cloud.jsonl',
    'fiqa': 'corpora/passage_level/fiqa.jsonl',
    'govt': 'corpora/passage_level/govt.jsonl'
}

# Prompts for advanced retrieval
QUERY_REWRITE_PROMPT = """Given the conversation below, rewrite the last user question to be a standalone, self-contained question that captures the full context needed to answer it.

Conversation:
{conversation}

Rewrite the last question to be standalone. Only output the rewritten question, nothing else."""

HYDE_PROMPT = """Given this question, write a short hypothetical passage that would answer it. The passage should be factual and informative.

Question: {question}

Write a hypothetical answer passage (2-3 sentences):"""

QUERY_DECOMPOSITION_PROMPT = """Analyze this question and determine if it requires multiple pieces of information to answer.

Question: {question}

If this is a complex question that needs multiple facts, break it into 2-3 simpler sub-questions.
If it's already simple and direct, just return the original question.

Output format (one question per line):"""


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


class AdvancedRetriever:
    """Advanced retriever with hybrid search, HyDE, and query decomposition."""

    def __init__(
        self,
        encoder_model: str = 'BAAI/bge-base-en-v1.5',
        reranker_model: str = 'BAAI/bge-reranker-large',  # Stronger reranker
        data_dir: str = 'task_A/data',
        use_reranker: bool = True,
        use_hyde: bool = True,
        use_query_decomposition: bool = True,
        use_hybrid: bool = True,
        llm_model: str = 'meta-llama/Llama-3.3-70B-Instruct',
        device: str = None,
        hf_api_key: str = None
    ):
        self.data_dir = data_dir
        self.use_reranker = use_reranker
        self.use_hyde = use_hyde
        self.use_query_decomposition = use_query_decomposition
        self.use_hybrid = use_hybrid

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")

        # Load encoder
        print(f"Loading encoder model: {encoder_model}")
        self.encoder = SentenceTransformer(encoder_model, device=self.device)

        # Load stronger reranker
        if self.use_reranker:
            print(f"Loading reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model, device=self.device)


        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if self.hf_api_key:
            self.llm_client = InferenceClient(model=llm_model, token=self.hf_api_key)
        else:
            print("Warning: No HF API key found. HyDE and query decomposition will be disabled.")
            self.use_hyde = False
            self.use_query_decomposition = False

        # Cache for loaded data
        self.corpus_embeddings = {}
        self.corpus_ids = {}
        self.corpus_texts = {}
        self.faiss_indices = {}
        self.bm25_indices = {}

        os.makedirs(data_dir, exist_ok=True)

    def _load_corpus(self, domain: str):
        """Load corpus embeddings and build indices."""
        if domain in self.corpus_embeddings:
            return

        emb_path = f"{self.data_dir}/{domain}_corpus.npy"
        ids_path = f"{self.data_dir}/{domain}_ids.json"
        texts_path = f"{self.data_dir}/{domain}_text_map.json"

        # Load pre-computed embeddings
        if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(texts_path):
            print(f"Loading pre-computed embeddings for {domain}...")
            self.corpus_embeddings[domain] = np.load(emb_path)
            with open(ids_path, 'r') as f:
                self.corpus_ids[domain] = json.load(f)
            with open(texts_path, 'r') as f:
                self.corpus_texts[domain] = json.load(f)
        else:
            raise FileNotFoundError(f"Pre-computed embeddings not found for {domain}")


        print(f"Building FAISS index for {domain}...")
        import faiss
        d = self.corpus_embeddings[domain].shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(self.corpus_embeddings[domain])
        self.faiss_indices[domain] = index

        # Build BM25 index for hybrid search
        if self.use_hybrid:
            print(f"Building BM25 index for {domain}...")
            bm25 = BM25Retriever()
            bm25.index(self.corpus_ids[domain], self.corpus_texts[domain])
            self.bm25_indices[domain] = bm25

    def _call_llm(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        """Call LLM for query processing."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""

    def _rewrite_query(self, conversation: List[Dict]) -> str:
        """Rewrite multi-turn query to standalone."""
        conv_str = ""
        for turn in conversation:
            speaker = turn.get('speaker', 'user')
            text = turn.get('text', '')
            conv_str += f"{speaker.capitalize()}: {text}\n"

        prompt = QUERY_REWRITE_PROMPT.format(conversation=conv_str.strip())
        rewritten = self._call_llm(prompt, max_tokens=256, temperature=0.1)

        if not rewritten:
            # Fallback to last user turn
            for turn in reversed(conversation):
                if turn.get('speaker') == 'user':
                    return turn.get('text', '')
            return ""

        # Clean up
        rewritten = rewritten.replace("Rewritten question:", "").strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        return rewritten

    def _generate_hyde_passage(self, query: str) -> str:
        """Generate hypothetical document for HyDE."""
        prompt = HYDE_PROMPT.format(question=query)
        hyde_passage = self._call_llm(prompt, max_tokens=150, temperature=0.3)
        return hyde_passage if hyde_passage else query

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        prompt = QUERY_DECOMPOSITION_PROMPT.format(question=query)
        response = self._call_llm(prompt, max_tokens=200, temperature=0.1)

        if not response:
            return [query]

        # Parse sub-queries
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
        sub_queries = [q.lstrip('0123456789.-) ') for q in sub_queries]
        sub_queries = [q for q in sub_queries if len(q) > 5]  # Filter very short lines

        return sub_queries if sub_queries else [query]

    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Tuple[str, float]]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """Fuse multiple result lists using RRF."""
        rrf_scores = defaultdict(float)

        for results in results_list:
            for rank, (doc_id, _) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    def _dense_search(self, query: str, domain: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Perform dense retrieval."""
        query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.encoder.encode(
            [query_with_prefix],
            normalize_embeddings=True
        )

        scores, indices = self.faiss_indices[domain].search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(self.corpus_ids[domain]):
                doc_id = self.corpus_ids[domain][idx]
                results.append((doc_id, float(scores[0][i])))

        return results

    def retrieve(
        self,
        task: Dict,
        top_k_initial: int = 100,  # More candidates for better recall
        top_k_final: int = 5
    ) -> List[Dict]:
        """
        Advanced retrieval with hybrid search, HyDE, and query decomposition.
        """
        collection = task.get('Collection', '')
        domain = COLLECTION_TO_DOMAIN.get(collection)

        if not domain:
            print(f"Unknown collection: {collection}")
            return []

        # Ensure corpus is loaded
        self._load_corpus(domain)

        # Get conversation
        conversation = task.get('input', [])

        # Get last user turn
        last_user_text = ""
        for turn in reversed(conversation):
            if turn.get('speaker') == 'user':
                last_user_text = turn.get('text', '')
                break

        if not last_user_text:
            return []

        # Step 1: Rewrite query if multi-turn
        if len(conversation) > 1 and self.hf_api_key:
            query_text = self._rewrite_query(conversation)
        else:
            query_text = last_user_text

        # Step 2: Collect all search results
        all_results = []

        # Original query - dense search
        dense_results = self._dense_search(query_text, domain, top_k_initial)
        all_results.append(dense_results)

        # BM25 search (if hybrid enabled)
        if self.use_hybrid and domain in self.bm25_indices:
            bm25_results = self.bm25_indices[domain].search(query_text, top_k_initial)
            all_results.append(bm25_results)

        # HyDE search (if enabled)
        if self.use_hyde and self.hf_api_key:
            hyde_passage = self._generate_hyde_passage(query_text)
            if hyde_passage and hyde_passage != query_text:
                hyde_results = self._dense_search(hyde_passage, domain, top_k_initial // 2)
                all_results.append(hyde_results)

        # Query decomposition (if enabled)
        if self.use_query_decomposition and self.hf_api_key:
            sub_queries = self._decompose_query(query_text)
            if len(sub_queries) > 1:
                for sub_q in sub_queries[:3]:  # Limit to 3 sub-queries
                    sub_results = self._dense_search(sub_q, domain, top_k_initial // 3)
                    all_results.append(sub_results)

        # Step 3: Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(all_results)

        # Step 4: Build candidate list
        candidates = []
        seen_ids = set()
        for doc_id, rrf_score in fused_results:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            doc_text = self.corpus_texts[domain].get(doc_id, "")
            candidates.append({
                'document_id': doc_id,
                'text': doc_text,
                'score': rrf_score
            })
            if len(candidates) >= top_k_initial:
                break

        # Step 5: Rerank with cross-encoder
        if self.use_reranker and candidates:
            pairs = [[query_text, c['text']] for c in candidates[:75]]  # Rerank top 75
            rerank_scores = self.reranker.predict(pairs)

            for i, score in enumerate(rerank_scores):
                if i < len(candidates):
                    candidates[i]['score'] = float(score)

            candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates[:top_k_final]
