"""
Dense Retriever for Task C: Full RAG Pipeline
Implements query rewriting, dense encoding, FAISS search, and cross-encoder reranking
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient
from .prompts import QUERY_REWRITE_PROMPT


COLLECTION_TO_DOMAIN = {
    'mt-rag-clapnq-elser-512-100-20240503': 'clapnq',
    'mt-rag-ibmcloud-elser-512-100-20240502': 'cloud',
    'mt-rag-fiqa-beir-elser-512-100-20240501': 'fiqa',
    'mt-rag-govt-elser-512-100-20240611': 'govt'
}

DOMAIN_TO_COLLECTION = {v: k for k, v in COLLECTION_TO_DOMAIN.items()}

# Corpus file paths
CORPUS_PATHS = {
    'clapnq': 'corpora/passage_level/clapnq.jsonl',
    'cloud': 'corpora/passage_level/cloud.jsonl',
    'fiqa': 'corpora/passage_level/fiqa.jsonl',
    'govt': 'corpora/passage_level/govt.jsonl'
}


class DenseRetriever:
    def __init__(
        self,
        encoder_model: str = 'BAAI/bge-base-en-v1.5',
        reranker_model: str = 'BAAI/bge-reranker-base',
        data_dir: str = 'task_A/data',  # Use pre-computed embeddings from task_A
        use_reranker: bool = True,
        use_query_rewrite: bool = True,
        rewrite_model: str = 'meta-llama/Llama-3.1-8B-Instruct',
        device: str = None,
        hf_api_key: str = None
    ):
        self.data_dir = data_dir
        self.use_reranker = use_reranker
        self.use_query_rewrite = use_query_rewrite

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

        # Load reranker if enabled
        if self.use_reranker:
            print(f"Loading reranker model: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model, device=self.device)

        # Setup query rewrite client if enabled
        if self.use_query_rewrite:
            self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if self.hf_api_key:
                self.rewrite_client = InferenceClient(model=rewrite_model, token=self.hf_api_key)
            else:
                print("Warning: No HF API key found. Query rewrite will be disabled.")
                self.use_query_rewrite = False

        # Cache for loaded data
        self.corpus_embeddings = {}
        self.corpus_ids = {}
        self.corpus_texts = {}
        self.faiss_indices = {}

        os.makedirs(data_dir, exist_ok=True)

    def _load_corpus(self, domain: str):
        """Load or compute corpus embeddings for a domain."""
        if domain in self.corpus_embeddings:
            return

        emb_path = f"{self.data_dir}/{domain}_corpus.npy"
        ids_path = f"{self.data_dir}/{domain}_ids.json"
        # task_A uses _text_map.json format
        texts_path = f"{self.data_dir}/{domain}_text_map.json"

        # Check if pre-computed embeddings exist
        if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(texts_path):
            print(f"Loading pre-computed embeddings for {domain}...")
            self.corpus_embeddings[domain] = np.load(emb_path)
            with open(ids_path, 'r') as f:
                self.corpus_ids[domain] = json.load(f)
            with open(texts_path, 'r') as f:
                self.corpus_texts[domain] = json.load(f)
        else:
            # Compute embeddings
            print(f"Computing embeddings for {domain}...")
            corpus_path = CORPUS_PATHS[domain]

            documents = []
            with open(corpus_path, 'r') as f:
                for line in f:
                    documents.append(json.loads(line))

            ids = []
            texts = []
            text_map = {}

            for doc in documents:
                doc_id = doc['_id']
                title = doc.get('title', '')
                content = doc.get('text', '')
                full_text = f"{title}\n{content}" if title else content

                ids.append(doc_id)
                texts.append(full_text)
                text_map[doc_id] = full_text

            print(f"Encoding {len(texts)} documents...")
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )

            # Save
            np.save(emb_path, embeddings)
            with open(ids_path, 'w') as f:
                json.dump(ids, f)
            with open(texts_path, 'w') as f:
                json.dump(text_map, f)

            self.corpus_embeddings[domain] = embeddings
            self.corpus_ids[domain] = ids
            self.corpus_texts[domain] = text_map


        print(f"Building FAISS index for {domain}...")
        import faiss
        d = self.corpus_embeddings[domain].shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(self.corpus_embeddings[domain])
        self.faiss_indices[domain] = index

    def _rewrite_query(self, conversation: List[Dict]) -> str:
        """Rewrite the last user turn to be standalone using LLM."""
        # Format conversation
        conv_str = ""
        for turn in conversation:
            speaker = turn.get('speaker', 'user')
            text = turn.get('text', '')
            conv_str += f"{speaker.capitalize()}: {text}\n"

        prompt = QUERY_REWRITE_PROMPT.format(conversation=conv_str.strip())

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.rewrite_client.chat_completion(
                messages,
                max_tokens=256,
                temperature=0.1
            )
            rewritten = response.choices[0].message.content.strip()
            # Clean up potential formatting
            rewritten = rewritten.replace("Rewritten question:", "").strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            return rewritten
        except Exception as e:
            print(f"Query rewrite failed: {e}")
            # Fall back to last user turn
            for turn in reversed(conversation):
                if turn.get('speaker') == 'user':
                    return turn.get('text', '')
            return ""

    def _get_query_text(self, task: Dict) -> str:
        """Extract and optionally rewrite query from task."""
        conversation = task.get('input', [])

        # Get last user turn
        last_user_text = ""
        for turn in reversed(conversation):
            if turn.get('speaker') == 'user':
                last_user_text = turn.get('text', '')
                break

        # Check if rewrite is needed (multi-turn conversation)
        if self.use_query_rewrite and len(conversation) > 1:
            return self._rewrite_query(conversation)

        return last_user_text

    def retrieve(
        self,
        task: Dict,
        top_k_initial: int = 50,
        top_k_final: int = 5
    ) -> List[Dict]:
        """
        Retrieve passages for a task.
        Returns list of context dicts with document_id, text, score.
        """
        collection = task.get('Collection', '')
        domain = COLLECTION_TO_DOMAIN.get(collection)

        if not domain:
            print(f"Unknown collection: {collection}")
            return []

        # Ensure corpus is loaded
        self._load_corpus(domain)

        # Get query text
        query_text = self._get_query_text(task)
        if not query_text:
            return []

        # Encode query with BGE prefix
        query_with_prefix = f"Represent this sentence for searching relevant passages: {query_text}"
        query_embedding = self.encoder.encode(
            [query_with_prefix],
            normalize_embeddings=True
        )

        # Search
        scores, indices = self.faiss_indices[domain].search(query_embedding, top_k_initial)

        # Build initial results
        candidates = []
        for i in range(top_k_initial):
            idx = indices[0][i]
            if idx < len(self.corpus_ids[domain]):
                doc_id = self.corpus_ids[domain][idx]
                doc_text = self.corpus_texts[domain].get(doc_id, "")
                candidates.append({
                    'document_id': doc_id,
                    'text': doc_text,
                    'score': float(scores[0][i])
                })

        # Rerank if enabled
        if self.use_reranker and candidates:
            pairs = [[query_text, c['text']] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for i, score in enumerate(rerank_scores):
                candidates[i]['score'] = float(score)

            candidates.sort(key=lambda x: x['score'], reverse=True)

        # Return top_k_final
        return candidates[:top_k_final]


def create_input_file_from_rag(rag_file: str, output_file: str):
    """
    Create a clean input file for Task C from RAG.jsonl.
    Removes contexts, targets, enrichments to create proper input format.
    """
    with open(rag_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            task = json.loads(line)
            # Keep only necessary fields for input
            clean_task = {
                'conversation_id': task.get('conversation_id'),
                'task_id': task.get('task_id'),
                'task_type': task.get('task_type', 'rag'),
                'turn': task.get('turn'),
                'Collection': task.get('Collection'),
                'dataset': task.get('dataset'),
                'input': task.get('input', [])
            }
            f_out.write(json.dumps(clean_task) + '\n')
