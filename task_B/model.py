"""
Task B Generator for Multi-turn RAG
Clean, optimized implementation for generation with reference passages.
"""

import os
import torch
from typing import List, Dict
from huggingface_hub import InferenceClient, login
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .prompts import DECOMPOSITION_PROMPT, GENERATION_PROMPT, EVIDENCE_EXTRACTION_PROMPT, GENERATION_PROMPT_FEWSHOT, GENERATION_PROMPT_EXTRACTIVE, GENERATION_PROMPT_ANSWER, GENERATION_PROMPT_BEST, GENERATION_PROMPT_ULTRA


def get_hf_token():
    """Get HuggingFace token from various sources."""
    # Check environment variables
    token = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    # Try to get from huggingface_hub's stored token
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    # Try reading from common token file locations
    token_paths = [
        os.path.expanduser("~/.huggingface/token"),
        os.path.expanduser("~/.cache/huggingface/token"),
    ]
    for path in token_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                token = f.read().strip()
                if token:
                    return token

    return None


class TaskBGenerator:
    """
    Generator for Task B: Generation with Reference Passages.
    Supports both HuggingFace API and local model inference.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        use_local: bool = False,
        device: str = "mps",
        api_key: str = None,
        prompt_type: str = "default",
        temperature: float = 0.1
    ):
        self.model_id = model_id
        self.api_key = api_key or get_hf_token()
        self.use_local = use_local
        self.device = device
        self.prompt_type = prompt_type
        self.temperature = temperature

        if self.use_local:
            self._init_local_model()
        else:
            if not self.api_key:
                raise ValueError(
                    "HuggingFace API token not found. Please set one of:\n"
                    "  - HF_API_KEY environment variable\n"
                    "  - HF_TOKEN environment variable\n"
                    "  - HUGGINGFACEHUB_API_TOKEN environment variable\n"
                    "  - Run: huggingface-cli login"
                )
            self.client = InferenceClient(model=model_id, token=self.api_key)

    def _init_local_model(self):
        """Initialize local model for inference."""
        print(f"Loading model {self.model_id} locally on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.api_key)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.float16,
            token=self.api_key
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device
        )

    def _call_model(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Call the LLM with the given prompt."""
        messages = [{"role": "user", "content": prompt}]

        try:
            if self.use_local:
                outputs = self.pipe(
                    messages,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                return outputs[0]['generated_text'][-1]['content']
            else:
                response = self.client.chat_completion(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Model call error: {e}")
            return ""

    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history as a string."""
        if not history:
            return "No previous conversation."

        lines = []
        for turn in history:
            speaker = turn.get('speaker', 'unknown').capitalize()
            text = turn.get('text', '')
            lines.append(f"{speaker}: {text}")

        return "\n".join(lines)

    def _format_contexts(self, contexts: List[Dict]) -> str:
        """Format context passages as a string."""
        if not contexts:
            return "No context available."

        passages = []
        for ctx in contexts:
            text = ctx.get('text', '').strip()
            if text:
                passages.append(text)

        return "\n\n---\n\n".join(passages)

    def _resolve_query(self, query: str, history: List[Dict]) -> str:
        """Resolve references in query using conversation history."""
        if not history:
            return query

        history_str = self._format_history(history)
        prompt = DECOMPOSITION_PROMPT.format(history=history_str, question=query)

        resolved = self._call_model(prompt, max_tokens=128, temperature=0.1).strip()

        # Clean up quotes
        if resolved.startswith('"') and resolved.endswith('"'):
            resolved = resolved[1:-1]

        return resolved if resolved and len(resolved) > 3 else query

    def _extract_evidence(self, context_str: str, query: str) -> str:
        """Extract relevant evidence from context for the question."""
        prompt = EVIDENCE_EXTRACTION_PROMPT.format(
            context=context_str,
            question=query
        )

        evidence = self._call_model(prompt, max_tokens=200, temperature=0.1).strip()

        # Clean up common prefixes
        for prefix in ["Relevant evidence:", "Evidence:", "Here are the relevant sentences:"]:
            if evidence.lower().startswith(prefix.lower()):
                evidence = evidence[len(prefix):].strip()

        return evidence

    def generate(self, task_data: Dict) -> str:
        """Generate an answer using direct extractive approach."""
        input_turns = task_data.get('input', [])
        contexts = task_data.get('contexts', [])

        if not input_turns:
            return "I don't have enough information to answer this question."

        # Extract query and history
        query = input_turns[-1].get('text', '') if input_turns else ''
        history = input_turns[:-1] if len(input_turns) > 1 else []

        # Format components
        history_str = self._format_history(history)
        context_str = self._format_contexts(contexts)

        # Resolve query references for multi-turn
        resolved_query = self._resolve_query(query, history) if history else query

        # Check for empty context
        if not contexts:
            return "I don't have enough information to answer this question."

        # Select prompt based on type
        if self.prompt_type == "fewshot":
            prompt_template = GENERATION_PROMPT_FEWSHOT
        elif self.prompt_type == "extractive":
            prompt_template = GENERATION_PROMPT_EXTRACTIVE
        elif self.prompt_type == "answer":
            prompt_template = GENERATION_PROMPT_ANSWER
        elif self.prompt_type == "best":
            prompt_template = GENERATION_PROMPT_BEST
        elif self.prompt_type == "ultra":
            prompt_template = GENERATION_PROMPT_ULTRA
        else:
            prompt_template = GENERATION_PROMPT

        # Direct extractive generation - single step
        prompt = prompt_template.format(
            context=context_str,
            history=history_str,
            question=resolved_query
        )

        answer = self._call_model(prompt, max_tokens=200, temperature=self.temperature).strip()

        # Clean up prefixes
        for prefix in ["Answer:", "ANSWER:", "Based on the context,"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        return answer if answer else "I don't have specific information about this."


# Backward compatibility alias
HierarchicalMemoryGenerator = TaskBGenerator
