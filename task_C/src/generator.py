"""
Generator module for Task C: Full RAG Pipeline
Implements answer generation based on retrieved passages
"""

import os
import torch
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .prompts import GENERATION_PROMPT, GENERATION_PROMPT_SIMPLE


class RAGGenerator:
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_local: bool = False,
        device: str = None,
        api_key: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        self.use_local = use_local

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        if self.use_local:
            print(f"Loading model {model_id} locally on {self.device}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.api_key)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=self.device if self.device != 'mps' else 'auto',
                    torch_dtype=torch.float16,
                    token=self.api_key
                )
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map=self.device if self.device != 'mps' else 'auto'
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
        else:
            if not self.api_key:
                raise ValueError("HF_API_KEY environment variable required for API usage.")
            print(f"Using HuggingFace API with model: {model_id}")
            self.client = InferenceClient(model=model_id, token=self.api_key)

    def _call_model(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        messages = [{"role": "user", "content": prompt}]

        if self.use_local:
            try:
                outputs = self.pipe(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                return outputs[0]['generated_text'][-1]['content']
            except Exception as e:
                print(f"Local model error: {e}")
                return "I do not have specific information to answer this question."
        else:
            try:
                response = self.client.chat_completion(
                    messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"API error: {e}")
                return "I do not have specific information to answer this question."

    def generate(self, task: Dict, contexts: List[Dict]) -> str:
        """
        Generate an answer for the task using retrieved contexts.

        Args:
            task: Task dict containing 'input' (conversation history)
            contexts: List of retrieved passage dicts with 'text' field

        Returns:
            Generated answer string
        """
        # Extract question (last user turn)
        conversation = task.get('input', [])
        question = ""
        for turn in reversed(conversation):
            if turn.get('speaker') == 'user':
                question = turn.get('text', '')
                break

        if not question:
            return "I do not have specific information to answer this question."

        # Handle empty contexts
        if not contexts:
            return "I do not have specific information to answer this question."

        # Format context passages
        context_str = ""
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get('text', '')
            title = ctx.get('title', '')
            if title:
                context_str += f"[Passage {i}] {title}\n{text}\n\n"
            else:
                context_str += f"[Passage {i}]\n{text}\n\n"

        # Format conversation history (excluding last question)
        history_str = ""
        for turn in conversation[:-1]:
            speaker = turn.get('speaker', 'user')
            text = turn.get('text', '')
            history_str += f"{speaker.capitalize()}: {text}\n"

        # Use full prompt if there's history, simple prompt otherwise
        if history_str.strip():
            prompt = GENERATION_PROMPT.format(
                context=context_str.strip(),
                history=history_str.strip(),
                question=question
            )
        else:
            prompt = GENERATION_PROMPT_SIMPLE.format(
                context=context_str.strip(),
                question=question
            )

        response = self._call_model(prompt)

        # Clean up response
        response = response.strip()

        # Remove potential "Answer:" prefix if the model repeats it
        if response.lower().startswith("answer:"):
            response = response[7:].strip()

        return response


class SimpleGenerator:
    """
    A simpler generator that uses pre-defined templates without an LLM.
    Useful for testing or when API is not available.
    """

    def generate(self, task: Dict, contexts: List[Dict]) -> str:
        """Generate a simple extractive answer from contexts."""
        if not contexts:
            return "I do not have specific information to answer this question."

        # Simple approach: return first few sentences from top context
        top_context = contexts[0].get('text', '')

        # Get first ~200 characters as a snippet
        if len(top_context) > 200:
            # Find sentence boundary
            end_idx = top_context.find('.', 150)
            if end_idx == -1:
                end_idx = 200
            snippet = top_context[:end_idx + 1]
        else:
            snippet = top_context

        return snippet if snippet else "I do not have specific information to answer this question."
