# Prompts for Task C: Full RAG Pipeline

QUERY_REWRITE_PROMPT = """Given the following conversation, rewrite the last user utterance into a single, standalone question that incorporates all necessary context from the conversation history.

Conversation:
{conversation}

Instructions:
- Make the question self-contained so it can be understood without the conversation history
- Resolve all pronouns and references (e.g., "it", "they", "he", "that") to their specific referents
- Keep the meaning and intent of the original question
- If the last utterance is already standalone, return it as-is
- Output ONLY the rewritten question, nothing else

Rewritten question:"""

GENERATION_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.

Context passages:
{context}

Conversation history:
{history}

Current question: {question}

Instructions:
- Answer the question using ONLY information from the context passages above
- If the context does not contain enough information to answer the question, respond with: "I do not have specific information to answer this question."
- Be concise, accurate, and directly address the question
- Do not make up or hallucinate any information not present in the context
- If only partial information is available, provide what you can and indicate what is missing
- Keep your answer under 150 words

Answer:"""

GENERATION_PROMPT_SIMPLE = """Given the following context and question, provide a helpful answer.

Context:
{context}

Question: {question}

Instructions:
- Use only information from the context
- If the answer cannot be found in the context, say "I do not have specific information to answer this question."
- Be concise and accurate

Answer:"""
