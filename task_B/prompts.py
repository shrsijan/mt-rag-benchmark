"""
Optimized prompts for Task B: Generation with Reference Passages
Evidence Extraction + Chain-of-Thought approach for improved grounding
"""

# Query resolution for multi-turn conversations
DECOMPOSITION_PROMPT = """Given the conversation and current question, rewrite the question to be self-contained.

Conversation:
{history}

Current question: {question}

Resolve any pronouns (it, they, this, that) using context. Return the resolved question:"""

# Step 1: Evidence extraction - identify relevant sentences
EVIDENCE_EXTRACTION_PROMPT = """Extract the most relevant sentences from the context that could answer the question.

Context:
{context}

Question: {question}

Extract 1-3 sentences that directly relate to the question. If no relevant information exists, respond with "NO_EVIDENCE".

Relevant evidence:"""


GENERATION_PROMPT_EVIDENCE = """You are answering a question based ONLY on the provided evidence. Copy exact phrases from the evidence.

Evidence:
{evidence}

Conversation History:
{history}

Question: {question}

Rules:
1. Use EXACT words and phrases from the evidence in your answer
2. Keep your answer to 1-2 sentences maximum
3. If the evidence says "NO_EVIDENCE" or doesn't contain the specific answer, respond: "I don't have specific information about this."
4. Never add facts not in the evidence

Answer:"""


GENERATION_PROMPT = """Answer the question using the context. Be concise and use exact phrases from the context.

Context:
{context}

History:
{history}

Question: {question}

Give a 1-sentence answer using words from the context. If nothing in the context relates to the question, say "I don't have specific information about this."

Answer:"""

# Few-shot prompt for improved accuracy
GENERATION_PROMPT_FEWSHOT = """Answer questions using ONLY the provided context. Copy exact phrases when possible.

Example 1:
Context: The Pittsburgh Steelers have won six Super Bowl championships, the most by any team in the NFL.
Question: Which team has won the most Super Bowls?
Answer: The Pittsburgh Steelers have won six Super Bowl championships, the most by any team in the NFL.

Example 2:
Context: Climate change can contribute to the intensity of some natural disasters, but the exact relationship varies.
Question: Does climate change cause all natural disasters?
Answer: Climate change can contribute to the intensity and frequency of some natural disasters, but it is not the sole cause of all disasters.

Example 3:
Context: The context discusses NFL teams and their schedules.
Question: What is the population of Tokyo?
Answer: I'm sorry, but I don't have the answer to your question.

Now answer this question:

Context:
{context}

History:
{history}

Question: {question}

Answer (copy exact phrases from context when available):"""

# Extractive prompt - copies full sentences from context
GENERATION_PROMPT_EXTRACTIVE = """Find and copy the sentence(s) from the context that best answer the question.

Context:
{context}

History:
{history}

Question: {question}

Instructions:
1. Find the most relevant sentence(s) in the context
2. Copy that sentence exactly, word-for-word
3. You may combine 2-3 sentences if needed
4. If no sentence answers the question, say: "I'm sorry, but I don't have the answer to your question."

Answer:"""

# Answer-oriented prompt - biased toward providing answers
GENERATION_PROMPT_ANSWER = """Your task is to answer the question using information from the context. Always try to find relevant information.

Context:
{context}

History:
{history}

Question: {question}

IMPORTANT: The context usually contains relevant information. Look carefully for any related facts, even if not a perfect match. Copy exact phrases from the context in your answer.

Only say "I'm sorry, but I don't have the answer to your question." if the context contains absolutely NO information related to the question topic.

Answer:"""

# Best prompt - optimized for high RB_agg
GENERATION_PROMPT_BEST = """Read the context carefully and answer the question. Use exact words and phrases from the context.

Context:
{context}

Conversation history:
{history}

Question: {question}

Requirements:
1. Find sentences in the context that relate to the question
2. Copy those sentences or key phrases directly into your answer
3. Keep your answer concise (1-3 sentences)
4. Only say "I'm sorry, but I don't have the answer to your question." if nothing in the context is relevant

Answer:"""

# Ultra-aggressive prompt - always tries to answer
GENERATION_PROMPT_ULTRA = """Your job is to answer questions by extracting relevant information from the context. ALWAYS provide an answer if there's any remotely relevant information.

Context:
{context}

History:
{history}

Question: {question}

Instructions:
- Search the context for ANY information related to the question
- Even partial or indirect answers are better than saying you don't know
- Copy exact sentences or phrases from the context
- Keep your answer to 1-3 sentences
- ONLY say "I'm sorry, but I don't have the answer to your question." if the context has ZERO relevant content

Provide your answer:"""

# Legacy prompt for fallback
GENERATION_PROMPT_DIRECT = """Answer the question using the context below. Use exact phrases from the context when possible.

Context:
{context}

Conversation History:
{history}

Question: {question}

Instructions:
- Give a direct answer in 1-3 sentences using words from the context
- If the context contains the specific answer, extract and use it
- If the context does NOT have the specific answer to THIS question, say: "I don't have specific information about [topic]."
- Do not add information not found in the context

Answer:"""
