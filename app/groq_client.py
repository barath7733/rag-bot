"""
Groq LLM client.

Groq is the ONLY LLM provider used in this project. No OpenAI SDK,
models, or API keys are used anywhere.
"""

from __future__ import annotations

import logging

from groq import Groq

from app.config import get_settings

logger = logging.getLogger("rag_chatbot.groq_client")

_client: Groq | None = None

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable general-purpose AI assistant. "
    "Answer the user's question directly and accurately using your own "
    "knowledge. Be clear and concise. If you are not certain about "
    "something, say so honestly instead of guessing."
)

WEB_SYSTEM_PROMPT = (
    "You are a helpful assistant answering using freshly retrieved web "
    "search results provided below. Rules:\n"
    "1. Base your answer on the retrieved web results — they contain "
    "more current information than your own training data.\n"
    "2. Do not invent facts, numbers, or dates not present in the results.\n"
    "3. If the results don't answer the question, say so honestly.\n"
    "4. Be concise and cite which source(s) support key claims by name "
    "when it's natural to do so.\n"
    "5. Note that results may be time-sensitive; do not overstate certainty."
)

RAG_SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the user's question "
    "PRIMARILY using the retrieved context provided below, which comes "
    "from documents the user has uploaded. Rules:\n"
    "1. Base your answer on the retrieved context whenever it is relevant.\n"
    "2. Do not invent, assume, or fabricate any facts, numbers, or details "
    "that are not present in the retrieved context.\n"
    "3. If the retrieved context does not contain enough information to "
    "answer the question, clearly state that the information could not "
    "be found in the uploaded documents, rather than guessing.\n"
    "4. You may use general knowledge only to explain or clarify terms "
    "found in the context — never to supply document-specific facts that "
    "are not in the context.\n"
    "5. Be concise and well-organized."
)


def _get_client() -> Groq:
    global _client
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _run_completion(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean error upstream
        logger.error("Groq API call failed: %s", exc)
        raise RuntimeError(f"The AI provider request failed: {exc}") from exc

    choice = completion.choices[0] if completion.choices else None
    if not choice or not choice.message or not choice.message.content:
        raise RuntimeError("The AI provider returned an empty response.")
    return choice.message.content.strip()


def generate_general_answer(question: str, history: list[dict[str, str]]) -> str:
    """General AI mode: answer directly with no document context."""
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return _run_completion(messages)


def generate_rag_answer(question: str, context: str, history: list[dict[str, str]]) -> str:
    """RAG mode: answer grounded in retrieved document context."""
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Retrieved context from uploaded documents:\n"
                f"---\n{context}\n---\n\n"
                f"Question: {question}"
            ),
        }
    )
    return _run_completion(messages)


def generate_web_answer(question: str, context: str, history: list[dict[str, str]]) -> str:
    """Web Search mode: answer grounded in freshly retrieved web results."""
    messages = [{"role": "system", "content": WEB_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Web search results:\n---\n{context}\n---\n\n"
                f"Question: {question}"
            ),
        }
    )
    return _run_completion(messages)


def classify_intent_needs_documents(question: str) -> bool:
    """
    Lightweight LLM-based classification used only in AUTO mode: does
    this question likely require looking at uploaded documents?

    Falls back to a conservative default (True — try RAG first) if the
    classification call itself fails, since RAG mode gracefully reports
    "not found in documents" when nothing relevant is retrieved.
    """
    settings = get_settings()
    client = _get_client()
    try:
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify whether answering the user's question would "
                        "require looking up information in uploaded documents "
                        "(as opposed to being answerable from general knowledge "
                        "or being small talk). Reply with exactly one word: "
                        "'DOCUMENT' or 'GENERAL'."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=5,
        )
        label = (completion.choices[0].message.content or "").strip().upper()
        return "DOCUMENT" in label
    except Exception as exc:  # noqa: BLE001
        logger.warning("Intent classification failed, defaulting to document search: %s", exc)
        return True
