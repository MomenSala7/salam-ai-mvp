import json
import os
import numpy as np
import streamlit as st

from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"
SIMILARITY_THRESHOLD = 0.50
KNOWLEDGE_BASE_PATH = "knowledge_base.json"


@st.cache_resource
def load_knowledge_index(api_key, knowledge_base_mtime):
    """Load verified documents and create embeddings once per KB version."""

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        all_documents = json.load(file)

    documents = [
        doc for doc in all_documents
        if doc.get("verification_status") == "verified"
    ]

    if not documents:
        raise ValueError(
            "No verified documents found in knowledge_base.json"
        )

    required_fields = [
        "id", "title", "source", "url", "category", "content"
    ]

    for doc in documents:
        missing_fields = [
            field for field in required_fields if not doc.get(field)
        ]
        if missing_fields:
            raise ValueError(
                f'Document {doc.get("id", "unknown")} is missing: '
                f'{", ".join(missing_fields)}'
            )

    client = genai.Client(api_key=api_key)

    embedding_texts = [
        (
            f'Title: {doc["title"]}\n'
            f'Category: {doc["category"]}\n'
            f'Authority scope: {doc.get("authority_scope", "")}\n'
            f'Content: {doc["content"]}'
        )
        for doc in documents
    ]

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=embedding_texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    vectors = np.array(
        [embedding.values for embedding in result.embeddings],
        dtype=np.float32
    )

    if len(vectors) != len(documents):
        raise ValueError(
            "The number of embeddings does not match the documents."
        )

    return client, documents, vectors


def retrieve_context(
    question,
    api_key,
    top_k=1,
    threshold=SIMILARITY_THRESHOLD
):
    """Retrieve relevant verified documents for a student's question."""

    if not question or not question.strip():
        return "", []

    if not api_key:
        raise ValueError("Gemini API key is missing.")

    # A changed knowledge-base file gets a fresh cached embedding index.
    knowledge_base_mtime = os.path.getmtime(KNOWLEDGE_BASE_PATH)

    client, documents, document_vectors = load_knowledge_index(
        api_key,
        knowledge_base_mtime
    )

    query_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question.strip(),
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    if not query_result.embeddings:
        return "", []

    query_vector = np.array(
        query_result.embeddings[0].values,
        dtype=np.float32
    )

    document_norms = np.linalg.norm(document_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        return "", []

    scores = np.dot(document_vectors, query_vector) / (
        document_norms * query_norm + 1e-10
    )

    ranked_indices = np.argsort(scores)[::-1]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < threshold:
            continue

        doc = documents[index]

        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "url": doc["url"],
            "category": doc["category"],
            "authority_scope": doc.get("authority_scope", ""),
            "content": doc["content"],
            # Internal evaluation only; app.py does not display this.
            "score": score
        })

        if len(results) >= max(1, top_k):
            break

    context_parts = []

    for i, item in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE [{i}]
Title: {item["title"]}
Organization: {item["source"]}
Category: {item["category"]}
Authority Scope: {item["authority_scope"]}
URL: {item["url"]}
Information: {item["content"]}
"""
        )

    return "\n\n".join(context_parts), results
