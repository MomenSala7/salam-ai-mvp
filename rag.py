"""Retrieval for Salam AI's verified UAE Golden Residency knowledge base."""

import json
import os
import re
import unicodedata

import numpy as np
import streamlit as st
from google import genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-001"
SIMILARITY_THRESHOLD = 0.30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base.json")


def _normalize(text):
    """Normalize Arabic/English text for simple keyword matching."""
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}))
    text = re.sub(r"[^\w\s%]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(normalized_query, terms):
    return any(_normalize(term) in normalized_query for term in terms if term)


def _query_routes(question):
    """Return tightly scoped document categories for clear question intents."""
    q = _normalize(question)
    routes = []

    fee_terms = ["رسوم", "تكلفه", "تكاليف", "بكام", "سعر", "كم يكلف", "fees", "fee", "cost", "price", "charges"]
    renewal_terms = ["تجديد", "تمديد", "انتهاء الاقامه", "صلاحية الاقامه", "renew", "renewal", "extend", "extension", "expiry"]
    docs_terms = ["الاوراق", "المستندات", "الوثائق", "المطلوب للتقديم", "جواز السفر", "documents", "document", "papers", "paperwork", "required documents"]
    process_terms = ["ازاي اقدم", "كيف اقدم", "طريقة التقديم", "خطوات التقديم", "اجراءات التقديم", "how to apply", "application steps", "apply for", "application process", "procedure"]
    ordinary_student_terms = ["اقامة دراسية", "اقامة الطالب", "تأشيرة الطالب", "فيزا طالب", "student visa", "study visa", "student residence"]

    is_fee = _contains_any(q, fee_terms)
    is_renewal = _contains_any(q, renewal_terms)
    is_docs = _contains_any(q, docs_terms)
    is_process = _contains_any(q, process_terms)
    is_dubai = "دبي" in q or "dubai" in q or "amer" in q or "آمر" in q
    other_emirate_terms = [
        "ابوظبي", "abu dhabi", "الشارقه", "sharjah", "عجمان", "ajman",
        "راس الخيمه", "ras al khaimah", "الفجيره", "fujairah",
        "ام القيوين", "umm al quwain"
    ]
    is_other_emirate = _contains_any(q, other_emirate_terms)

    # A specific student category is identified independently of the service intent.
    secondary_terms = ["الثانويه", "الصف الثاني عشر", "الثانوية العامة", "طلاب المدارس", "طالب مدرسة", "high school", "secondary school", "grade 12"]
    uae_grad_terms = ["جامعات الامارات", "جامعات الدولة", "جامعة داخل الامارات", "جامعة في الامارات", "جامعة بالامارات", "خريج جامعة", "خريج من جامعة الامارات", "خريج جامعة اماراتية", "خريجي الجامعات المحلية", "جامعات محلية", "category a", "category b", "الفئة a", "الفئة b", "فئة a", "فئة b", "تصنيفها a", "تصنيفها b", "uae university", "local university"]
    international_terms = ["الجامعات العالمية", "الجامعات الدولية", "جامعة خارج الامارات", "خارج الامارات", "خريج جامعة دولية", "جامعة اجنبية", "top 100", "أفضل 100", "افضل 100", "international university", "foreign university", "outside the uae", "abroad", "في مصر", "مصر", "السعودية", "الاردن", "الكويت", "قطر", "البحرين", "عمان", "لبنان", "العراق", "سوريا", "egypt", "saudi arabia", "jordan", "kuwait", "qatar", "bahrain", "oman", "lebanon", "iraq", "syria", "united kingdom", "uk", "united states", "usa", "canada", "india", "pakistan"]

    if is_fee:
        routes.append("gdrfa_dubai_fees_process")
    elif is_renewal:
        routes.extend(["icp_general_renewal", "gdrfa_dubai_renewal_validity"])
    elif is_docs:
        routes.append("gdrfa_dubai_documents")
        if _contains_any(q, secondary_terms):
            routes.append("secondary_students")
        elif _contains_any(q, international_terms):
            routes.append("international_graduates")
        elif _contains_any(q, uae_grad_terms):
            routes.append("uae_university_graduates")
    elif is_process:
        routes.append("gdrfa_dubai_fees_process")
        routes.append("gdrfa_dubai_documents")

    if not routes:
        if _contains_any(q, secondary_terms):
            routes.append("secondary_students")
        elif _contains_any(q, international_terms):
            routes.append("international_graduates")
        elif _contains_any(q, uae_grad_terms):
            routes.append("uae_university_graduates")
        elif _contains_any(q, ordinary_student_terms):
            routes.extend(["ordinary_student_residence", "student_golden_residency_overview"])
        elif _contains_any(q, ["الطلاب", "الطلبة", "طالب", "خريجين", "خريجو", "student", "students", "graduates", "eligibility", "شروط", "golden visa", "golden residency", "الإقامة الذهبية"]):
            routes.extend([
                "student_golden_residency_overview",
                "secondary_students",
                "uae_university_graduates",
                "international_graduates",
            ])

    # Keep order while removing duplicates.
    return list(dict.fromkeys(routes))


@st.cache_resource

def load_knowledge_index(api_key, knowledge_base_mtime):
    """Load verified official documents and embed them once per KB version."""
    del knowledge_base_mtime  # Included in the cache key to invalidate stale embeddings.

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        all_documents = json.load(file)

    documents = [
        doc for doc in all_documents
        if doc.get("verification_status") == "verified"
    ]
    if not documents:
        raise ValueError("No verified documents found in knowledge_base.json")

    required_fields = ["id", "title", "source", "url", "category", "content"]
    for doc in documents:
        missing = [field for field in required_fields if not doc.get(field)]
        if missing:
            raise ValueError(
                f"Document {doc.get('id', 'unknown')} is missing: {', '.join(missing)}"
            )

    client = genai.Client(api_key=api_key)
    texts = []
    for doc in documents:
        keyword_text = " ".join(
            doc.get("keywords_ar", []) + doc.get("keywords_en", [])
        )
        texts.append(
            f"Title: {doc['title']}\n"
            f"Category: {doc['category']}\n"
            f"Jurisdiction: {doc.get('jurisdiction', '')}\n"
            f"Scope: {doc.get('authority_scope', '')}\n"
            f"Keywords: {keyword_text}\n"
            f"Content: {doc['content']}"
        )

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    vectors = np.asarray(
        [embedding.values for embedding in result.embeddings],
        dtype=np.float32,
    )
    if len(vectors) != len(documents):
        raise ValueError("The number of embeddings does not match the documents.")

    return client, documents, vectors


def _keyword_score(question, doc):
    q = _normalize(question)
    keywords = doc.get("keywords_ar", []) + doc.get("keywords_en", [])
    hits = 0
    for keyword in keywords:
        normalized_keyword = _normalize(keyword)
        if normalized_keyword and normalized_keyword in q:
            hits += 1
    # Small additive boost; embeddings remain the main relevance signal.
    return min(0.24, hits * 0.06), hits


def retrieve_context(
    question,
    api_key,
    top_k=3,
    threshold=SIMILARITY_THRESHOLD,
):
    """Retrieve relevant verified sources using intent routing + multilingual embeddings."""
    if not question or not question.strip():
        return "", []

    routes = _query_routes(question)
    normalized_question = _normalize(question)
    other_emirate_terms = [
        "ابوظبي", "abu dhabi", "الشارقه", "sharjah", "عجمان", "ajman",
        "راس الخيمه", "ras al khaimah", "الفجيره", "fujairah",
        "ام القيوين", "umm al quwain"
    ]
    is_other_emirate = _contains_any(normalized_question, other_emirate_terms)
    service_specific_routes = {
        "gdrfa_dubai_documents",
        "gdrfa_dubai_fees_process",
        "gdrfa_dubai_renewal_validity",
    }
    if is_other_emirate:
        if any(route in {"gdrfa_dubai_fees_process", "gdrfa_dubai_documents"} for route in routes):
            return "", []
        routes = [route for route in routes if route not in service_specific_routes]

    kb_mtime = os.path.getmtime(KNOWLEDGE_BASE_PATH)
    client, documents, document_vectors = load_knowledge_index(api_key, kb_mtime)

    query_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not query_result.embeddings:
        return "", []

    query_vector = np.asarray(query_result.embeddings[0].values, dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    if query_norm == 0:
        return "", []

    cosine_scores = np.dot(document_vectors, query_vector) / (
        document_norms * query_norm + 1e-10
    )

    ranked = []
    for index, doc in enumerate(documents):
        keyword_boost, keyword_hits = _keyword_score(question, doc)
        route_boost = 0.30 if doc.get("category") in routes else 0.0
        combined_score = float(cosine_scores[index]) + keyword_boost + route_boost
        ranked.append((combined_score, float(cosine_scores[index]), keyword_hits, doc))

    # If the question clearly maps to an intent/category, do not return unrelated categories.
    if routes:
        routed = [item for item in ranked if item[3].get("category") in routes]
        if routed:
            ranked = routed

    ranked.sort(key=lambda item: item[0], reverse=True)

    results = []
    for combined_score, cosine_score, keyword_hits, doc in ranked:
        # A clearly routed exact document or strong keyword match can pass even if
        # semantic similarity is modest; unrelated low-score docs are rejected.
        routed_match = bool(routes and doc.get("category") in routes)
        if cosine_score < threshold and keyword_hits == 0 and not routed_match:
            continue
        if cosine_score < 0.20 and keyword_hits == 0 and not routed_match:
            continue

        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "url": doc["url"],
            "category": doc["category"],
            "authority_scope": doc.get("authority_scope", ""),
            "jurisdiction": doc.get("jurisdiction", ""),
            "content": doc["content"],
            "score": combined_score,
        })
        if len(results) >= max(1, top_k):
            break

    context_parts = []
    for index, item in enumerate(results, start=1):
        context_parts.append(
            f"SOURCE [{index}]\n"
            f"Title: {item['title']}\n"
            f"Organization: {item['source']}\n"
            f"Category: {item['category']}\n"
            f"Authority scope: {item['authority_scope']}\n"
            f"Jurisdiction: {item['jurisdiction']}\n"
            f"URL: {item['url']}\n"
            f"Information: {item['content']}"
        )

    return "\n\n".join(context_parts), results
