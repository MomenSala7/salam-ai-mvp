import os
import re
import streamlit as st
from google import genai
from google.genai import types
from rag import retrieve_context


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salam AI | Golden Visa Assistant",
    page_icon="🎓",
    layout="centered"
)


# ---------------- DESIGN ----------------
st.markdown("""
<style>
.main {
    max-width: 900px;
    margin: auto;
}
.hero {
    background: linear-gradient(135deg, #102C46, #176B68);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #E4F1F1;
}
.disclaimer {
    background: #FFF7E6;
    color: #624A1E;
    padding: 12px;
    border-radius: 10px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- API CONFIG ----------------
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")


# ---------------- LANGUAGE DETECTION ----------------
def is_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


# ---------------- QUESTION CLASSIFICATION ----------------
def is_fee_question(question):
    q = question.lower()

    # Match explicit fee/cost wording, not every use of "how much".
    english_patterns = [
        r"\bfees?\b",
        r"\bcosts?\b",
        r"\bprices?\b",
        r"\bapplication charges?\b",
        r"\bhow much\b.{0,60}\b(cost|fee|charge|price)\b",
        r"\bwhat is the\b.{0,30}\b(fee|cost|price)\b"
    ]

    arabic_keywords = [
        "رسوم",
        "تكلفة",
        "تكاليف",
        "سعر",
        "بكام",
        "كم يكلف",
        "كم تبلغ رسوم",
        "كم رسوم"
    ]

    return (
        any(re.search(pattern, q) for pattern in english_patterns)
        or any(word in q for word in arabic_keywords)
    )


def is_guarantee_question(question):
    q = question.lower()

    english_patterns = [
        r"\bguarantee\b",
        r"\bguaranteed\b",
        r"\bcan you assure\b",
        r"\bconfirm that i am eligible\b",
        r"\bconfirm my eligibility\b"
    ]

    arabic_keywords = [
        "تضمن",
        "ضامن",
        "تضمنلي",
        "تأكد أني مؤهل",
        "تأكيد أهليتي"
    ]

    return (
        any(re.search(pattern, q) for pattern in english_patterns)
        or any(word in q for word in arabic_keywords)
    )


# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image("logo.png", use_container_width=True)

st.markdown("""
<div class="hero">
    <h3>Smart Golden Visa Student Assistant</h3>
    <p>Your intelligent guide to understanding the UAE Golden Visa.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- SYSTEM INSTRUCTION ----------------
SYSTEM_INSTRUCTION = """
You are Salam AI, a professional and friendly student assistant
specialized in explaining the UAE Golden Visa.

LANGUAGE:
- Answer in the same language used by the student.
- Support both Arabic and English.

RELIABILITY:
- Use retrieved official-source information as the primary basis
  for specific Golden Visa requirements.
- Do not invent eligibility requirements, GPA thresholds, fees,
  processing times, or legal conditions.
- Do not claim that a student is officially eligible.
- Do not treat missing information as proof that a requirement
  does not exist.
- If the retrieved source does not answer the question, clearly say
  that the current knowledge base does not contain enough information.
- Direct the student to the relevant official authority when needed.
- Distinguish official information from general explanations.
- Do not combine requirements from different student categories.
- Do not apply a requirement to a category unless the retrieved
  source explicitly supports that category.

SOURCE ACCURACY:
- A retrieved document is not automatically relevant to every question.
- Use a source only if its actual content directly supports the answer.
- Do not cite a source just because it was retrieved.
- If no retrieved source supports the answer, do not use [1].
- Never invent a source, citation, or URL.

STYLE:
- Keep answers clear, organized, and student-friendly.
- Ask follow-up questions when necessary.
- Provide preliminary educational guidance only.
- Never make official eligibility decisions.
"""


# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("logo.png", width=180)

    st.caption("Student Golden Visa Assistant")
    st.divider()

    st.markdown("### About")

    st.write(
        "An AI-powered assistant designed to help students "
        "understand UAE Golden Visa information."
    )

    st.markdown("### Official references")

    st.link_button(
        "GDRFA Dubai",
        "https://www.gdrfad.gov.ae/"
    )

    st.link_button(
        "ICP UAE",
        "https://icp.gov.ae/en/services/uae-golden-residency/"
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Salam AI MVP | Educational prototype")


# ---------------- DISPLAY SOURCES ----------------
def display_sources(sources):
    if not sources:
        return

    with st.expander("📚 Sources / المصادر"):
        for index, source in enumerate(sources, start=1):

            title = source.get("title", "Official source")
            url = source.get("url", "")
            organization = source.get(
                "source",
                "Official authority"
            )

            if url:
                st.markdown(
                    f'**[{index}] [{title}]({url})**'
                )
            else:
                st.markdown(f"**[{index}] {title}**")

            st.caption(organization)


# ---------------- CHAT DISPLAY ----------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))


# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask Salam AI about the Golden Visa...")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        sources = []

        if not api_key:

            answer = (
                "The AI service is not configured yet. "
                "Please add the Gemini API key in the application settings."
            )

            st.warning(answer)

        else:

            try:

                # 1. Handle questions that do not need RAG
                if is_fee_question(prompt):

                    if is_arabic(prompt):
                        answer = (
                            "قاعدة المعرفة الحالية لا تحتوي على معلومات "
                            "موثقة كافية عن رسوم التقديم على الإقامة الذهبية. "
                            "يرجى مراجعة الهيئة الاتحادية للهوية والجنسية "
                            "والجمارك وأمن المنافذ (ICP)، أو الجهة المختصة "
                            "في الإمارة، لمعرفة الرسوم الرسمية المحدثة."
                        )
                    else:
                        answer = (
                            "The current knowledge base does not contain "
                            "verified information about Golden Visa application "
                            "fees. Please check directly with the Federal "
                            "Authority for Identity, Citizenship, Customs "
                            "and Port Security (ICP), or the relevant "
                            "emirate authority, for the latest official fees."
                        )

                    sources = []
                    st.markdown(answer)

                elif is_guarantee_question(prompt):

                    if is_arabic(prompt):
                        answer = (
                            "لا، لا يمكنني ضمان أهليتك الرسمية للحصول على "
                            "الإقامة الذهبية في الإمارات. أنا أقدم إرشادات "
                            "تعليمية أولية فقط، والقرار النهائي بشأن الأهلية "
                            "يعود إلى الجهات الإماراتية المختصة. "
                            "يمكنني مساعدتك في مراجعة الشروط المنشورة "
                            "ومقارنتها بمعلوماتك."
                        )
                    else:
                        answer = (
                            "No, I cannot guarantee that you are officially "
                            "eligible for the UAE Golden Visa. I provide "
                            "preliminary educational guidance only. The final "
                            "eligibility decision is made by the relevant "
                            "UAE authorities. I can help you review the "
                            "published requirements against your information."
                        )

                    sources = []
                    st.markdown(answer)

                else:

                    # 2. Retrieve relevant verified knowledge
                    with st.spinner(
                        "Searching official knowledge sources..."
                    ):

                        retrieved_context, sources = retrieve_context(
                            question=prompt,
                            api_key=api_key,
                            top_k=1
                        )

                    # 3. Check retrieval results
                    if not sources or not retrieved_context.strip():

                        if is_arabic(prompt):
                            answer = (
                                "لم أجد معلومات موثقة ومرتبطة بسؤالك "
                                "بدرجة كافية في قاعدة المعرفة الحالية. "
                                "يرجى مراجعة موقع ICP أو GDRFA الرسمي "
                                "للحصول على أحدث المعلومات."
                            )
                        else:
                            answer = (
                                "I couldn't find sufficiently relevant "
                                "information in my current knowledge base "
                                "to answer this question reliably. "
                                "Please check the official ICP or GDRFA "
                                "website for the latest information."
                            )

                        sources = []
                        st.markdown(answer)

                    else:

                        # 4. Prepare recent conversation
                        conversation = []

                        for item in st.session_state.messages[-10:]:

                            conversation.append(
                                f'{item["role"].upper()}: '
                                f'{item["content"]}'
                            )

                        # 5. Generate answer using Gemini
                        client = genai.Client(api_key=api_key)

                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",

                            contents=(
                                SYSTEM_INSTRUCTION
                                + "\n\nRETRIEVED OFFICIAL KNOWLEDGE:\n"
                                + retrieved_context
                                + "\n\nCONVERSATION:\n"
                                + "\n".join(conversation)
                                + "\n\nASSISTANT:"
                            ),

                            config=types.GenerateContentConfig(
                                temperature=0.2,
                                max_output_tokens=700
                            )
                        )

                        answer = response.text or (
                            "I could not generate an answer. "
                            "Please try again."
                        )

                        st.markdown(answer)

                        display_sources(sources)

            except Exception as error:

                if is_arabic(prompt):
                    answer = (
                        "عذرًا، حدث خطأ أثناء معالجة سؤالك. "
                        "يرجى المحاولة مرة أخرى."
                    )
                else:
                    answer = (
                        "Sorry, the AI service encountered an error. "
                        "Please try again. If the problem continues, "
                        "check the API configuration."
                    )

                sources = []

                st.error(answer)
                st.caption(f"Technical details: {error}")

    # 6. Save answer and its sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
