import os
import re
import difflib
import unicodedata
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

/* Chat typography and direction: Arabic RTL, English LTR */
div[data-testid="stChatMessage"]:has(.salam-rtl-marker) [data-testid="stMarkdownContainer"] {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext;
    font-size: 18px !important;
    line-height: 1.9 !important;
}

div[data-testid="stChatMessage"]:has(.salam-rtl-marker) [data-testid="stMarkdownContainer"] p {
    direction: rtl !important;
    text-align: right !important;
    margin-bottom: 0.9rem !important;
    line-height: 1.9 !important;
}

div[data-testid="stChatMessage"]:has(.salam-rtl-marker) [data-testid="stMarkdownContainer"] ul,
div[data-testid="stChatMessage"]:has(.salam-rtl-marker) [data-testid="stMarkdownContainer"] ol {
    direction: rtl !important;
    text-align: right !important;
    padding-right: 1.5rem !important;
    padding-left: 0 !important;
}

div[data-testid="stChatMessage"]:has(.salam-ltr-marker) [data-testid="stMarkdownContainer"] {
    direction: ltr !important;
    text-align: left !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}

div[data-testid="stChatMessage"]:has(.salam-ltr-marker) [data-testid="stMarkdownContainer"] p {
    direction: ltr !important;
    text-align: left !important;
    margin-bottom: 0.75rem !important;
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


# ---------------- APPROVED FAQ RESPONSES ----------------
def normalize_arabic(text):
    """Normalize Arabic text to make FAQ matching tolerant of minor variations."""
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[\u064B-\u065F\u0670ـ]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


APPROVED_FAQS = [
    {
        "questions": [
            "ما شروط الاقامة الذهبي للطلبة",
            "ما شروط الاقامة الذهبية للطلبة",
            "ما شروط الاقامة الذهبية للطلاب",
            "ما هي شروط الاقامة الذهبية للطلبة",
            "شروط الاقامة الذهبية للطلبة في دبي",
        ],
        "answer": (
            "تشمل شروط الإقامة الذهبية للطلبة والخريجين في دبي تحقيق معدل "
            "تراكمي لا يقل عن 95% لطلاب الثانوية، أو 3.5/3.8 لخريجي الجامعات "
            "المحلية والدولية (من أفضل 100 جامعة)، مع ألا يزيد عمر الشهادة "
            "عن سنتين. يجب أن يكون الملف صادرًا عن دبي وموافقًا عليه من "
            "الهيئة الاتحادية للهوية والجنسية، وتصل صلاحية الإقامة إلى "
            "10 سنوات دون تجديد."
        ),
    },
    {
        "questions": [
            "ما هي رسوم استخراج الاقامة الذهبية للطلبة",
            "ما رسوم استخراج الاقامة الذهبية للطلبة",
            "كم رسوم الاقامة الذهبية للطلبة",
            "ما رسوم الاقامة الذهبية للطلاب",
            "تكلفة استخراج الاقامة الذهبية للطلبة",
        ],
        "answer": (
            "تبلغ رسوم استخراج الإقامة الذهبية للطلبة 1100 درهم إماراتي، "
            "بالإضافة إلى رسوم إضافية مثل درهم المعرفة والابتكار ورسوم "
            "التوصيل. يرجى ملاحظة أن رسوم الإصدار تزداد بمقدار 100 درهم "
            "سنوياً في حال تجاوز مدة الإقامة سنتين."
        ),
    },
    {
        "questions": [
            "هل يمكن تجديد الاقامة الذهبية للطلبة",
            "هل يمكن تجديد الاقامة الذهبية للطلاب",
            "هل الاقامة الذهبية للطلبة قابلة للتجديد",
            "هل يتم تجديد الاقامة الذهبية للطلبة",
        ],
        "answer": (
            "نعم، يمكن تجديد الإقامة الذهبية للطلبة والخريجين المتفوقين، "
            "حيث تصل صلاحيتها إلى 10 سنوات دون الحاجة للتجديد السنوي، "
            "مع إمكانية تمديدها وفقاً للشروط المحددة."
        ),
    },
]


def get_approved_faq_answer(question):
    """Return an approved answer for a close FAQ match, otherwise None."""
    normalized_question = normalize_arabic(question)
    if not normalized_question:
        return None

    # These approved responses specifically describe the Dubai process.
    # Do not apply them when a different emirate is explicitly requested.
    other_emirates = (
        "ابوظبي", "الشارقه", "عجمان", "راس الخيمه",
        "الفجيره", "ام القيوين",
    )
    if any(emirate in normalized_question for emirate in other_emirates):
        return None

    best_answer = None
    best_score = 0.0

    for faq in APPROVED_FAQS:
        for known_question in faq["questions"]:
            normalized_known = normalize_arabic(known_question)

            if normalized_question == normalized_known:
                return faq["answer"]

            score = difflib.SequenceMatcher(
                None, normalized_question, normalized_known
            ).ratio()

            # Conservative threshold reduces accidental matches between FAQ topics.
            if score > best_score:
                best_score = score
                best_answer = faq["answer"]

    return best_answer if best_score >= 0.84 else None


def render_chat_text(text):
    """Render chat text with direction and typography based on its language."""
    direction_class = "salam-rtl-marker" if is_arabic(text) else "salam-ltr-marker"
    st.markdown(
        f'<span class="{direction_class}" aria-hidden="true"></span>',
        unsafe_allow_html=True
    )
    st.markdown(text)


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
- When answering in Arabic, use clear and natural Modern Standard Arabic.

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

RESPONSE STYLE:
- Write answers in natural, connected paragraphs.
- Do not use numbered lists (1, 2, 3).
- Avoid bullet points and excessive headings.
- Present requirements as clear sentences instead of separate numbered items.
- Use a professional and formal tone similar to official government
  service information.
- Keep the answer concise, clear, and easy for students to understand.
- Preserve the exact meaning, figures, conditions, and details
  stated in the official source.
- Do not paraphrase in a way that changes the official meaning.
- Do not add unsupported explanations or assumptions.
- For Arabic answers, use natural Arabic sentence structure.
- Do not add English translations unless the student requests them.

CONVERSATION:
- Answer the student's actual question directly whenever supported
  by the available information.
- Ask a follow-up question only when it is genuinely necessary.
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

        render_chat_text(message["content"])

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
        render_chat_text(prompt)

    with st.chat_message("assistant"):

        sources = []

        # Approved FAQ answers take priority over Gemini and RAG.
        # This also allows these exact answers to work without an API key.
        faq_answer = get_approved_faq_answer(prompt)

        if faq_answer:
            answer = faq_answer
            sources = []
            render_chat_text(answer)

        elif not api_key:

            answer = (
                "The AI service is not configured yet. "
                "Please add the Gemini API key in the application settings."
            )

            render_chat_text(answer)

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
                    render_chat_text(answer)

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
                    render_chat_text(answer)

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
                        render_chat_text(answer)

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

                        render_chat_text(answer)

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

                render_chat_text(answer)
                st.caption(f"Technical details: {error}")

    # 6. Save answer and its sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
