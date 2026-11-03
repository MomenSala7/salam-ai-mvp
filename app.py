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


def render_chat_text(text):
    """Render chat text with direction and typography based on its language."""
    direction_class = "salam-rtl-marker" if is_arabic(text) else "salam-ltr-marker"
    st.markdown(
        f'<span class="{direction_class}" aria-hidden="true"></span>',
        unsafe_allow_html=True
    )
    st.markdown(text)


# ---------------- QUESTION CLASSIFICATION ----------------
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
You are Salam AI, a professional student assistant specialized in the UAE Golden Residency for outstanding students and graduates.

LANGUAGE:
- Answer in the same language used by the student.
- Support Arabic and English.
- For Arabic, use clear Modern Standard Arabic and natural sentence structure.

ROLE AND ANSWER QUALITY:
- Give a useful, direct, sufficiently detailed answer whenever the retrieved official sources support it.
- Do not answer with a generic referral to an official website when the retrieved sources already contain the requested facts.
- Explain the relevant requirements, academic thresholds, documents, application steps, fees, and service timing when those details are present in the retrieved sources and relevant to the question.
- Do not dump every detail into every answer. Select only the details needed to answer the student's actual question.
- If the question is broad (for example, "What are the requirements for students?"), explain that there are separate categories and summarize the distinctions between secondary-school students, graduates of UAE universities, and graduates of international universities. Do not mix their requirements. Ask which category applies only after giving this useful overview.
- If the student's category is clear, answer for that category only.
- If an essential detail is missing and the answer depends on it, state the relevant distinction and ask one concise follow-up question.

OFFICIAL ACCURACY:
- Treat the retrieved official-source content as the primary and controlling basis for specific requirements.
- Never invent or estimate GPA thresholds, percentages, fees, processing times, required documents, or legal conditions.
- Preserve all numbers, units, conditions, and category distinctions exactly as stated in the sources.
- Do not claim that a student is officially eligible or guarantee approval.
- Do not treat missing information as proof that a requirement does not exist.
- Distinguish verified official facts from general guidance.
- If the retrieved sources do not answer the question, say clearly what specific information is unavailable and direct the student to the relevant authority.

JURISDICTION AND SERVICE SCOPE:
- GDRFA Dubai service details, document checklists, fees, processing time, and renewal statements apply specifically to the Dubai service. Explicitly say "وفق خدمة GDRFA دبي" / "For the GDRFA Dubai service" when using them.
- Never present the Dubai fee schedule as a UAE-wide fee schedule.
- The AED 500 item is labelled by GDRFA Dubai as a fee inside the country. Do not imply that it applies to every applicant.
- Do not calculate or promise a final total payment. Tell the user to confirm the exact amount in the official payment screen because it may depend on the application circumstances.
- If a student asks about another emirate, do not use Dubai-only fees or application rules as if they apply there.
- Keep ordinary student residence visas separate from the Golden Residency for outstanding students.
- For renewal questions, distinguish ICP's general programme description (automatic renewal) from GDRFA Dubai's student-service-specific statement (student permits are not renewed after expiry / are non-extendable under the stated service details). Explain that the published pages do not reconcile this scope difference. Do not promise renewal; advise the student to confirm the procedure for their specific file with the competent authority. Do not generalize the GDRFA Dubai statement to other emirates or all Golden Residency categories.

SOURCE ACCURACY:
- A retrieved document is not automatically relevant to every question.
- Use a source only when its actual content supports the answer.
- Cite only the supplied source labels such as [1] or [2]. Never invent a citation, source, or URL.
- If no retrieved source supports a claim, do not cite it.
- When sources have a limited jurisdiction, explain that limitation in the answer.

RESPONSE STYLE:
- Write natural, connected paragraphs in a professional, helpful tone.
- Do not use numbered lists (1, 2, 3).
- Prefer short paragraphs and descriptive wording. Avoid excessive headings and avoid unnecessary bullet lists.
- Do not place numbers beside each requirement as a numbered list. Keep official numbers such as 95%, 3.5, 3.8, AED 1,100 and 10 years when they are source-supported.
- Keep the answer easy for a student to understand without changing the official meaning.
- Ask follow-up questions only when genuinely necessary.
- Provide preliminary educational guidance only; final eligibility and approval are determined by the competent authorities.
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

            jurisdiction = source.get("jurisdiction", "")
            scope = source.get("authority_scope", "")
            caption_parts = [organization]
            if jurisdiction:
                caption_parts.append(jurisdiction)
            if scope:
                caption_parts.append(scope)
            st.caption(" • ".join(caption_parts))


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

        if not api_key:

            answer = (
                "The AI service is not configured yet. "
                "Please add the Gemini API key in the application settings."
            )

            render_chat_text(answer)

        else:

            try:

                # Handle guarantee questions directly; all factual questions,
                # including fees and application steps, go through official-source RAG.
                if is_guarantee_question(prompt):

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
                        "جارٍ البحث في المصادر الرسمية..." if is_arabic(prompt)
                        else "Searching official knowledge sources..."
                    ):

                        retrieved_context, sources = retrieve_context(
                            question=prompt,
                            api_key=api_key,
                            top_k=4
                        )

                    # 3. Check retrieval results
                    if not sources or not retrieved_context.strip():

                        if is_arabic(prompt):
                            answer = (
                                "لم أعثر في المصادر الرسمية المتاحة لدي على "
                                "معلومات كافية للإجابة عن هذا الجزء من سؤالك بدقة. "
                                "يمكنك تحديد الإمارة أو فئة الطالب، أو مراجعة "
                                "الجهة المختصة للحصول على تفاصيل الطلب المحدثة."
                            )
                        else:
                            answer = (
                                "I could not find enough verified official "
                                "information to answer this part reliably. "
                                "You can specify the emirate or student category, "
                                "or confirm the application details with the "
                                "relevant authority."
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
                                + "\n\nWrite a direct, complete answer using only the relevant retrieved facts. "
                                + "Explain any Dubai-only scope. Do not use numbered lists. "
                                + "Keep official numerical requirements unchanged. "
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
