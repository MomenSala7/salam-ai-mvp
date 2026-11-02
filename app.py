
import os
import streamlit as st
from google import genai
from google.genai import types

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
api_key = None

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎓 Salam AI</h1>
    <h3>Smart Golden Visa Student Assistant</h3>
    <p>Your intelligent guide to understanding the UAE Golden Visa.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<strong>Important:</strong> Salam AI provides educational information
and preliminary guidance only. It does not issue official eligibility
decisions or replace UAE government authorities.
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- KNOWLEDGE CONTEXT ----------------
KNOWLEDGE = """
Project: Salam AI – Smart Golden Visa Student Assistant.

Purpose:
Help university students understand information about the UAE Golden Visa,
provide preliminary educational guidance, and direct students to relevant
official authorities or university departments.

The project proposal identifies the following official reference:
GDRFA Dubai – Golden Residence:
https://www.gdrfad.gov.ae/

Important limitations:
- Do not invent eligibility requirements, GPA thresholds, fees, or processing times.
- Do not claim that a student is officially eligible.
- Do not treat the project proposal as a complete legal requirements database.
- If a specific legal condition is not included in the supplied context,
  clearly say that it needs verification from the official UAE authority.
- Distinguish general information from confirmed official requirements.
- Encourage students to verify current rules with the relevant authority.
"""

SYSTEM_INSTRUCTION = """
You are Salam AI, a professional and friendly student assistant
specialized in explaining the UAE Golden Visa.

Answer in the same language used by the student (Arabic or English).

Use the supplied project context carefully.
Never invent legal requirements, eligibility thresholds, fees, or timelines.
If the context does not contain the answer, say so clearly and provide
the official reference link when relevant.

You may explain general concepts and ask follow-up questions.
You may provide preliminary guidance but never make official eligibility
decisions.

Keep answers clear, organized, and student-friendly.
"""

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎓 Salam AI")
    st.caption("Student Golden Visa Assistant")
    st.divider()

    st.markdown("### About")
    st.write(
        "An AI-powered assistant designed to help students "
        "understand UAE Golden Visa information."
    )

    st.markdown("### Official reference")
    st.link_button(
        "GDRFA Dubai",
        "https://www.gdrfad.gov.ae/"
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Salam AI MVP | Educational prototype")

# ---------------- CHAT DISPLAY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        if not api_key:
            answer = (
                "The AI service is not configured yet. "
                "Please add the Gemini API key in the application settings."
            )
            st.warning(answer)

        else:
            try:
                client = genai.Client(api_key=api_key)

                conversation = []
                for item in st.session_state.messages[-10:]:
                    conversation.append(
                        f'{item["role"].upper()}: {item["content"]}'
                    )

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=(
                        SYSTEM_INSTRUCTION
                        + "\n\nPROJECT KNOWLEDGE:\n"
                        + KNOWLEDGE
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
                    "I could not generate an answer. Please try again."
                )
                st.markdown(answer)

            except Exception as error:
                answer = (
                    "Sorry, the AI service encountered an error. "
                    "Please try again. If the problem continues, "
                    "check the API configuration."
                )
                st.error(answer)
                st.caption(f"Technical details: {error}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })