import datetime
import os
from pathlib import Path

import pdfplumber
import requests
import streamlit as st


# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="Aryan CHATBOT", layout="centered")


def load_env_file():
    env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_groq_api_key():
    load_env_file()

    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return ""

    return str(api_key).strip().strip('"').strip("'")


def get_setting(name: str, default: str):
    try:
        value = st.secrets.get(name) or os.getenv(name) or default
    except Exception:
        value = os.getenv(name) or default

    return str(value).strip().strip('"').strip("'")


def is_valid_groq_api_key(api_key: str):
    return bool(api_key and api_key.startswith("gsk_") and len(api_key) > 20)


GROQ_API_KEY = get_groq_api_key()
PDF_PATH = Path(__file__).parent / "Data" / "comapanypolicys.pdf"
MODEL_NAME = get_setting("GROQ_MODEL", "llama-3.1-8b-instant")

if not is_valid_groq_api_key(GROQ_API_KEY):
    st.error(
        "GROQ_API_KEY is missing or invalid. Add a fresh key in Streamlit Cloud "
        "Secrets as: GROQ_API_KEY = \"gsk_...\""
    )
    st.stop()

if not PDF_PATH.exists():
    st.error(f"PDF file not found: {PDF_PATH}")
    st.stop()


# -------------------- LOAD + CHUNK PDF --------------------
@st.cache_data
def load_chunks(max_chars: int = 600):
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            tx = page.extract_text()
            if tx:
                text += tx + "\n"

    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""

    for part in raw_parts:
        if len(buf) + len(part) <= max_chars:
            buf += " " + part
        else:
            chunks.append(buf.strip())
            buf = part

    if buf:
        chunks.append(buf.strip())

    return chunks


pdf_chunks = load_chunks()


# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 3):
    q_words = set(query.lower().split())
    scored = []

    for ch in pdf_chunks:
        ch_words = set(ch.lower().split())
        score = len(q_words & ch_words)
        if score > 0:
            scored.append((score, ch))

    if not scored:
        return ""

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([c for _, c in scored[:top_k]])


# -------------------- GROQ API CALL --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 401:
            return (
                "Groq API key is unauthorized. Create a new Groq key, add it in "
                "Streamlit Cloud Secrets, then restart the app."
            )
        if response.status_code == 400:
            return (
                f"Groq rejected the request. Check that GROQ_MODEL is available "
                f"for your account. Current model: {MODEL_NAME}"
            )
        if response.status_code == 429:
            return "Groq rate limit reached. Please wait a minute and try again."
        if response.status_code >= 500:
            return "Groq service is temporarily unavailable. Please try again shortly."
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "Groq request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "Could not connect to Groq. Please check the app network and try again."
    except Exception as exc:
        return f"Groq API Error: {exc}"


# -------------------- RAG + UPDATED INFO --------------------
def is_creator_question(question: str):
    normalized = question.lower().strip()
    creator_phrases = [
        "who created you",
        "who made you",
        "who built you",
        "who developed you",
        "kisne banaya",
        "kon banaya",
        "creator",
    ]
    return any(phrase in normalized for phrase in creator_phrases)


def get_answer(question: str, history):
    if is_creator_question(question):
        return "Aryan kumar created me."

    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y (%Y)")
    pdf_strength = len(context.strip())

    if pdf_strength < 50:
        # PDF does not contain relevant information, so use the model's general knowledge.
        system_prompt = f"""
You are Aryan ka Chatbot.

Rules:
- Give clear and direct answers.
- If someone asks who created you, say Aryan kumar created you.
- Use your updated general knowledge (today = {today}).
- Do NOT say anything about "searching", "checking", "researching", or "not knowing".
- Never restrict information to the year 2023.
"""
    else:
        # PDF has useful context, so use it first while allowing current general info.
        system_prompt = f"""
You are Aryan ka Chatbot.

Use the following PDF text as your main reference.
If updated information (today = {today}) is needed, include it naturally.

PDF Context:
---------------------
{context}
---------------------

Rules:
- Provide confident and direct answers.
- If someone asks who created you, say Aryan kumar created you.
- Do NOT say "I am searching" or "I am researching".
- Never limit your knowledge to only 2023.
"""

    messages = [{"role": "system", "content": system_prompt}]

    for m in history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": question})

    return llama_chat(messages)


# -------------------- STREAMLIT UI --------------------
st.title("Aryan ka Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Assalam o Alaikum! Main Aryan ka Chatbot hoon. "
                "Jo Bhi Phouchna Bindaas Phoucho Mai Ho Na Apki Madad Kay Liye"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Apna sawal likho...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            answer = get_answer(user_input, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
