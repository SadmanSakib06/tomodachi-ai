import os
import time

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
MODEL = "gemini-3.5-flash-lite"  # Fast, low-thinking model for everyday chat

st.set_page_config(page_title="Tomodachi AI", page_icon="🌸", layout="centered")
st.title("🌸 Tomodachi AI")
st.caption("Your bilingual Japanese–English small-talk companion")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY was not found. Check the .env file.")
    st.stop()

language = st.selectbox("Language / 言語", ["English", "日本語"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

placeholder = "Type your message..." if language == "English" else "メッセージをかいてください..."

if prompt := st.chat_input(placeholder):
    with st.chat_message("user"):
        st.write(prompt)

    # Only send completed exchanges and the new prompt. A failed API call
    # will not leave a stray user turn in the conversation history.
    turns = st.session_state.messages[-12:] + [
        {"role": "user", "content": prompt}
    ]
    contents = [
        types.Content(
            role="model" if turn["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=turn["content"])],
        )
        for turn in turns
    ]

    system_instruction = (
        "You are Tomodachi AI, a friendly bilingual small-talk companion. "
        f"Reply in {'Japanese' if language == '日本語' else 'English'}. "
        "Talk naturally, remember details from this conversation, and keep "
        "replies usually to 1–3 sentences. Ask an occasional relevant "
        "follow-up question, but not in every reply. In Japanese, prefer "
        "simple, natural everyday language."
    )

    with st.chat_message("assistant"):
        start = time.perf_counter()
        with st.spinner("Tomodachi is thinking..."):
            try:
                # 20-second network timeout, no repeated retries while debugging.
                with genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(
                        timeout=20000,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                ) as client:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            max_output_tokens=512,
                        ),
                    )
                reply = (response.text or "").strip()
            except errors.APIError as exc:
                st.error(f"Gemini API error (HTTP {exc.code}): {exc.message}")
                reply = None
            except Exception as exc:
                st.error(f"Connection/code error ({type(exc).__name__}): {exc}")
                reply = None

        if reply:
            st.write(reply)
            st.caption(f"Response time: {time.perf_counter() - start:.1f} seconds")
            st.session_state.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ])
        elif reply == "":
            st.warning("The model returned no visible text. Try again or check model availability.")
