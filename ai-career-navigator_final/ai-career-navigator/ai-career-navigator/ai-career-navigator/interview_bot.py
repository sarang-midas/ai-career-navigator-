
import streamlit as st
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = "You are a helpful, strict mock interviewer. Ask one question at a time, wait for answer, then give brief feedback and the next question."

def run_mock_interview(target_role: str = "Data Analyst"):
    if "interview_history" not in st.session_state:
        st.session_state.interview_history = [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Start a mock interview for the role: {target_role}. Begin with a friendly greeting and the first HR question."}
        ]
    st.write(f"**Target Role:** {target_role}")
    container = st.container()

    # Display history excluding system
    for msg in st.session_state.interview_history:
        if msg["role"] == "assistant":
            container.chat_message("assistant").markdown(msg["content"])
        elif msg["role"] == "user" and msg["content"] != st.session_state.interview_history[1]["content"]:
            container.chat_message("user").markdown(msg["content"])

    user_inp = st.chat_input("Type your answer and press Enter")
    if user_inp:
        st.session_state.interview_history.append({"role":"user","content":user_inp})
        with st.spinner("Evaluating..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.interview_history,
                temperature=0.5
            )
        answer = resp.choices[0].message.content
        st.session_state.interview_history.append({"role":"assistant","content":answer})
        st.rerun()
