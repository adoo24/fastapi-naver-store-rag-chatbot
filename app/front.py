import streamlit as st
import requests

# Streamlit 앱 제목
st.set_page_config(page_title="Naver Store Chatbot", layout="centered")
st.title("💬 Naver Store Chatbot")

# SSE 응답을 처리하는 제너레이터
def sse_stream(question: str):
    """
    SSE(Server-Sent Events) 응답을 처리하는 제너레이터.
    """
    try:
        with requests.get(f"http://localhost:8000/chat?session_id=test_session&question={question}", stream=True) as response:
            for line in response.iter_lines():
                if line:
                    message = line.decode("utf-8").replace("data: ", "")
                    if message == "[END]":
                        break
                    yield message  # 실시간으로 메시지를 스트리밍
    except Exception as e:
        yield f"Error: {e}"

# 사용자 입력 처리
if prompt := st.chat_input("Ask me anything about Smart Store..."):
    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(prompt)

    # 봇의 응답을 실시간으로 출력
    with st.chat_message("bot"):
        for message in sse_stream(prompt):
            # 개행 문자를 HTML <br> 태그로 변환
            formatted_message = message.replace("\n", "<br>")
            st.markdown(formatted_message, unsafe_allow_html=True)