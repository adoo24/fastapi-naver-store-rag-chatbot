import streamlit as st
import requests

# Streamlit 앱 제목
st.set_page_config(page_title="Real-time SSE Chatbot", layout="centered")
st.title("💬 Real-time SSE Chatbot")

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# SSE 응답을 받는 함수
def fetch_sse(question: str):
    """
    SSE(Server-Sent Events) 응답을 실시간으로 받아오는 함수.
    """
    try:
        with requests.get(f"http://localhost:8000/chat?session_id=test_session&question={question}", stream=True) as response:
            bot_message = ""  # 봇의 응답을 누적 저장
            for line in response.iter_lines():
                if line:
                    message = line.decode("utf-8").replace("data: ", "").strip()
                    if message == "[END]":
                        break
                    bot_message += message + "\n"  # 메시지를 누적
            # 최종 메시지를 세션 상태에 추가
            st.session_state.messages.append({"role": "bot", "content": bot_message.strip()})
    except Exception as e:
        # 에러 메시지를 세션 상태에 추가
        st.session_state.messages.append({"role": "error", "content": f"Error fetching SSE: {e}"})

# 사용자 입력 처리
if prompt := st.chat_input("Ask me anything about Smart Store..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    # SSE 요청 처리
    fetch_sse(prompt)

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 개행을 HTML의 <br><br> 태그로 변환하여 표시 (여백 추가)
        formatted_message = message["content"].replace("\n", "<br><br>")
        st.markdown(formatted_message, unsafe_allow_html=True)