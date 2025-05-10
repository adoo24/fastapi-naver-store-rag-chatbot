import streamlit as st
import redis
import json
import plotly.express as px
import pandas as pd

# Redis 클라이언트 설정 (localhost 기준)
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

st.title("FAQ 키워드 및 질문 보강 현황")
st.subheader("Redis 기반 실시간 Bubble Chart")

# Redis에서 키워드 데이터 로드
keywords_data = redis_client.hgetall("keywords")
insufficient_data = redis_client.hgetall("insufficient_context_questions")

# 키워드 데이터 처리
keyword_list = list(keywords_data.keys())
keyword_count = list(map(int, keywords_data.values()))

# 키워드 데이터프레임 생성
df_keywords = pd.DataFrame({
    "Keyword": keyword_list,
    "Count": keyword_count
})

# Bubble Chart 생성
fig = px.scatter(
    df_keywords, 
    x="Keyword", 
    y="Count", 
    size="Count", 
    color="Keyword", 
    size_max=100,
    title="FAQ 키워드 빈도 (Bubble Chart)"
)
st.plotly_chart(fig)

# 보강 필요한 질문들 출력
st.subheader("보강 필요한 질문들")
if insufficient_data:
    for question, vector in insufficient_data.items():
        st.write(f"- {question}")
else:
    st.write("보강이 필요한 질문이 없습니다.")
