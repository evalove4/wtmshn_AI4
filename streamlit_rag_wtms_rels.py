__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# 환경변수 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 상수 설정
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '50'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

class AdvancedRAGSystem:
            st.warning(f"분석 데이터 저장 실패: {e}")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="호남권 WTMS Q&A 챗봇",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 사이드바
    with st.sidebar:
        st.title("🔧 시스템 설정")
        
        # 모델 설정
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        selected_model = st.selectbox("모델 선택", model_options)
        os.environ['OPENAI_MODEL'] = selected_model
        
        # 검색 설정
        st.subheader("검색 설정")
        search_type = st.selectbox("검색 방식", ["mmr", "similarity"])
        num_results = st.slider("검색 결과 수", 3, 10, 5)
        
        # 시스템 상태
        st.subheader("📊 시스템 상태")
        if 'rag_system' in st.session_state:
            if hasattr(st.session_state.rag_system, 'query_analytics'):
                total_queries = len(st.session_state.rag_system.query_analytics)
                st.metric("총 질문 수", total_queries)
                
                if total_queries > 0:
                    avg_success_rate = sum(data['success_rate'] for data in st.session_state.rag_system.query_analytics.values()) / total_queries
                    st.metric("평균 성공률", f"{avg_success_rate:.1%}")
        
        # 데이터 관리
        st.subheader("🗃️ 데이터 관리")
        if st.button("분석 데이터 저장"):
            if 'rag_system' in st.session_state:
                st.session_state.rag_system.save_analytics_data()
                st.success("저장 완료!")
        
        if st.button("채팅 기록 초기화"):
            if 'messages' in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "수질관제시스템에 대해 무엇이든 물어보세요! 😊"}
                ]
                st.success("채팅 기록이 초기화되었습니다!")
    
    # 메인 화면
    st.title("호남권 WTMS Q&A 챗봇 💬")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    🌊 <strong>수질TMS 전문 AI 어시스턴트</strong><br>
    본 서비스는 한국환경공단 호남권 수질TMS 관제센터에 의해 운영되며, 
    챗봇의 답변에는 오류가 있을 수 있으니 중요한 사항은 반드시 법령 등 출처를 확인해보시기 바랍니다. 📖
    </div>
    """, unsafe_allow_html=True)
    
    # RAG 시스템 초기화
    if 'rag_system' not in st.session_state:
        with st.spinner("🔄 시스템을 초기화하고 있습니다..."):
            st.session_state.rag_system = AdvancedRAGSystem()
            if st.session_state.rag_system.initialize():
                st.success("✅ 시스템이 성공적으로 초기화되었습니다!")
            else:
                st.error("❌ 시스템 초기화에 실패했습니다.")
                st.stop()
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "수질관제시스템에 대해 무엇이든 물어보세요! 😊"}
        ]
    
    if "feedback_mode" not in st.session_state:
        st.session_state.feedback_mode = False
    
    # 채팅 인터페이스
    chat_container = st.container()
    
    with chat_container:
        # 기존 메시지 표시
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg['role']):
                st.write(msg['content'])
                
                # 어시스턴트 메시지에 피드백 버튼 추가
                if msg['role'] == 'assistant' and i > 0:  # 첫 번째 환영 메시지 제외
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
                    
                    with col1:
                        if st.button("👍", key=f"good_{i}"):
                            st.session_state.rag_system.save_feedback(
                                st.session_state.messages[i-1]['content'],
                                msg['content'],
                                5,
                                "좋음"
                            )
                            st.success("피드백 감사합니다!")
                    
                    with col2:
                        if st.button("👎", key=f"bad_{i}"):
                            st.session_state.rag_system.save_feedback(
                                st.session_state.messages[i-1]['content'],
                                msg['content'],
                                1,
                                "나쁨"
                            )
                            st.warning("피드백이 기록되었습니다.")
                    
                    with col3:
                        if st.button("🔄", key=f"retry_{i}"):
                            # 재생성 로직
                            with st.spinner("답변을 다시 생성하고 있습니다..."):
                                result = st.session_state.rag_system.query_with_analytics(
                                    st.session_state.messages[i-1]['content']
                                )
                                st.session_state.messages[i] = {
                                    "role": "assistant", 
                                    "content": result['response']
                                }
                                st.rerun()
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력해주세요 😊"):
        # 사용자 메시지 표시
        with st.chat_message("human"):
            st.write(prompt)
        st.session_state.messages.append({"role": "human", "content": prompt})
        
        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("🤔 답변을 생성하고 있습니다..."):
                result = st.session_state.rag_system.query_with_analytics(prompt)
                
                # 응답 표시
                st.write(result['response'])
                
                # 성능 정보 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"⏱️ 처리시간: {result['processing_time']:.2f}초")
                with col2:
                    if result['success']:
                        st.caption("✅ 성공")
                    else:
                        st.caption("❌ 실패")
                
                # 유사 질문 표시
                if result['similar_queries']:
                    with st.expander("🔍 관련 질문들"):
                        for similar_q in result['similar_queries']:
                            st.write(f"• {similar_q}")
                
                # 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['response']
                })
                
                # 채팅 기록 업데이트
                st.session_state.rag_system.add_to_chat_history(prompt, result['response'])

if __name__ == "__main__":
    main()
