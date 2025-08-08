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
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.chat_history = []
        self.feedback_data = []
        self.query_analytics = {}
        
    def initialize(self):
        """시스템 초기화"""
        try:
            file_path = "./all_pages.md"
            if not os.path.exists(file_path):
                st.error(f"❌ {file_path} 파일이 존재하지 않습니다.")
                return False
            
            # 문서 로드 및 벡터스토어 생성
            docs = self.load_and_split_documents(file_path)
            self.vectorstore = self.create_or_load_vectorstore(docs)
            self.retriever = self.create_advanced_retriever()
            self.rag_chain = self.create_rag_chain()
            
            # 분석 데이터 로드
            self.load_analytics_data()
            
            return True
        except Exception as e:
            st.error(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    @st.cache_data
    def load_and_split_documents(_self, file_path: str) -> List[Document]:
        """문서 로드 및 분할"""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        
        # 향상된 텍스트 분할기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # 메타데이터 향상
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'source_file': file_path,
                'processed_at': datetime.now().isoformat(),
                'chunk_length': len(doc.page_content)
            })
        
        return split_docs
    
    @st.cache_resource
    def create_or_load_vectorstore(_self, docs: List[Document]):
        """벡터스토어 생성 또는 로드"""
        persist_directory = "/tmp/chroma_db"
        
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            dimensions=1536
        )
        
        if os.path.exists(persist_directory):
            try:
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings
                )
                # 기존 데이터 확인
                if vectorstore._collection.count() > 0:
                    return vectorstore
            except Exception as e:
                st.warning(f"기존 벡터스토어 로드 실패: {e}")
        
        # 새로운 벡터스토어 생성
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vectorstore
    
    def create_advanced_retriever(self):
        """고급 검색기 생성"""
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )
        return retriever
    
    def create_rag_chain(self):
        """향상된 RAG 체인 생성"""
        qa_system_prompt = """
        당신은 호남권 수질TMS 관제센터의 전문 AI 어시스턴트입니다.
        
        다음 지침을 따라 답변해주세요:
        1. 제공된 컨텍스트를 기반으로 정확하고 상세한 답변을 제공하세요.
        2. 답변을 알 수 없는 경우, 솔직히 모른다고 말하세요.
        3. 한국어로 정중하고 전문적인 언어를 사용하세요.
        4. 관련 법령이나 규정이 있다면 언급해주세요.
        5. 이모지를 적절히 사용해서 친근하게 답변하세요.
        6. 답변의 신뢰도가 낮다면 추가 확인을 권장하세요.
        
        컨텍스트: {context}
        
        채팅 기록: {chat_history}
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0.1,
            max_tokens=1000
        )
        
        def enhanced_format_docs(docs):
            """향상된 문서 포맷팅"""
            if not docs:
                return "관련 정보를 찾을 수 없습니다."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get('source_file', '알 수 없음')
                formatted.append(f"[참고자료 {i}]\n{content}\n")
            
            return "\n".join(formatted)
        
        def add_chat_history(inputs):
            """채팅 기록 추가"""
            inputs["chat_history"] = self.format_chat_history()
            return inputs
        
        rag_chain = (
            {"context": self.retriever | enhanced_format_docs, "input": RunnablePassthrough()}
            | RunnablePassthrough.assign(chat_history=lambda x: self.format_chat_history())
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def format_chat_history(self, limit: int = 3) -> str:
        """최근 채팅 기록 포맷팅"""
        if not self.chat_history:
            return "이전 대화 없음"
        
        recent_history = self.chat_history[-limit:]
        formatted = []
        
        for exchange in recent_history:
            if 'user' in exchange and 'assistant' in exchange:
                formatted.append(f"사용자: {exchange['user']}")
                formatted.append(f"어시스턴트: {exchange['assistant'][:100]}...")
        
        return "\n".join(formatted)
    
    def query_with_analytics(self, query: str) -> Dict[str, Any]:
        """분석 기능이 포함된 쿼리 처리"""
        start_time = time.time()
        
        # 쿼리 전처리
        processed_query = self.preprocess_query(query)
        
        # 유사 질문 검색
        similar_queries = self.find_similar_queries(query)
        
        # RAG 실행
        try:
            response = self.rag_chain.invoke(processed_query)
            
            # 응답 후처리
            enhanced_response = self.postprocess_response(response, query)
            
            # 분석 데이터 업데이트
            processing_time = time.time() - start_time
            self.update_analytics(query, processing_time, True)
            
            return {
                'response': enhanced_response,
                'similar_queries': similar_queries,
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            self.update_analytics(query, time.time() - start_time, False)
            return {
                'response': f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}",
                'similar_queries': [],
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 불필요한 문자 제거
        query = re.sub(r'[^\w\s가-힣]', ' ', query)
        
        # 키워드 확장
        keyword_expansions = {
            'TMS': '원격감시체계 TMS 텔레메트리',
            '수질': '수질 오염 측정 모니터링',
            '관제': '관제 모니터링 감시 제어',
            'COD': 'COD 화학적산소요구량',
            'BOD': 'BOD 생화학적산소요구량'
        }
        
        for keyword, expansion in keyword_expansions.items():
            if keyword in query:
                query = query.replace(keyword, expansion)
        
        return query.strip()
    
    def postprocess_response(self, response: str, original_query: str) -> str:
        """응답 후처리"""
        # 신뢰도 평가
        confidence_keywords = ['확실', '명확', '정확', '규정', '법령']
        uncertainty_keywords = ['아마', '추정', '가능성', '알 수 없']
        
        confidence_score = sum(1 for keyword in confidence_keywords if keyword in response)
        uncertainty_score = sum(1 for keyword in uncertainty_keywords if keyword in response)
        
        # 신뢰도가 낮은 경우 경고 추가
        if uncertainty_score > confidence_score:
            response += "\n\n⚠️ **참고**: 이 답변의 정확성을 위해 관련 법령이나 공식 문서를 확인해보시기 바랍니다."
        
        # 관련 링크나 추가 정보 제안
        if any(keyword in original_query.lower() for keyword in ['법령', '규정', '기준']):
            response += "\n\n📋 **추가 정보**: 정확한 법적 기준은 환경부 홈페이지나 관련 법령을 직접 확인하시기 바랍니다."
        
        return response
    
    def find_similar_queries(self, query: str, limit: int = 3) -> List[str]:
        """유사한 이전 질문 찾기"""
        if not hasattr(self, 'query_analytics') or not self.query_analytics:
            return []
        
        # 간단한 유사도 계산 (실제로는 임베딩 기반으로 개선 가능)
        query_words = set(query.lower().split())
        similar = []
        
        for past_query, data in self.query_analytics.items():
            past_words = set(past_query.lower().split())
            similarity = len(query_words & past_words) / len(query_words | past_words)
            
            if similarity > SIMILARITY_THRESHOLD:
                similar.append((past_query, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in similar[:limit]]
    
    def update_analytics(self, query: str, processing_time: float, success: bool):
        """분석 데이터 업데이트"""
        if query not in self.query_analytics:
            self.query_analytics[query] = {
                'count': 0,
                'avg_time': 0,
                'success_rate': 0,
                'last_asked': None
            }
        
        data = self.query_analytics[query]
        data['count'] += 1
        data['avg_time'] = (data['avg_time'] * (data['count'] - 1) + processing_time) / data['count']
        data['success_rate'] = (data['success_rate'] * (data['count'] - 1) + (1 if success else 0)) / data['count']
        data['last_asked'] = datetime.now().isoformat()
    
    def add_to_chat_history(self, user_msg: str, assistant_msg: str):
        """채팅 기록 추가"""
        self.chat_history.append({
            'user': user_msg,
            'assistant': assistant_msg,
            'timestamp': datetime.now().isoformat()
        })
        
        # 기록 길이 제한
        if len(self.chat_history) > MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-MAX_CHAT_HISTORY:]
    
    def save_feedback(self, query: str, response: str, rating: int, comment: str = ""):
        """피드백 저장"""
        feedback = {
            'query': query,
            'response': response,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_data.append(feedback)
        
        # 임시 저장 (실제로는 데이터베이스나 파일에 저장)
        try:
            with open('/tmp/feedback.json', 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"피드백 저장 실패: {e}")
    
    def load_analytics_data(self):
        """분석 데이터 로드"""
        try:
            if os.path.exists('/tmp/analytics.json'):
                with open('/tmp/analytics.json', 'r', encoding='utf-8') as f:
                    self.query_analytics = json.load(f)
        except Exception:
            self.query_analytics = {}
    
    def save_analytics_data(self):
        """분석 데이터 저장"""
        try:
            with open('/tmp/analytics.json', 'w', encoding='utf-8') as f:
                json.dump(self.query_analytics, f, ensure_ascii=False, indent=2)
        except Exception as e:
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
