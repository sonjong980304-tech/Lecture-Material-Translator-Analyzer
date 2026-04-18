import streamlit as st
import os
from dotenv import load_dotenv
import time
# LangChain 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_teddynote import logging
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader

# API 키 및 로깅 설정
load_dotenv()
logging.langsmith("Myproject_PDF_rag")

st.set_page_config(page_title="강의자료 번역 챗봇", layout="wide")
st.title("강의자료 번역 챗봇📚✍️")

# --- 초기화 세션 상태 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}

# --- 사이드바 설정 ---
with st.sidebar:
    st.button(
        "대화 초기화",
        on_click=lambda: st.session_state.messages.clear(),
        key="clear_btn",
    )
    uploaded_files = st.file_uploader(
        "파일업로드", type=["PDF"], accept_multiple_files=True
    )
    selected_model = st.selectbox(
        "llm선택", ["gpt-4o-mini", "gpt-4o", "gpt-5.4","gpt-5.4-mini"], index=3
    )
    session_id = st.text_input("세션 ID", "abc123")
    st.caption("made by sonjong")

# --- 페이지 스타일 설정 (베이지 톤) ---
# --- 페이지 스타일 설정 (베이지 톤 꽉 채우기) ---
st.markdown(
    """
    <style>
    /* 1. 전체 앱 배경 (위아래 하얀 여백 제거) */
    .stApp {
        background-color: #F9F7F2; /* 더 고급스러운 웜 베이지 */
    }

    /* 2. 사이드바 배경 */
    [data-testid="stSidebar"] {
        background-color: #F2E8D5; /* 사이드바는 살짝 더 짙게 */
    }

    /* 3. 메인 본문 영역 여백 제거 및 배경 고정 */
    .main {
        background-color: #F9F7F2;
    }

    /* 4. 상단 헤더 가리기 (하얀 줄 방지) */
    header {
        background-color: rgba(0,0,0,0) !important;
    }

    /* 5. 채팅 입력창 위치 및 배경색 최적화 */
    .stChatInputContainer {
        background-color: #F9F7F2 !important;
        padding-bottom: 20px;
    }

    /* 6. 텍스트 가독성을 위한 진한 브라운 컬러 */
    h1, h2, h3, p, span, li {
        color: #4B3621 !important;
    }

    /* 7. 버튼 및 위젯 테두리 부드럽게 */
    .stButton>button {
        border-radius: 10px;
        border: 1px solid #D2B48C;
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 함수 정의 영역 ---


def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


@st.cache_resource(show_spinner="PDF 페이지를 분석 중입니다...")
def embed_files(files):
    all_raw_docs = []
    for file in files:
        cache_dir = "./.cache/files"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        file_path = os.path.join(cache_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # 1. 페이지별로 정확히 로드 (쪼개지 마세요!)
        loader = PyPDFLoader(file_path)
        docs = loader.load() # load_and_split()이 아니라 load()를 씁니다.
        
        for i, doc in enumerate(docs):
            doc.metadata["source"] = file.name
            doc.metadata["page"] = i + 1
            all_raw_docs.append(doc)

    # 2. 벡터 DB 생성 (채팅용)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=all_raw_docs, embedding=embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k": 10}), all_raw_docs

def format_docs(docs):
    return "\n\n".join(
        [
            f"--- [Page {d.metadata.get('page', 'Unknown')}] ---\n{d.page_content}"
            for d in docs
        ]
    )


def create_chains(retriever, model_name="gpt-5.4"):
    # 1. 번역용 프롬프트 (기존 유지)
    translation_prompt = load_prompt("prompts/Translation.yaml", encoding="utf-8")

    # 2. 채팅용 프롬프트 (기존 유지)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 대학생의 학습을 돕는 유능한 AI 조교입니다. [Context]를 바탕으로 답하세요.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # itemgetter("question")을 사용하여 입력 딕셔너리에서 질문만 추출합니다.
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    # 세션 기록 래퍼 (기존 유지)
    with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    translation_chain = translation_prompt | llm | StrOutputParser()
    return with_message_history, translation_chain



# --- 실행 영역 ---
# 메시지 출력
for msg in st.session_state["messages"]:
    st.chat_message(msg.role).write(msg.content)
    
if uploaded_files:
    retriever, all_docs = embed_files(uploaded_files)
    chat_chain, trans_chain = create_chains(retriever, model_name=selected_model)
    st.session_state["chat_chain"] = chat_chain
    st.session_state["trans_chain"] = trans_chain
    st.session_state["all_docs"] = all_docs

    # 사이드바 번역 버튼
    with st.sidebar:
        start_btn = st.button("📄 전체 페이지 번역 실행")

    if start_btn:
        if "all_docs" in st.session_state:
            all_docs = st.session_state["all_docs"]
            total_pages = len(all_docs) # 이제 정확히 37이 나올 겁니다.
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, doc in enumerate(all_docs):
                page_num = i + 1
                
                # 현재 처리 중인 실제 페이지 내용 길이 확인
                content_len = len(doc.page_content)
                status_text.text(f"현재 {page_num}/{total_pages} 페이지 번역 중... (내용 길이: {content_len})")
                
                st.markdown(f"### 📑 Page {page_num} 번역 및 해설")
                
                with st.chat_message("assistant"):
                    try:
                        # trans_chain은 리트리버 없이 순수하게 'context'와 'question'만 받습니다.
                        response = st.session_state["trans_chain"].stream({
                            "context": doc.page_content,
                            "question": "이 페이지의 내용을 절대로 요약하지 말고 한국어로 전체 번역해줘."
                        })
                        full_response = st.write_stream(response)
                        
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=f"### Page {page_num}\n{full_response}")
                        )
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        break
                
                time.sleep(1) # API 과부하 방지
                progress_bar.progress(page_num / total_pages)
                st.divider()
            
            st.rerun()

# 채팅 입력창
user_input = st.chat_input("추가로 궁금한 점을 물어보세요!")

if user_input:
    # 1. 사용자 메시지 출력 및 저장
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # 2. 체인 존재 여부 확인 후 실행
    if "chat_chain" in st.session_state:
        chat_chain = st.session_state["chat_chain"]  # 세션에서 체인 가져오기

        with st.chat_message("assistant"):
            # RunnableWithMessageHistory가 인식할 수 있도록 딕셔너리 형태로 전달
            response = chat_chain.stream(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            ai_answer = st.write_stream(response)

        # AI 답변 저장
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=ai_answer)
        )
    else:
        st.error("먼저 PDF 파일을 업로드해 주세요.")
