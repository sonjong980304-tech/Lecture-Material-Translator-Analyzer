# Lecture-Material-Translator-Analyzer

기존의 범용 LLM 서비스에 대용량 PDF 강의 자료를 업로드할 경우, 모델의 출력 토큰 제한으로 인해 내용이 임의로 요약되거나 일부 페이지가 누락되는 문제가 발생합니다.

이 프로젝트는 LangChain과 RAG(Retrieval-Augmented Generation) 기술을 활용하여, 영문 강의 자료를 **단 한 페이지의 누락 없이 완벽하게 번역**하고, 학습 내용에 대해 **심도 있는 멀티턴 대화**가 가능한 맞춤형 학습 챗봇을 구현하였습니다.

(현재 Claude AI로도 구현이 가능하지만 토큰 비용이 매우 크므로, GPT API를 활용하는 것이 더욱 효율적입니다.)

추가로, 번역되는 내용마다 핵심 용어 정리와 추가 해설을 달아 페이지마다 내용을 이해하는 데 도움을 줄 수 있도록 하였습니다.

🔗 https://lecture-material-translator-analyzer-pqf6lyjskqleky8qwpuwgo.streamlit.app/

---

## 주요 기능

- **전체 번역**: PDF 전 페이지를 누락·요약 없이 한국어로 완전 번역
- **페이지별 해설**: 핵심 용어 정리 + 개념 이해를 위한 추가 설명 자동 생성
- **RAG 기반 질의응답**: 업로드한 자료를 기반으로 멀티턴 대화 지원
- **GPT-4o 모델 선택**: gpt-4o / gpt-4o-mini 중 선택 가능

---

## 사용된 LangChain 핵심 기능

### 1. LCEL (LangChain Expression Language) — 파이프 연산자 체인

LangChain의 `|` 연산자를 사용해 여러 컴포넌트를 선언적으로 연결합니다. 이 프로젝트에서는 두 개의 독립적인 체인을 구성합니다.

**RAG 체인 (질의응답용):**
```python
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
```
`itemgetter("question")`이 입력 딕셔너리에서 질문 텍스트를 꺼내 retriever(벡터 검색)와 프롬프트로 동시에 흘려보냅니다.

**번역 체인 (전체 번역용):**
```python
translation_chain = translation_prompt | llm | StrOutputParser()
```
히스토리가 없는 단순 체인으로, 페이지 텍스트를 받아 번역 결과를 반환합니다.

---

### 2. RunnableWithMessageHistory — 멀티턴 대화 히스토리 관리

RAG 체인을 `RunnableWithMessageHistory`로 감싸 세션 ID 기반 대화 히스토리를 자동으로 주입·저장합니다.

```python
with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,          # 세션 ID → ChatMessageHistory 반환
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

`ChatMessageHistory`는 Streamlit의 `session_state["store"]`에 세션별로 저장되며, 체인 호출 시 `config={"configurable": {"session_id": ...}}`를 통해 해당 세션의 히스토리가 자동으로 프롬프트에 삽입됩니다.

**번역 체인에는 의도적으로 이 래퍼를 사용하지 않습니다** — 번역은 페이지마다 독립적으로 처리해야 하므로 이전 번역 결과가 다음 호출에 영향을 주지 않도록 stateless로 유지합니다.

---

### 3. RAG (Retrieval-Augmented Generation) — FAISS 벡터 검색

PDF 전체를 한 번에 LLM에 넣는 대신, 각 페이지를 임베딩해 FAISS 벡터 DB에 저장합니다. 질문이 들어오면 의미적으로 가장 유사한 상위 5개 청크만 컨텍스트로 전달합니다.

```python
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=all_raw_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

100페이지 문서라도 실제 LLM에 전달되는 것은 5슬라이드 분량의 텍스트뿐이므로 토큰 효율이 높습니다.

---

### 4. MessagesPlaceholder — 프롬프트 내 히스토리 삽입 위치 선언

`ChatPromptTemplate`에서 `MessagesPlaceholder`를 사용해 대화 히스토리가 들어갈 위치를 명시합니다. `RunnableWithMessageHistory`가 실행 시 이 위치에 이전 대화 내용을 자동으로 채워넣습니다.

```python
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 대학생의 학습을 돕는 유능한 AI 조교입니다. [Context]를 바탕으로 답하세요."),
    MessagesPlaceholder(variable_name="chat_history"),  # 히스토리 자동 삽입
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])
```

---

### 5. load_prompt — YAML 파일로 프롬프트 분리 관리

번역 프롬프트를 코드에 하드코딩하지 않고 `prompts/Translation.yaml`에 외부 파일로 분리해 관리합니다.

```python
translation_prompt = load_prompt("prompts/Translation.yaml", encoding="utf-8")
```

YAML 프롬프트에는 `{context}`(번역할 페이지 내용)와 `{question}`(페이지 번호 안내 지시) 두 변수가 정의되어 있으며, 완전 번역 + 핵심 용어 정리 + 추가 설명의 3단 출력 구조를 강제합니다.

---

### 6. PyMuPDFLoader — 고정밀 PDF 파싱

`PyMuPDFLoader`를 사용해 PPT 기반 PDF에서 텍스트를 페이지 단위로 정확하게 추출합니다. 각 페이지 문서에 `source`(파일명)와 `page`(페이지 번호) 메타데이터를 부착해 추후 검색 결과에서 출처를 추적할 수 있습니다.

---

## 컨텍스트 제한을 해결한 방법

일반 LLM 서비스에서 PDF 전체를 한 번에 보내면 토큰 한도를 초과해 페이지가 생략되거나 요약됩니다. 이 프로젝트는 세 가지 전략으로 이 문제를 해결합니다.

### 전략 1: 페이지 단위 분할 번역

번역 기능은 PDF 전체를 한 번에 보내지 않고, **페이지 하나씩 개별 LLM 호출**로 처리합니다.

```python
for doc in all_raw_docs:
    response = translation_chain.stream({
        "context": doc.page_content,   # 현재 페이지 텍스트만 전달
        "question": f"당신은 현재 {page_num}페이지를 작업 중입니다. 이전 내용은 잊고, 오직 이 [Context]만 한국어로 전체 번역하세요."
    })
```

- 각 LLM 호출에는 **현재 페이지 텍스트만** 컨텍스트로 들어갑니다.
- 번역 체인은 히스토리가 없는 **stateless 체인**이므로, 이전 페이지 번역 결과가 다음 호출의 컨텍스트를 차지하지 않습니다.
- 질문 필드에 "이전 내용은 잊고"를 명시해 LLM이 누적 문맥 없이 해당 페이지에만 집중하도록 유도합니다.
- 결과적으로 200페이지 문서라도 각 호출은 1페이지 분량의 토큰만 소비하므로 토큰 한도에 걸리지 않습니다.

### 전략 2: 중복 페이지 제거

PPT를 PDF로 변환하면 동일한 슬라이드가 연속으로 추출되는 경우가 있습니다. 이전 페이지와 내용이 동일하면 스킵하는 중복 제거 로직으로 불필요한 번역 호출과 토큰 낭비를 방지합니다.

```python
if current_text == last_page_text:
    continue
```

### 전략 3: RAG — 필요한 청크만 선택적 검색 (채팅 모드)

채팅 모드에서는 PDF 전체 대신 질문과 의미적으로 가장 유사한 상위 5개 청크(`k=5`)만 LLM에 전달합니다. 방대한 문서라도 LLM이 실제로 읽는 것은 관련 있는 5페이지 분량뿐입니다.

---

## 프로젝트 구조

```
.
├── 01_PDF.py              # 메인 Streamlit 앱
├── prompts/
│   └── Translation.yaml   # 번역 프롬프트 템플릿 (YAML)
├── requirements.txt       # 의존성 목록
└── pyproject.toml
```

---

## 실행 방법

```bash
pip install -r requirements.txt
```

`.env` 파일에 OpenAI API 키를 설정합니다:
```
OPENAI_API_KEY=your_key_here
```

```bash
streamlit run 01_PDF.py
```

---

## 화면 미리보기

<img width="960" height="446" alt="image" src="https://github.com/user-attachments/assets/e3bee4a8-27d9-4320-8c86-4981c8f42862" />

웹사이트 첫 화면입니다. 왼쪽 바 상단에 PDF 파일을 업로드합니다.

<img width="959" height="446" alt="image" src="https://github.com/user-attachments/assets/0cd2a3ea-1639-4f62-92c7-9a9ce1faef90" />

<img width="956" height="448" alt="image" src="https://github.com/user-attachments/assets/b0b56ada-9448-4eab-94f1-599925349319" />
