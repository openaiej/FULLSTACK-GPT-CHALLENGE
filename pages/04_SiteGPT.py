# 필요한 라이브러리 임포트
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  # 체인 실행을 위한 유틸리티
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 작은 조각으로 나누는 기능
from langchain.prompts import ChatPromptTemplate  # LangChain 프롬프트 템플릿
from langchain_community.document_loaders import SitemapLoader  # 사이트맵에서 데이터를 로드하는 기능
from langchain_community.vectorstores import FAISS  # FAISS 기반 벡터 검색
from langchain_openai import OpenAIEmbeddings  # OpenAI의 임베딩 모델
from langchain_openai import ChatOpenAI  # OpenAI LLM 모델 사용
import streamlit as st  # Streamlit을 활용한 UI 구성
import os
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하는 기능

# .env 파일에서 환경 변수 로드 (API 키 설정)
load_dotenv()
# 환경 변수 설정 (USER_AGENT 추가)
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# 기본 요청 헤더 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# OpenAI의 ChatGPT 모델 초기화 (API 키 필요)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # 환경 변수에서 API 키 로드
    temperature=0.1,  # 낮은 온도로 더 일관된 답변을 생성
)

# 답변을 생성하는 프롬프트 템플릿
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

# 문서를 기반으로 질문에 대한 답변을 생성하는 함수
def get_answers(inputs):
    docs = inputs["docs"]  # 가져온 문서 리스트
    question = inputs["question"]  # 사용자의 질문
    answers_chain = answers_prompt | llm  # LangChain 체인 실행

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,  # LLM을 이용해 문서별 답변 생성
                "source": doc.metadata["source"],  # 문서 출처
                "date": doc.metadata["lastmod"],  # 문서의 마지막 수정 날짜
            }
            for doc in docs
        ],
    }

# 여러 개의 답변 중 가장 적절한 답변을 선택하는 프롬프트
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# 점수가 높은 답변을 선택하는 함수
def choose_answer(inputs):
    answers = inputs["answers"]  # 생성된 답변 리스트
    question = inputs["question"]  # 사용자의 질문
    choose_chain = choose_prompt | llm  # LangChain 체인 실행

    # 답변을 하나의 문자열로 합침 (출처와 날짜 포함)
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

# 웹페이지에서 헤더/푸터 제거 후 텍스트 추출
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()  # 헤더 삭제
    if footer:
        footer.decompose()  # 푸터 삭제
    return (
        str(soup.get_text())  # 텍스트만 추출
        .replace("\n", " ")  # 줄바꿈 제거
        .replace("\xa0", " ")  # 특수문자 제거
        .replace("CloseSearch Submit Blog", "")  # 불필요한 텍스트 제거
    )

# Streamlit에서 데이터를 캐싱하여 성능 최적화 (문제 해결을 위해 캐싱 제거)
@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    print("🔹 load_website() 함수 시작")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_kwargs = {"headers": headers}
    loader.requests_per_second = 2

    print("🔹 SitemapLoader 설정 완료. 크롤링 시작...")
    docs = loader.load()
    print(f"🔹 크롤링 완료! 문서 개수: {len(docs)}")

    # ✅ 샘플링하여 실행 (전체 16,217개 문서는 너무 많음)
    docs = docs[:1000]  # 문서 개수를 줄여서 테스트
    print(f"🔹 샘플링 후 문서 개수: {len(docs)}")

    split_docs = splitter.split_documents(docs)
    print(f"🔹 문서 나누기 완료! 분할된 문서 개수: {len(split_docs)}")

    print("🔹 FAISS 벡터 저장소 생성 시작...")

    # ✅ OpenAIEmbeddings 최신 방식 적용 (batch_size 문제 해결)
    vector_store = FAISS.from_documents(
        split_docs, OpenAIEmbeddings()
    )

    print("🔹 FAISS 벡터 저장소 생성 완료!")

    return vector_store.as_retriever()


# Streamlit 페이지 설정
st.set_page_config(
    page_title="SiteGPT",  # 페이지 제목
    page_icon="🖥️",  # 페이지 아이콘
)

# 페이지 제목 및 설명
st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

# Sidebar에 URL 입력 창 추가
with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

# 사용자가 URL을 입력했을 때 동작
if url:
    if ".xml" not in url:  # 올바른 사이트맵 URL인지 확인
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")  # 오류 메시지 출력
    else:
        retriever = load_website(url)  # 웹사이트 크롤링 및 벡터화
        query = st.text_input("Ask a question to the website.")  # 질문 입력 필드

        if query:  # 사용자가 질문을 입력하면 실행
            chain = (
                {
                    "docs": retriever,  # 벡터 검색을 통해 문서 검색
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)  # 문서 기반 답변 생성
                | RunnableLambda(choose_answer)  # 가장 적절한 답변 선택
            )
            result = chain.invoke(query)  # 체인 실행 후 결과 반환
            st.markdown(result.content.replace("$", "\$"))  # Markdown으로 출력 (특수문자 처리)
