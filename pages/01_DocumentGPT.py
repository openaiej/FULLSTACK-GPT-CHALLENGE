from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import os
import streamlit as st
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# 파일을 임베딩하여 검색 가능한 벡터 저장소 생성
#@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 저장소 및 텍스트 분할 설정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

import openai

def verify_openai_key(api_key):
    """
    OpenAI API 키가 유효한지 확인하는 함수 (최신 버전 대응).
    """
    try:
        openai.api_key = api_key  # ✅ API 키 설정
        openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Ping"}]
        )
        return True  # 키가 유효하면 True 반환
    except openai.AuthenticationError:
        return False  # 잘못된 키
    except Exception as e:
        st.error(f"⚠️ API 검증 중 오류 발생: {str(e)}")
        return False
    
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", placeholder="Enter your OpenAI API Key", type="password")
    
    # API 키가 입력되지 않으면 파일 업로드를 막음
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API Key to enable file upload.")
        file = None  # 파일 업로드 비활성화
    else:
        file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

    # 📌 GitHub Repository 링크 추가
    st.markdown(
        """
        ---
        ### 🔗 Project Repository
        [📂 GitHub Repository](https://github.com/openaiej/FULLSTACK-GPT-CHALLENGE.git)
        """
    )
# OpenAI API 키 설정
if openai_api_key:
# API 키 검증
    if verify_openai_key(openai_api_key):
        st.session_state["openai_api_key"] = openai_api_key
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            api_key=st.session_state["openai_api_key"]
        )
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        file = st.file_uploader("📂 Upload a file", type=["pdf", "txt", "docx"])
    else:
        st.error("❌ Invalid OpenAI API Key! Please enter a valid key.")
        file = None



    
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    
    if message:
        if openai_api_key is None or openai_api_key.strip() == "":
            st.error("⚠️ OpenAI API 키를 입력하세요.")
        else:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                chain.invoke(message)


else:
    st.session_state["messages"] = []

