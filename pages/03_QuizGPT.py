import json
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI  # 변경된 import
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.runnables import RunnablePassthrough
import openai

class JsonOutputParser(BaseOutputParser):
    def parse(self, ai_message):
        """AIMessage 객체에서 content 또는 function_call 데이터를 JSON으로 변환"""
        if ai_message.content:
            text = ai_message.content
        elif "function_call" in ai_message.additional_kwargs:
            text = ai_message.additional_kwargs["function_call"]["arguments"]
        else:
            raise ValueError("No valid content or function call arguments found.")

        #print(f"Parsed Text: {text}")  # 디버깅 로그 추가
        return json.loads(text)  # JSON 변환


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")



# 퀴즈 생성 JSON 포맷 정의
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

# OpenAI 모델 정의
llm = None
def format_docs(docs):
    """문서 리스트를 문자열로 변환"""
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    """업로드한 파일을 분할"""
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic,difficulty):
    """퀴즈 생성 체인 실행"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"주어진 글로 퀴즈를 만든다. 난이도 {difficulty}"),
            ("human", "{equation_statement}"),
        ]
    )

    formatted_text = format_docs(_docs)  # 텍스트 변환
    runnable = {"equation_statement": RunnablePassthrough()} | prompt | llm
    result = runnable.invoke({"equation_statement": formatted_text})
    # print(f"run_quiz_chain={result}")
    return output_parser.parse(result)  # JSON으로 변환


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    """위키 검색 결과 반환"""
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.invoke(term)  # get_relevant_documents 대신 invoke 사용
    return docs
def verify_openai_key(api_key):
    """
    OpenAI API 키가 유효한지 확인하는 함수 (최신 버전 대응).
    """
    try:
        client = openai.Client(api_key=api_key)  # ✅ 최신 방식 (openai.Client 사용)
        client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Ping"}]
        )
        return True  # 키가 유효하면 True 반환
    except openai.AuthenticationError:
        return False  # 잘못된 키
    except Exception as e:
        st.error(f"⚠️ API 검증 중 오류 발생: {str(e)}")
        return False

docs = None
topic = None
difficulty="Hard"
# 사이드바 UI
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", placeholder="Enter your OpenAI API Key", type="password")
    if openai_api_key:
        is_valid_key =verify_openai_key(openai_api_key)
        # API 키가 입력되지 않으면 파일 업로드를 막음
        if is_valid_key==False :
            st.error("❌ Invalid OpenAI API Key! Please enter a valid key.")
            file = None  # 파일 업로드 비활성화
            st.session_state["openai_api_key"]=""
        else:
            st.success("✅ Valid OpenAI API Key!")
            st.session_state["openai_api_key"] = openai_api_key
            llm=ChatOpenAI(
                temperature=0.1,
                model_name="gpt-4-turbo",
                api_key=st.session_state["openai_api_key"]
            ).bind(
                function_call={"name": "create_quiz"},
                functions=[
                    function,
                ],
            )
            difficulty = st.selectbox(
                "난이도를 선택하세요",
                ("Hard", "Medium", "Easy"),
                index=0,
                placeholder="Select",
            )
            choice = st.selectbox(
                "Choose what you want to use.",
                ("File", "Wikipedia Article"),
            )

            if choice == "File":
                file = st.file_uploader(
                    "Upload a .docx , .txt or .pdf file",
                    type=["pdf", "txt", "docx"],
                )
                if file:
                    docs = split_file(file)
                    topic = file.name  # 파일명 저장
            else:
                topic = st.text_input("Search Wikipedia...")
                if topic:
                    docs = wiki_search(topic)


# 퀴즈 생성 여부 판단
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic,difficulty)

    # 퀴즈 UI 표시
    if response and "questions" in response:
        with st.form("questions_form"):
            correct_count = 0
            total_questions = len(response["questions"])
            user_answers = []

            for idx, question in enumerate(response["questions"]):
                # st.write(question["question"])
                options = [answer["answer"] for answer in question["answers"]]

                selected_answer = st.radio(
                    question["question"],
                    options,
                    index=None,
                    key=f"question_{idx}",
                )

                if selected_answer:
                    is_correct = any(
                        answer["answer"] == selected_answer and answer["correct"]
                        for answer in question["answers"]
                    )
                    user_answers.append((question["question"], selected_answer, is_correct))

                    if is_correct:
                        st.success("Correct!")
                        correct_count += 1
                    else:
                        st.error("Wrong!")

            # 결과 제출 버튼
            button = st.form_submit_button("Submit Answers")
            if button:
                st.write(f"Your score: {correct_count}/{total_questions}")
                if correct_count==total_questions:
                    st.balloons()
    else:
        st.error("Failed to generate quiz. Please try again.")
