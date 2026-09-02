import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import streamlit as st

# Загружаем переменные окружения из .env
load_dotenv()

# Настройки
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"

# Создаём эмбеддинги (та же модель, что использовали при индексации)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Загружаем векторную базу данных
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

# Настраиваем клиент DeepSeek
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# Функция для получения ответа
def get_answer(question):
    # Ищем похожие фрагменты
    docs = vectorstore.similarity_search(question, k=8)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Формируем промпт
    prompt = f"""You are a friendly and competent assistant for international students at the university.

Your task is to assist the student using ONLY the information provided in the guidebook sections.
Answer in detail, step by step, and structure your response (using organized paragraphs). For clarity, you can use lists and numbering. No more than 15 sentences.
If you don't find the necessary information for the fifth time, honestly say: "I couldn't find the answer in the materials provided. I recommend contacting the university's International Office."
Don't make up facts.
Answer only in the same language, ONLY and ALWAYS in the same language as the question (if the question is in English, answer in English; if in English, answer in English). If you answer in a language other than the one the question was asked, you will be removed.

Фрагменты гайдбука:
{context}

Вопрос студента: {question}

Твой подробный ответ:"""

    # Отправляем запрос в DeepSeek
    response = llm.invoke(prompt)
    return response.content

# Интерфейс Streamlit
st.set_page_config(page_title="Student Guide Bot", page_icon="🎓")
st.title("🎓 Student Guide Chatbot")
st.write("Ask questions about student life, tuition, visas, etc.")

# Поле для ввода
user_question = st.text_input("Your question:", placeholder="For example: How to extend my visa?")

if user_question:
    with st.spinner("Looking for an answer..."):
        try:
            answer = get_answer(user_question)
            st.write("### Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")