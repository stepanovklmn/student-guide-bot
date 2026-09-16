import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()

VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"


@st.cache_resource
def load_retrievers():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    raw = vectorstore.get()
    all_docs = [Document(page_content=text) for text in raw["documents"]]
    bm25_retriever = BM25Retriever.from_documents(
        all_docs,
        preprocess_func=lambda text: text.lower().split()
    )
    bm25_retriever.k = 8
    return vector_retriever, bm25_retriever


@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.1,
    )


vector_retriever, bm25_retriever = load_retrievers()
llm = load_llm()


def hybrid_search(question, top_n=8):
    vector_docs = vector_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 0.7 / (60 + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 0.3 / (60 + rank)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: -scores[k])
    return [doc_map[k] for k in sorted_keys[:top_n]]


def rewrite_query(question):
    prompt = f"""Rewrite this student question into 3-5 keywords for search.
Include synonyms in Russian and English. Keep abbreviations like MODF, VMI, SNILS.
Return ONLY keywords separated by spaces, nothing else.

Question: {question}
Keywords:"""
    try:
        return llm.invoke(prompt).content.strip()[:200]
    except Exception:
        return question


def get_answer(question):
    rewritten = rewrite_query(question)
    search_query = question + " " + rewritten

    docs = hybrid_search(search_query, top_n=8)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""CRITICAL RULE: Answer in the SAME language as the question. The question is written in some language — detect it and answer in exactly that language, no matter what language the context is in.

Be concise but helpful. Use ONLY the context.

- If the direct answer IS in the context → answer it in 10-15 sentences, structured with short paragraphs or bullet points.
- If the direct answer is NOT in the context → output 5-8 sentences:
  1. Start with: "I couldn't find a direct answer in the guidebook." (translated to the question's language)
  2. Then give the closest related information from the context in 3-5 sentences.
  3. Optionally end with one sentence advising to contact the International Office.
- If nothing related at all → output 5-8 sentences explaining what topics the guidebook does cover, and suggest contacting the International Office.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (in the same language as the question):"""

    return llm.invoke(prompt).content


st.set_page_config(page_title="Student Guide Bot", page_icon="🎓")
st.title("🎓 Student Guide Chatbot")
st.write("Ask questions / Задайте вопрос")

user_question = st.text_input("Your question / Ваш вопрос:")

if user_question:
    with st.spinner("Looking for an answer..."):
        try:
            answer = get_answer(user_question)
            st.write("### Answer / Ответ:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")