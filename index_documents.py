import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import glob

load_dotenv()

PDF_FOLDER = "./guidebooks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DB_DIR = "./chroma_db"

pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"В папке '{PDF_FOLDER}' не найдено ни одного PDF-файла.")

print(f"Найдено PDF-файлов: {len(pdf_files)}")

documents = []
for pdf_file in pdf_files:
    print(f"Загружаю: {pdf_file}")
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

print(f"Всего страниц загружено: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Создано чанков: {len(chunks)}")

print("Создаю эмбеддинги...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Индексирую документы в Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)
vectorstore.persist()
print(f"Готово! Векторная база сохранена в '{VECTOR_DB_DIR}'")