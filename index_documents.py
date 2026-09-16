import os
import re
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

DOCX_FOLDER = "./guidebooks"
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

docx_files = glob.glob(os.path.join(DOCX_FOLDER, "*.docx"))
if not docx_files:
    raise FileNotFoundError("Нет DOCX в guidebooks")

print(f"Найдено DOCX: {len(docx_files)}")

documents = []
for f in docx_files:
    print(f"Загружаю: {f}")
    documents.extend(Docx2txtLoader(f).load())

full_text = "\n".join([doc.page_content for doc in documents])

blocks = [b.strip() for b in re.split(r'\n\s*\n', full_text) if b.strip()]

chunks = []
current = ""

for i, block in enumerate(blocks):
    # если следующий непустой блок начинается с "Вопрос:" — значит текущий это ЗАГОЛОВОК
    next_is_question = (i + 1 < len(blocks) and blocks[i + 1].startswith("Вопрос:"))

    if next_is_question and not block.startswith("Вопрос:"):
        # сохраняем текущий чанк и начинаем новый с заголовка
        if current:
            chunks.append(current.strip())
        current = block
    else:
        # продолжение текущего чанка
        if current:
            current += "\n\n" + block
        else:
            current = block

if current:
    chunks.append(current.strip())
# Если совсем ничего не нашли — оставляем одну секцию
if not chunks:
    chunks = [full_text]

chunks = [Document(page_content=c, metadata={"source": "guidebook"}) for c in chunks]
print(f"Создано чанков: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)
vectorstore.persist()
print("Готово!")