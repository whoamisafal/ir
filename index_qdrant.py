# from db import docs_col
# from langchain_core.documents import Document
# from rag_langchain import RAGSystem

# # Initialize RAG system
# rag = RAGSystem(gemini_api_key='AIzaSyDgHf4Xe7vKRJB6hQfbj0C1KyP_uCQGkFs')

# batch_size = 500
# documents = []

# # Fetch documents in batches
# cursor = docs_col.find()
# batch_docs = []
# count = 0

# for doc in cursor:
#     title = doc.get("title", "")
#     description = doc.get("description", "")
#     keywords = " ".join(doc.get("keywords", []))
#     visible_text = doc.get("visible_text", "")
#     full_text = f"{title} {description} {keywords} {visible_text}"

#     batch_docs.append(Document(page_content=full_text, meta={
#         'url': doc.get("url", "")
#     }))
#     count += 1

#     # Once batch reaches batch_size, process it
#     if count % batch_size == 0:
#         # Split into chunks
#         chunks = rag.split_documents(batch_docs, chunk_size=200, chunk_overlap=50)
#         # Insert into vector store
#         rag.add_to_vector_store(chunks)
#         # Clear batch_docs for next batch
#         batch_docs = []

# # Process remaining documents if any
# if batch_docs:
#     chunks = rag.split_documents(batch_docs, chunk_size=200, chunk_overlap=50)
#     rag.add_to_vector_store(chunks)

# print("All documents inserted in batches of 500!")


from concurrent.futures import ThreadPoolExecutor, as_completed
from db import docs_col
from langchain_core.documents import Document
from rag_langchain import RAGSystem
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------
# Config
# -----------------------
rag = RAGSystem(gemini_api_key='AIzaSyDgHf4Xe7vKRJB6hQfbj0C1KyP_uCQGkFs')
BATCH_SIZE = 500
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
WORKERS = 4

# -----------------------
# Split documents into chunks
# -----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def split_docs(documents):
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    return chunks

# -----------------------
# Process a batch
# -----------------------
def process_batch(batch_docs):
    chunks = split_docs(batch_docs)
    rag.add_documents(chunks,chunk_size=500)
    return len(batch_docs)

# -----------------------
# Fetch from DB & process
# -----------------------
cursor = docs_col.find({}, {"title":1, "description":1, "keywords":1, "visible_text":1, "url":1})
batch_docs = []
futures = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    count = 0
    for doc in cursor:
        title = doc.get("title", "")
        description = doc.get("description", "")
        keywords = " ".join(doc.get("keywords", []))
        visible_text = doc.get("visible_text", "")
        full_text = f"{title} {description} {keywords} {visible_text}"

        batch_docs.append(Document(page_content=full_text, metadata={"url": doc.get("url", "")}))
        count += 1

        if len(batch_docs) == BATCH_SIZE:
            futures.append(executor.submit(process_batch, batch_docs.copy()))
            batch_docs = []

    if batch_docs:
        futures.append(executor.submit(process_batch, batch_docs.copy()))

    total_processed = 0
    for future in as_completed(futures):
        total_processed += future.result()

print(f" All documents processed! Total documents: {total_processed}")

