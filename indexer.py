


import json
import re
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rag_langchain import RAGSystem
from langchain_core.documents import Document

# -----------------------
# MongoDB Setup
# -----------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["ir_database"]
docs_col = db["documents"]
index_col = db["inverted_index"]

# -----------------------
# NLTK Setup
# -----------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]

# -----------------------
# Load JSONL and Insert/Update
# -----------------------
INPUT_FILE = "search_index.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents= []

    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        url = record.get("url")
        if not url:
            continue

        # Combine all text fields
        title = record.get("title", "")
        description = record.get("description", "")
        keywords = " ".join(record.get("keywords", []))
        visible_text = record.get("visible_text", "")
        full_text = f"{title} {description} {keywords} {visible_text}"

        

        # Preprocess tokens
        tokens = preprocess_text(full_text)

        # Check if document exists
        existing_doc = docs_col.find_one({"url": url})

        if existing_doc:
            # Document exists, check content hash
            if existing_doc.get("content_hash") == record.get("content_hash"):
                # No changes, skip
                continue
            else:
                # Update document
                doc_id = existing_doc["_id"]

                documents.append(Document(page_content=full_text,meta={
                    'url':record.get("url","")
                }))

                # Remove old tokens from inverted index
                old_tokens = existing_doc.get("tokens", [])
                for token in set(old_tokens):
                    index_col.update_one(
                        {"token": token},
                        {"$pull": {"postings": doc_id}}
                    )

                # Update document fields
                docs_col.update_one(
                    {"_id": doc_id},
                    {"$set": {
                        "title": title,
                        "description": description,
                        "keywords": record.get("keywords", []),
                        "visible_text": visible_text,
                        "image_urls": record.get("image_urls", []),
                        "content_hash": record.get("content_hash"),
                        "crawled_at": record.get("crawled_at"),
                        "depth": record.get("depth"),
                        "internal_links": record.get("internal_links",[]),
                        "tokens":tokens
                    }}
                )

        else:
            # Insert new document
            doc_id = docs_col.estimated_document_count() + 1
            documents.append(Document(page_content=full_text,meta={
                    'url':record.get("url","")
            }))
            docs_col.insert_one({
                "_id": doc_id,
                "url": url,
                "title": title,
                "description": description,
                "keywords": record.get("keywords", []),
                "visible_text": visible_text,
                "image_urls": record.get("image_urls", []),
                "content_hash": record.get("content_hash"),
                "crawled_at": record.get("crawled_at"),
                "depth": record.get("depth"),
                "internal_links": record.get("internal_links",[]),
                  "tokens":tokens
            })

        # Update inverted index
        for token in set(tokens):
            index_col.update_one(
                {"token": token},
                {"$addToSet": {"postings": doc_id}},
                upsert=True
            )

    # Insert the data 
    rag = RAGSystem(gemini_api_key='AIzaSyDgHf4Xe7vKRJB6hQfbj0C1KyP_uCQGkFs')
    chunks = rag.split_documents(documents, chunk_size=200, chunk_overlap=50)
    rag.create_vector_store(chunks)   

print("✅ Documents updated/added and inverted index updated!")
print("📚 Total documents:", docs_col.count_documents({}))
print("📚 Total indexed terms:", index_col.count_documents({}))
