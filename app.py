# app.py
from flask import Flask, request, render_template,Response,stream_with_context
import os
from search import hybrid_search
from rag_langchain import RAGSystem
from dotenv import load_dotenv

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
cerebras_key = os.getenv("CEREBRAS_API_KEY")
# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)


# def load_indexes():
#     with open(compressed_index_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     doc_id_map = data["doc_id_map"]
#     compressed_index = data["index"]
#     id_to_doc = {v: k for k, v in doc_id_map.items()}

#     with open(tfidf_file, "r", encoding="utf-8") as f:
#         tfidf_index = json.load(f)

#     return compressed_index, tfidf_index, id_to_doc


# -----------------------------
# Search function
# -----------------------------
# def search(query, top_k=5):
#     result = hybrid_search(query=query,top_k=top_k)
#     return result

def search(query, top_k=5, method="bm25"):
    """
    Perform search using either keyword (BM25) or vector (semantic) search.
    
    Parameters:
        query (str): user query
        top_k (int): number of final results to return
        method (str): 'bm25' for keyword search, 'vector' for semantic search
    
    Returns:
        List of dicts with keys: url, title/description/page_content, score
    """
    if not query:
        return []

    unique_results = []

    # ----------------------
    # 1️⃣ Keyword search (BM25)
    # ----------------------
    if method.lower() == "bm25":
        unique_results = hybrid_search(query=query,top_k=top_k)

    # ----------------------
    # 2️⃣ Vector search (semantic)
    # ----------------------
    elif method.lower() == "vector":
        unique_results = rag.qdrant_query_unique(query, fetch_k=top_k*3) 
    else:
        raise ValueError("Invalid search method. Choose 'bm25' or 'vector'.")

    return unique_results


# -----------------------------
# Home / Search page
# -----------------------------
rag = RAGSystem(gemini_api_key=gemini_key)


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "")
        method = request.form.get("method",'bm25')
        top_k = int(request.form.get('top_k', 10))
        results = search(query,top_k=top_k,method=method)
    return render_template("index.html", query=query, results=results)

@app.route("/stream", methods=["POST"])
def stream_answer():
    data = request.get_json()
    query = data.get("query")
    
    try:
        def generate():
            rag_chain = rag.create_rag_chain()
            for chunk in rag_chain.stream({"input": query}):
                # Only yield the text content from the 'answer' key
                if "answer" in chunk:
                    yield chunk["answer"]
    except:
        print("Error")
        return ""
    
    return Response(stream_with_context(generate()), mimetype='text/plain')


# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False)
