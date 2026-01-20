from preprocess import preprocess_text
from rank_bm25 import BM25Okapi
from db import index_col,docs_col
import numpy as np

def hybrid_search(query, top_k=5):
    #  Tokenize query
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    #  Inverted Index → Candidate Docs
    candidate_doc_ids = set()
    for token in query_tokens:
        token_entry = index_col.find_one({"token": token})
        if token_entry:
            candidate_doc_ids.update(token_entry["postings"])
    
    if not candidate_doc_ids:
        return []

    # Fetch candidate docs from MongoDB
    candidates = []
    candidate_tokens_list = []
    for doc_id in candidate_doc_ids:
        doc = docs_col.find_one({"_id": doc_id})
        if doc:
            candidates.append(doc)
            candidate_tokens_list.append(doc["tokens"])
    
    if not candidates:
        return []

    #  BM25 Ranking
    bm25 = BM25Okapi(candidate_tokens_list)
    scores = bm25.get_scores(query_tokens)

    # Sort top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = candidates[idx]
        results.append({
            "doc_id": doc["_id"],
            "title": doc["title"],
            "url": doc["url"],
            "description": doc["description"],
            'visible_text':doc['visible_text'],
            "score": float(scores[idx]),
            "description":doc['description']
        })

    return results

