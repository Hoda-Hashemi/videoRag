from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

def create_tfidf_index(chunks):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def retrieve_tfidf(query, vectorizer, tfidf_matrix, chunks, k=5):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()
    top_k_indices = np.argsort(scores)[::-1][:k]
    results = [{'chunk': chunks[idx], 'score': scores[idx]} for idx in top_k_indices if scores[idx] > 0]
    return results

def create_bm25_index(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)
    return bm25

def retrieve_bm25(query, bm25, chunks, k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    results = [{'chunk': chunks[idx], 'score': scores[idx]} for idx in top_k_indices if scores[idx] > 0]
    return results

