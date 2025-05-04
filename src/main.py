#%%
print('Loading libraries and functions... ')

from video_processing import extract_frames, generate_image_embeddings
from transcription import transcribe_audio
from embedding import text_embeddings

import json
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

from semantic_retrieval import faiss_retrieve , create_postgres_ivfflat_table, postgres_ivfflat_retrieve, create_postgres_hnsw_table,retrieve_postgres_hnsw

from utils import load_golden_test_set, plot_similarity_scores,load_retrieval_results

from lexical_retrieval import create_bm25_index, retrieve_bm25,retrieve_tfidf,create_tfidf_index

#%%
def main():
    audio_file = "../data/audio.mp3"
    transcription_file = "../content/transcription.txt"
    segments_file = "../content/segments.txt"
    chunks_file = "../content/chunks.txt"
    clean_chunks_file = "../content/clean_chunks.txt"
    video_file = "../data/video.mp4"
    image_folder = "../frames"
    text_embedding_file = "../embeddings/embeddings.pt"
    image_embedding_file = "../embeddings/image_embeddings.pt"
    golden_test_file = "GoldTestSet.json"
    k = 5

    #* output dir:
    faiss_json_file = '../output/faiss_retrieval_results.json'
  
    ivfflat_json_file = "../output/ivfflat_retrieval_results.json"
    
    hnsw_json_file = '../output/hnsw_retrieval_results.json'

    tfidf_json_file = '../output/tfidf_retrieval_results.json'
    bm25_json_file = '../output/bm25_retrieval_results.json'

    #*Chunking and Embedding
    print('Transcribing the audio file ...')
    text_chunks = transcribe_audio(audio_file, transcription_file, segments_file, chunks_file, clean_chunks_file)

    print('Embedding the chunks ...')
    chunk_embeddings = text_embeddings(text_chunks, text_embedding_file)

    extract_frames(video_file, image_folder, interval=5)
    generate_image_embeddings(image_folder, image_embedding_file)
    
    #*Queries 
    golden_queries, query_embeddings = load_golden_test_set(golden_test_file)

    #* Process each golden query
    faiss_retrieval_results = []

    for idx, (query, query_emb) in enumerate(zip(golden_queries, query_embeddings)):
        # Ensure query_emb is shaped (1, dimension) for FAISS
        query_emb = query_emb.reshape(1, -1)
        
        # Retrieve using FAISS
        indices, distances, retrieved_chunks = faiss_retrieve(chunk_embeddings, query_emb, text_chunks, k=k)
        
        # Evaluate based on query type
        answer_segments = []
        reason = query.get('reason', '')
        
        if query['type'] == 'answerable':
            answer_segments = query.get('answer_segments', [])
        
        # Store results
        result = {
            'id': query['id'],
            'question': query['question'],
            'timestamp': query.get('timestamp', 0.0),  # Default to 0.0 if missing
            'type': query['type'],
            'retrieved_indices': indices.tolist(),
            'distances': distances.tolist(),
            'retrieved_chunks': retrieved_chunks,
            'answer_segments': answer_segments,
            'reason': reason if query['type'] == 'unanswerable' else ''
        }
        faiss_retrieval_results.append(result)
        
        # Print results for each query
        print(f"\nQuery {query['id']}: {query['question']} (Timestamp: {query.get('timestamp', 0.0)})")
        if query['type'] == 'unanswerable':
            print(f"Unanswerable: {reason}")
        for i, (idx, dist, chunk) in enumerate(zip(indices, distances, retrieved_chunks)):
            match = " (Match)" if query['type'] == 'answerable' and any(chunk in ans for ans in answer_segments) else ""
            print(f"  Result {i + 1}: Index={idx}, Similarity={dist:.4f}, Text={chunk}{match}")

    # Save results
    with open(faiss_json_file, 'w') as f:
        json.dump(faiss_retrieval_results, f, indent=2)
    print(f"Results saved to {faiss_json_file}")
    #*IVFFLAT
    conn_ivfflat = psycopg2.connect(dbname="my_semantic_db", user="hodahashemi")
    cur_ivfflat = conn_ivfflat.cursor()

    create_postgres_ivfflat_table(conn_ivfflat, cur_ivfflat)

    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        cur_ivfflat.execute("INSERT INTO chunks (chunk, embedding) VALUES (%s, %s)", (chunk, embedding.tolist()))
    conn_ivfflat.commit()

    # Load golden queries
    golden_queries, query_embeddings = load_golden_test_set(golden_test_file)

    # Process each golden query (IVF-Flat retrieval)
    retrieval_results = []
    for idx, (query, query_emb) in enumerate(zip(golden_queries, query_embeddings)):
        query_emb = query_emb.reshape(1, -1)
        indices, distances, retrieved_chunks = postgres_ivfflat_retrieve(conn_ivfflat, cur_ivfflat, query_emb, text_chunks, k=k)
        
        answer_segments = []
        reason = query.get('reason', '')
        if query['type'] == 'answerable':
            answer_segments = query.get('answer_segments', [])
        
        result = {
            'id': query['id'],
            'question': query['question'],
            'timestamp': query.get('timestamp', 0.0),
            'type': query['type'],
            'retrieved_indices': indices.tolist(),
            'distances': distances.tolist(),
            'retrieved_chunks': retrieved_chunks,
            'answer_segments': answer_segments,
            'reason': reason if query['type'] == 'unanswerable' else ''
        }
        retrieval_results.append(result)
        
        print(f"\nQuery {query['id']}: {query['question']} (Timestamp: {query.get('timestamp', 0.0)})")
        if query['type'] == 'unanswerable':
            print(f"Unanswerable: {reason}")
        for i, (idx, dist, chunk) in enumerate(zip(indices, distances, retrieved_chunks)):
            match = " (Match)" if query['type'] == 'answerable' and any(chunk in ans for ans in answer_segments) else ""
            print(f"  Result {i + 1}: Index={idx}, Similarity={dist:.4f}, Text={chunk}{match}")

    # Save retrieval results to JSON
    with open(ivfflat_json_file, 'w') as f:
        json.dump(retrieval_results, f)

    # Clean up
    cur_ivfflat.close()
    conn_ivfflat.close()

    #*HSNW

    # HNSW Retrieval (PostgreSQL)
    print("Performing HNSW retrieval...")
    conn = psycopg2.connect(dbname="my_semantic_db", user="hodahashemi")
    cur = conn.cursor()

    create_postgres_hnsw_table(conn, cur)
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        cur.execute("INSERT INTO chunks (chunk, embedding) VALUES (%s, %s)", (chunk, embedding.tolist()))
    conn.commit()

    hnsw_results = []
    for idx, query in enumerate(golden_queries):
        query_emb = query_embeddings[idx]  # Assumes query_embeddings is aligned with golden_queries
        indices, similarities, retrieved_chunks = retrieve_postgres_hnsw(conn, cur, query_emb, text_chunks, k=5)
        
        answer_segments = []
        reason = query.get('reason', '')
        if query['type'] == 'answerable':
            answer_segments = query.get('answer_segments', [])
        
        result = {
            'id': query['id'],
            'question': query['question'],
            'timestamp': query.get('timestamp', 0.0),
            'type': query['type'],
            'retrieved_indices': indices.tolist(),
            'distances': similarities.tolist(),
            'retrieved_chunks': retrieved_chunks,
            'answer_segments': answer_segments,
            'reason': reason if query['type'] == 'unanswerable' else ''
        }
        hnsw_results.append(result)
        
        print(f"\nQuery {query['id']}: {query['question']} (Timestamp: {query.get('timestamp', 0.0)})")
        if query['type'] == 'unanswerable':
            print(f"Unanswerable: {reason}")
        for i, (idx, sim, chunk) in enumerate(zip(indices, similarities, retrieved_chunks)):
            match = " (Match)" if query['type'] == 'answerable' and any(chunk in ans for ans in answer_segments) else ""
            print(f"  Result {i + 1}: Index={idx}, Similarity={sim:.4f}, Text={chunk}{match}")

    # Save HNSW results
    with open(hnsw_json_file, 'w') as f:
        json.dump(hnsw_results, f, indent=2)
    print(f"HNSW results saved to {hnsw_json_file}")

    # Clean up
    cur.close()
    conn.close()

    #* TF-IDF Retrieval

    print("Performing TF-IDF retrieval...")
    vectorizer, tfidf_matrix = create_tfidf_index(text_chunks)
    tfidf_results = []
    for query in golden_queries:
        query_text = query['question']
        results = retrieve_tfidf(query_text, vectorizer, tfidf_matrix, text_chunks, k=5)
        
        # Ensure k results, padding with zeros if fewer
        indices = [i for i, res in enumerate(results)][:k]
        similarities = [res['score'] for res in results][:k]
        retrieved_chunks = [res['chunk'] for res in results][:k]
        while len(similarities) < k:
            similarities.append(0.0)
            retrieved_chunks.append("")
            indices.append(-1)
        
        answer_segments = []
        reason = query.get('reason', '')
        if query['type'] == 'answerable':
            answer_segments = query.get('answer_segments', [])
        
        result = {
            'id': query['id'],
            'question': query['question'],
            'timestamp': query.get('timestamp', 0.0),
            'type': query['type'],
            'retrieved_indices': indices,
            'distances': similarities,
            'retrieved_chunks': retrieved_chunks,
            'answer_segments': answer_segments,
            'reason': reason if query['type'] == 'unanswerable' else ''
        }
        tfidf_results.append(result)
        
        print(f"\nQuery {query['id']}: {query['question']} (Timestamp: {query.get('timestamp', 0.0)})")
        if query['type'] == 'unanswerable':
            print(f"Unanswerable: {reason}")
        for i, (idx, sim, chunk) in enumerate(zip(indices, similarities, retrieved_chunks)):
            match = " (Match)" if query['type'] == 'answerable' and any(chunk in ans for ans in answer_segments) else ""
            print(f"  Result {i + 1}: Index={idx}, Similarity={sim:.4f}, Text={chunk}{match}")

    # Save TF-IDF results
    with open(tfidf_json_file, 'w') as f:
        json.dump(tfidf_results, f, indent=2)
    print(f"TF-IDF results saved to {tfidf_json_file}")

    #* BM25 Retrieval
    print("Performing BM25 retrieval...")
    bm25 = create_bm25_index(text_chunks)
    bm25_results = []
    for query in golden_queries:
        query_text = query['question']
        results = retrieve_bm25(query_text, bm25, text_chunks, k=5)
        
        # Ensure k results, padding with zeros if fewer
        indices = [i for i, res in enumerate(results)][:k]
        similarities = [res['score'] for res in results][:k]
        retrieved_chunks = [res['chunk'] for res in results][:k]
        while len(similarities) < k:
            similarities.append(0.0)
            retrieved_chunks.append("")
            indices.append(-1)
        
        answer_segments = []
        reason = query.get('reason', '')
        if query['type'] == 'answerable':
            answer_segments = query.get('answer_segments', [])
        
        result = {
            'id': query['id'],
            'question': query['question'],
            'timestamp': query.get('timestamp', 0.0),
            'type': query['type'],
            'retrieved_indices': indices,
            'distances': similarities,
            'retrieved_chunks': retrieved_chunks,
            'answer_segments': answer_segments,
            'reason': reason if query['type'] == 'unanswerable' else ''
        }
        bm25_results.append(result)
        
        print(f"\nQuery {query['id']}: {query['question']} (Timestamp: {query.get('timestamp', 0.0)})")
        if query['type'] == 'unanswerable':
            print(f"Unanswerable: {reason}")
        for i, (idx, sim, chunk) in enumerate(zip(indices, similarities, retrieved_chunks)):
            match = " (Match)" if query['type'] == 'answerable' and any(chunk in ans for ans in answer_segments) else ""
            print(f"  Result {i + 1}: Index={idx}, Similarity={sim:.4f}, Text={chunk}{match}")

    # Save BM25 results

    with open(bm25_json_file, 'w') as f:
        json.dump(bm25_results, f, indent=2)
    print(f"BM25 results saved to {bm25_json_file}")

if __name__ == "__main__":
    main()

# %%

