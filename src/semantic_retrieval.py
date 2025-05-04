import faiss
import numpy as np

def faiss_retrieve(embeddings, query_embedding, text_chunks, k=5):
    embeddings = embeddings.cpu().numpy().astype('float32')    
    query_embedding = query_embedding.cpu().numpy().astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    
    results = [text_chunks[i] for i in indices[0]]
    
    return indices[0], distances[0], results

def create_postgres_ivfflat_table(conn, cur):
    # Implement your table creation logic here
    # Example: Create table with chunk (text) and embedding (vector)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            chunk TEXT,
            embedding VECTOR(384)
        );
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding);
    """)
    conn.commit()

# Placeholder for IVF-Flat retrieval (implement based on your setup)
def postgres_ivfflat_retrieve(conn, cur, query_emb, text_chunks, k=5):
    # Convert query embedding to string format for PostgreSQL
    query_emb_str = str(query_emb.tolist()[0])
    cur.execute("""
        SELECT id, chunk, embedding <-> %s::vector AS distance
        FROM chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    """, (query_emb_str, query_emb_str, k))
    rows = cur.fetchall()
    
    indices = [row[0] - 1 for row in rows]  # Adjust for 0-based indexing
    distances = [row[2] for row in rows]
    retrieved_chunks = [row[1] for row in rows]
    return np.array(indices), np.array(distances), retrieved_chunks



def create_postgres_hnsw_table(conn, cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            chunk TEXT,
            embedding VECTOR(384)
        );
        CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()

def retrieve_postgres_hnsw(conn, cur, query_emb, chunks, k=5):

    # Flatten if 2D array
    if query_emb.ndim == 2:
        query_emb = query_emb.squeeze()
    query_emb_str = str(query_emb.tolist())  # Convert entire vector to string
    print(f"Query embedding shape: {query_emb.shape}")
    print(f"Query embedding string: {query_emb_str[:50]}...")  # Truncate for readability
    
    cur.execute("""
        SELECT id, chunk, 1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_emb_str, query_emb_str, k))
    
    rows = cur.fetchall()
    indices = [row[0] - 1 for row in rows]  # Adjust for 0-based indexing
    similarities = [row[2] for row in rows]
    retrieved_chunks = [row[1] for row in rows]
    
    return np.array(indices), np.array(similarities), retrieved_chunks

