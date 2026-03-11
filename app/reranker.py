from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def deduplicate_chunks(combined_results):
    
    seen = set()
    unique_chunks = []

    for chunk in combined_results:
        text = chunk["text"]

        if text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    return unique_chunks

def rerank_chunks(query, chunks):

    pairs = [(query, chunk["text"]) for chunk in chunks]

    scores = reranker_model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked_chunks = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_chunks

def apply_mmr(query_embedding, chunk_embeddings, chunks, top_k=5, lambda_param=0.7):

    selected = []
    selected_indices = []

    similarity_to_query = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    )[0]

    similarity_between_chunks = cosine_similarity(chunk_embeddings)

    first_idx = np.argmax(similarity_to_query)
    selected_indices.append(first_idx)

    while len(selected_indices) < top_k:

        mmr_scores = []

        for i in range(len(chunks)):

            if i in selected_indices:
                continue

            relevance = similarity_to_query[i]

            redundancy = max(
                similarity_between_chunks[i][j]
                for j in selected_indices
            )

            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy

            mmr_scores.append((i, mmr))

        next_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(next_idx)

    selected = [chunks[i] for i in selected_indices]

    return selected