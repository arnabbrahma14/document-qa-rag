from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, docs, top_k=5):

    pairs = [(query, doc) for doc in docs]

    scores = reranker_model.predict(pairs)

    scored = list(zip(docs, scores))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]