from app.embeddings import model
from app.vector_store import collection

def query_rag(question, top_k=3):
    query_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas
