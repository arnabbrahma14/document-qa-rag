from app.embeddings import model
from app.pinecone_service import pc_index

def query_pinecone(question, top_k=3):
    query_embedding = model.encode([question])

    return pc_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )