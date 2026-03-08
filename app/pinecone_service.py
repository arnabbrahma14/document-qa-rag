import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "document-qa-index"


def get_index():
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pinecone.Index(index_name)


pc_index = get_index()


# ✅ Check if index is empty
def is_index_empty():
    stats = pc_index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    return total_vectors == 0


# ✅ Store embeddings (with batching)
def store_pc_embeddings(chunks, embeddings):

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"alice-chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["content"],
                "source": "merchant_of_venice_summary.pdf"
            }
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        pc_index.upsert(vectors=vectors[i:i+batch_size])


# ✅ Query function (LLM ready)
def query_pinecone(query_embedding, top_k=10):

    results = pc_index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    formatted_results = []

    for match in results["matches"]:
        formatted_results.append({
            "id": match["id"],
            "score": round(match["score"], 4),
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "unknown")
        })
    return formatted_results
