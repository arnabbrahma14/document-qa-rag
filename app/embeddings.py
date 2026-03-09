from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(texts)

    vectors = []

    for chunk, embedding in zip(chunks, embeddings):

        vectors.append({
            "id": chunk["id"],          # unique chunk id
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["content"],
                "document_name": chunk["metadata"]["document_name"],
                "page_number": chunk["metadata"]["page_number"],
                "chunk_id": chunk["metadata"]["chunk_id"]
            }
        })

    return vectors