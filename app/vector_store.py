import chromadb

client = chromadb.Client()
collection = client.create_collection("doc_qa")

def store_embeddings(chunks, embeddings):
    ids = [str(i) for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{"page": chunk["page_number"]} for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
