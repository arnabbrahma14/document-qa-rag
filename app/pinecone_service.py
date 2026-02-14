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
                dimension=384,   # because MiniLM is 384
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    return pinecone.Index(index_name)

pc_index = get_index()

def store_pc_embeddings(chunks, embeddings):
     
    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
                "metadata": {
                    "text": chunk["content"]
                }
            })

    pc_index.upsert(vectors=vectors)
    


# class PineconeService:

#     def __init__(self):
#         self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         self.index_name = "document-qa-index"

#         # Create index if not exists
#         if self.index_name not in self.pc.list_indexes().names():
#             self.pc.create_index(
#                 name=self.index_name,
#                 dimension=384,   # because MiniLM is 384
#                 metric="cosine",
#                 spec=ServerlessSpec(
#                     cloud="aws",
#                     region="us-east-1"
#                 )
#             )

#         self.index = self.pc.Index(self.index_name)

#     def upsert(self, chunks, embeddings):

#         vectors = []

#         for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#             vectors.append({
#                 "id": f"chunk-{i}",
#                 "values": embedding.tolist(),
#                 "metadata": {
#                     "text": chunk["content"]
#                 }
#             })

#         self.index.upsert(vectors=vectors)

#     def query_pinecone(self, query_embedding, top_k=3):

#         return self.index.query(
#             vector=query_embedding.tolist(),
#             top_k=top_k,
#             include_metadata=True
#         )
