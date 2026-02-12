from app.ingestion import extract_text
from app.chunking import chunk_text
from app.embeddings import generate_embeddings
from app.vector_store import store_embeddings
from app.rag_pipeline import query_rag
import json


pages = extract_text("data/alice_in_wonderland.md")
      
#print(json.dumps(pages[0], indent=4)) 

# Step 2: Chunk
chunks = chunk_text(pages)

# Step 3: Embed
embeddings = generate_embeddings(chunks)

# Step 4: Store
store_embeddings(chunks, embeddings)

print("Document ingested successfully!")


while(True):
# Step 5: Ask Question
   question = input("Ask a question: ")

   docs, metas = query_rag(question)

   print("\nAnswer Context:\n")
   for doc, meta in zip(docs, metas):
       print(f"(Page {meta['page']})")
       print(doc)
       print("-" * 50)