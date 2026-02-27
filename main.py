from app.ingestion import extract_text
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, model
from app.pinecone_service import (
    store_pc_embeddings,
    is_index_empty,
    query_pinecone
)
from app.llm_service import generate_rag_answer

# -------------------------------
# ✅ Single-time ingestion logic
# -------------------------------

if is_index_empty():
    print("🔄 Index is empty. Running ingestion...\n")

    pages = extract_text("data/alice_in_wonderland.md")
    chunks = chunk_text(pages)
    embeddings = generate_embeddings(chunks)

    store_pc_embeddings(chunks, embeddings)

    print("✅ Document ingested successfully!\n")

else:
    print("✅ Index already contains data. Skipping ingestion.\n")


# -------------------------------
# ✅ Query Loop
# -------------------------------

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Embed question
    query_embedding = model.encode([question])[0]

    # Retrieve from Pinecone
    results = query_pinecone(query_embedding)

    print("\n🔎 Retrieved Contexts:\n")

    for i, result in enumerate(results, 1):
        print(f"[{i}] ID: {result['id']}")
        print(f"    Score : {result['score']}")
        print(f"    Source: {result['source']}")
        print(f"    Text  : {result['text'][:300]}...")
        print("-" * 60)

    # -------------------------------
    # ✅ LLM-ready formatted context
    # -------------------------------

    llm_context = "\n\n".join(
        [f"Source {i+1}:\n{res['text']}"
         for i, res in enumerate(results)]
    )

    print("\n📦 LLM INPUT FORMAT:\n")
    print("QUESTION:")
    print(question)
    print("\nCONTEXT:")
    print(llm_context)
    print("\n" + "="*80)
    
    # Step 3: Generate final grounded answer
    final_answer = generate_rag_answer(question, results)

    print("\n🤖 FINAL ANSWER:\n")
    print(final_answer)
    print("\n" + "=" * 80)