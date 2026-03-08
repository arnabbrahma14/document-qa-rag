from app.ingestion import extract_text
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, model
from app.pinecone_service import (
    store_pc_embeddings,
    is_index_empty,
    query_pinecone
)
from app.llm_service import generate_rag_answer
from app.bm25_index import BM25Index
from app.reranker import rerank_documents

# -------------------------------
# ✅ Single-time ingestion logic
# -------------------------------

# if is_index_empty():
#     print("🔄 Index is empty. Running ingestion...\n")

#     pages = extract_text("data/merchant_of_venice_summary.pdf")
#     chunks = chunk_text(pages)
#     embeddings = generate_embeddings(chunks)

#     store_pc_embeddings(chunks, embeddings)

#     print("✅ Document ingested successfully!\n")

# else:
#     print("✅ Index already contains data. Skipping ingestion.\n")
    
pages = extract_text("data/merchant_of_venice_summary.pdf")
chunks = chunk_text(pages)
embeddings = generate_embeddings(chunks)

store_pc_embeddings(chunks, embeddings)

#Using BM_25 searching
bm25_index = BM25Index(chunks)


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

    
    #Retrieve from bm_25
    results_bm25 = bm25_index.search(question, top_k=10)
    
    # print("#" * 120 + "\nPinecone Search Result\n"  + "#" * 120)

    # for i, result in enumerate(results, 1):
    #     print(f"[{i}] ID: {result['id']}")
    #     print(f"    Score : {result['score']}")
    #     print(f"    Source: {result['source']}")
    #     print(f"    Text  : {result['text']}")
    #     print(f"    Text Length : {len(result['text'])}")
    #     print("-" * 60)
    
    # print("#" * 120 + "\nBM_25 Search Result\n"  + "#" * 120)
    # for chunk, score in results_bm25:
    #     print(score)
    #     print(chunk["content"])

     
    # pinecone_context = "\n\n".join(
    #     [f"Source {i+1}:\n{res['text']}"
    #      for i, res in enumerate(results)]
    # )

    # bm25_context = "\n\n".join(
    #     [f"Source {i+len(results) + 1}:\n{chunk[0]['content']}" 
    #      for i, chunk in enumerate(results_bm25)]
    # )

    hybrid_chunk = [res['text'] for res in results] + [res[0]['content'] for res in results_bm25]

    print("#" * 1000)
    print("Initial top k Chunk")
    for res in hybrid_chunk:
        print(res[:300] + "\n")
    print("#" * 1000)
    
    rerank_chunk = rerank_documents(question, hybrid_chunk, 5)

    print("#" * 1000)
    print("Reranked Chunk")
    for res in rerank_chunk:
        print(res[:300] + "\n")
    print("#" * 1000)

    llm_context = "\n\n".join([f"Source {i+1}:\n{chunk}" 
         for i, chunk in enumerate(rerank_chunk)])

    # print("\n🔎 Retrieved Contexts:\n")

    # for i, result in enumerate(results, 1):
    #     print(f"[{i}] ID: {result['id']}")
    #     print(f"    Score : {result['score']}")
    #     print(f"    Source: {result['source']}")
    #     print(f"    Text  : {result['text'][:300]}...")
    #     print("-" * 60)

    # # -------------------------------
    # # ✅ LLM-ready formatted context
    # # -------------------------------

    # llm_context = "\n\n".join(
    #     [f"Source {i+1}:\n{res['text']}"
    #      for i, res in enumerate(results)]
    # )

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