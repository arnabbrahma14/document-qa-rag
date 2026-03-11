from app.ingestion import extract_text
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, model, embeddings_for_mmr
from app.pinecone_service import (
    store_pc_embeddings,
    is_index_empty,
    query_pinecone
)
from app.llm_service import generate_rag_answer, prepare_context_and_citations, format_response
from app.bm25_index import BM25Index
from app.reranker import deduplicate_chunks, rerank_chunks, apply_mmr
from app.retrieval_response_formatter import format_results
import json

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
chunks = chunk_text(pages, "merchant_of_venice_summary.pdf")
vectors = generate_embeddings(chunks)

store_pc_embeddings(vectors)

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



    hybrid_result = format_results(results, "vector") + format_results(results_bm25, "bm25")

    print("#" * 150)
    print("Hybrid Result = Vector + BM25")
    print("#" * 150)

    print(json.dumps(hybrid_result, indent=4))

    # for x in hybrid_result:
    #     print(x)

    #De duplicating using set
    deduplicate_result = deduplicate_chunks(hybrid_result)

    print("\n")
    print("#" * 150)
    print("Deduplicate Result")
    print("#" * 150)
    print("\n")

    #Re ranking results
    re_ranked_result = rerank_chunks(question, deduplicate_chunks(hybrid_result))

    # for x in re_ranked_result:
    #     print(x)

    print(json.dumps(re_ranked_result, indent=4))


    #Creating embeddings for mmr 
    query_embed, chunk_embed = embeddings_for_mmr(question, re_ranked_result)

    #final chunks after mmr
    final_result = apply_mmr(query_embed, chunk_embed, re_ranked_result)

    print("\n")
    print("#" * 150)
    print("Final Result")
    print("#" * 150)
    print("\n")

    # for x in final_result:
    #     print(x)

    print(json.dumps(final_result, indent=4))

    
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

    # hybrid_chunk = [res['text'] for res in results] + [res[0]['content'] for res in results_bm25]

    # print("#" * 1000)
    # print("Initial top k Chunk")
    # for res in hybrid_chunk:
    #     print(res[:300] + "\n")
    # print("#" * 1000)
    
    # rerank_chunk = rerank_documents(question, hybrid_chunk, 5)

    # print("#" * 1000)
    # print("Reranked Chunk")
    # for res in rerank_chunk:
    #     print(res[:300] + "\n")
    # print("#" * 1000)

    context, citations = prepare_context_and_citations(final_result)

    

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
    print(context)
    print("\n" + "="*80)
    
    # Step 3: Generate final grounded answer
    final_answer = generate_rag_answer(question, context)

    print("\n🤖 FINAL ANSWER:\n")
    print(format_response(final_answer, citations))
    print("\n" + "=" * 80)