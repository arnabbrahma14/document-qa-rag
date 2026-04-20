from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# import your pipeline components
from app.ingestion import extract_text
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, model, embeddings_for_mmr
from app.pinecone_service import (
    store_pc_embeddings,
    is_index_empty,
    query_pinecone
)
from app.llm_service import generate_rag_answer, prepare_context_and_citations
from app.bm25_index import BM25Index
from app.reranker import deduplicate_chunks, rerank_chunks, apply_mmr
from app.retrieval_response_formatter import format_results


# -----------------------------
# GLOBAL STATE (shared)
# -----------------------------
bm25_index = None


# -----------------------------
# STARTUP HANDLER
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_index

    print("🚀 Initializing RAG pipeline...")

    pages = extract_text("data/merchant_of_venice_summary.pdf")
    chunks = chunk_text(pages, "merchant_of_venice_summary.pdf")

    vectors = generate_embeddings(chunks)
    bm25_index = BM25Index(chunks)

    if is_index_empty():
        store_pc_embeddings(vectors)

    print("✅ RAG pipeline ready!")

    yield


app = FastAPI(lifespan=lifespan)


# -----------------------------
# REQUEST / RESPONSE SCHEMA
# -----------------------------
class QueryRequest(BaseModel):
    question: str


# -----------------------------
# CORE PIPELINE FUNCTION
# -----------------------------
def run_rag_pipeline(question: str):
    query_embedding = model.encode([question])[0]

    results = query_pinecone(query_embedding)
    results_bm25 = bm25_index.search(question, top_k=10)

    hybrid_result = (
        format_results(results, "vector") +
        format_results(results_bm25, "bm25")
    )

    dedup = deduplicate_chunks(hybrid_result)
    reranked = rerank_chunks(question, dedup)

    query_embed, chunk_embed = embeddings_for_mmr(question, reranked)
    final_chunks = apply_mmr(query_embed, chunk_embed, reranked)

    context, citations = prepare_context_and_citations(final_chunks)

    answer = generate_rag_answer(question, context)

    return {
        "answer": answer,
        "contexts": [c["text"] for c in final_chunks]
    }


# -----------------------------
# API ROUTES
# -----------------------------
import time

@app.post("/ask")
def ask_question(request: QueryRequest):
    start_time = time.time()

    try:
        result = run_rag_pipeline(request.question)

        duration = time.time() - start_time
        print(f"📥 Question: {request.question}")
        print(f"📤 Answer length: {len(result['answer'])} chars")
        print(f"⏱️ Time taken: {duration:.2f} sec")

        return result

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Error: {str(e)} | Time: {duration:.2f}s")

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}