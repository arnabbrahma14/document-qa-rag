import os
from dotenv import load_dotenv
# from google import genai
from groq import Groq

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def prepare_context_and_citations(retrieval_results):

    context_parts = []
    citations = []

    for i, chunk in enumerate(retrieval_results, start=1):

        text = chunk["text"]
        document = chunk["document_name"]
        page = chunk["page"]

        # Add citation index to chunk
        context_parts.append(f"[{i}] {text}")

        # Save citation metadata
        citations.append({
            "id": i,
            "document": document,
            "page": page
        })

    context = "\n\n".join(context_parts)

    return context, citations

    # context_parts = []
    # citations = []

    # for i, match in enumerate(retrieval_results["matches"], start=1):

    #     metadata = match["metadata"]

    #     text = metadata["text"]
    #     document = metadata["document_name"]
    #     page = metadata["page_number"]

    #     # Add citation index to chunk
    #     context_parts.append(f"[{i}] {text}")

    #     # Save citation metadata
    #     citations.append({
    #         "id": i,
    #         "document": document,
    #         "page": page
    #     })

    # context = "\n\n".join(context_parts)

    # return context, citations


def generate_rag_answer(question: str, context_block: str) -> str:
    """
    retrieved_docs = [
        {
            "id": "...",
            "score": 0.87,
            "text": "...",
            "source": "..."
        }
    ]
    """

    # Build context block
    # context_block = "\n\n".join(
    #     [
    #         f"Source {i+1} (score: {doc['score']}):\n{doc['text']}"
    #         for i, doc in enumerate(retrieved_docs)
    #     ]
    # )

    prompt = f"""
You are a helpful AI assistant.

Answer the QUESTION using ONLY the provided CONTEXT.
If the answer is not found in the context, say:
"I don't know based on the provided context."

Do not make up information.
Be concise and clear.
Whenever you use information from the context,
cite it using [number].

-----------------------
CONTEXT:
{context_block}
-----------------------

QUESTION:
{question}

FINAL ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or llama3-70b-8192
        messages=[
             {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

def format_response(answer, citations):

    sources = "\n\nSources:\n"

    for citation in citations:

        sources += (
            f"[{citation['id']}] "
            f"{citation['document']} — Page {citation['page']}\n"
        )

    final_response = answer + sources

    return final_response