import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_rag_answer(question: str, retrieved_docs: list[dict]) -> str:
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
    context_block = "\n\n".join(
        [
            f"Source {i+1} (score: {doc['score']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the QUESTION using ONLY the provided CONTEXT.
If the answer is not found in the context, say:
"I don't know based on the provided context."

Do not make up information.
Be concise and clear.
Cite sources using [Source X] format.

-----------------------
CONTEXT:
{context_block}
-----------------------

QUESTION:
{question}

FINAL ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",  # use the one that worked for you
        contents=prompt,
    )

    return response.text
