import os
from ingestion import extract_text
from chunking import chunk_text

from dotenv import load_dotenv
load_dotenv()

from ragas.testset import TestsetGenerator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

pages = extract_text("../data/merchant_of_venice_summary.pdf")
chunks = chunk_text(pages, "merchant_of_venice_summary.pdf")

# -------------------------------
# 3. Build LangChain Docs
# -------------------------------

# Merge every N chunks into one document
def merge_chunks(chunks, merge_every=5):
    docs = []
    for i in range(0, len(chunks), merge_every):
        batch = chunks[i:i + merge_every]
        merged_text = "\n\n".join(c["content"] for c in batch)
        docs.append(
            Document(
                page_content=merged_text,
                metadata={
                    "document": batch[0]["metadata"]["document_name"],  # dict access
                    "page": batch[0]["metadata"]["page_number"]  
                }
            )
        )
    return docs

docs = merge_chunks(chunks, merge_every=5)


# docs = []
# for chunk in chunks:
#     docs.append(
#         Document(
#             page_content=chunk["content"],
#             metadata={
#                 "document": chunk["metadata"]["document_name"],
#                 "page": chunk["metadata"]["page_number"]
#             }
#         )
#     )

# -------------------------------
# 4. Generate QA Testset
# -------------------------------



def generate_testset(docs):
    generator_llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"  # newer than embedding-001
    )

    # Fix 1 & 2: instantiate TestsetGenerator with both llm and embeddings
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        generator_llm,   # critic_llm — can use same model
        embeddings
    )

    # Fix 3: now correctly calling generate on the generator, not the llm
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=3,
        with_debugging_logs=True
    )

    return testset

# -------------------------------
# 5. Save Dataset
# -------------------------------

def save_dataset(testset):
    df = testset.to_pandas()

    os.makedirs("evaluation_dataset", exist_ok=True)

    df.to_json(
        "evaluation_dataset/golden_dataset.json",
        orient="records",
        indent=2
    )

    print("Dataset saved successfully")

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    print("Generating QA testset...")
    testset = generate_testset(docs)  

    print("Saving dataset...")
    save_dataset(testset)