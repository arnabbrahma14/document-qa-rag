import json
import os
import sys
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    # answer_relevancy,
    # context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# 👉 Your pipeline
from rag_test import run_rag_pipeline


# -------------------------------
# 1. Setup
# -------------------------------
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=512,
)

ragas_llm = LangchainLLMWrapper(llm) #n=1 for evaluation, to get a single answer per question else throwing error due to multiple answers returned by default by ragas. but groq allows 1.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# -------------------------------
# 2. Load Golden Dataset
# -------------------------------
with open("golden_dataset_merchant_venice.json") as f:
    golden = json.load(f)

records = golden["records"]


# -------------------------------
# 3. Run RAG + Build Eval Data
# -------------------------------
eval_data = []

for sample in records[:5]:
    question = sample["question"]

    try:
        result = run_rag_pipeline(question)

        eval_data.append({
            "question": question,
            "answer": result["answer"],
            "contexts": result["contexts"],   # must be list[str]
            "ground_truth": sample["ground_truth_answer"]
        })

    except Exception as e:
        print(f"❌ Error on question: {question}")
        print(str(e))
        continue


dataset = Dataset.from_list(eval_data)


# -------------------------------
# 4. Evaluate
# -------------------------------
print("\n🚀 Running RAG Evaluation...\n")

scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        # answer_relevancy,
        # context_precision,
        context_recall,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n📊 Evaluation Results:")
print(scores)


# -------------------------------
# 5. CI/CD Gate (FAIL if bad)
# -------------------------------
thresholds = {
    "faithfulness": 0.7,
    # "answer_relevancy": 0.7,
    # "context_precision": 0.6,
    "context_recall": 0.6,
}

fail = False

for metric, threshold in thresholds.items():
    value = scores[metric]
    
    # Handle list vs float
    if isinstance(value, list):
        avg_value = sum(value) / len(value)
    else:
        avg_value = value

    if avg_value < threshold:
        print(f"❌ {metric} below threshold: {avg_value} < {threshold}")
    
    # if value < threshold:
    #     print(f"❌ {metric} below threshold: {value} < {threshold}")
        fail = True

if fail:
    print("\n🚫 Evaluation FAILED")
    sys.exit(1)
else:
    print("\n✅ Evaluation PASSED")
    sys.exit(0)