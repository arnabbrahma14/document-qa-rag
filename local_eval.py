import json
from rag_test import run_rag_pipeline

import os

from dotenv import load_dotenv
load_dotenv()

with open("golden_dataset_merchant_venice.json") as f:
    golden_data = json.load(f)

records = golden_data["records"]

eval_data = []

for sample in records:
    question = sample["question"]

    result = run_rag_pipeline(question)

    eval_data.append({
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": sample["ground_truth_answer"]
    })

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
)

from datasets import Dataset

dataset = Dataset.from_list(eval_data)

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper

ragas_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset,
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm  
)

print(result)