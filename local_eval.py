"""
local_eval.py — RAGAS 0.4.3, with rate limit handling
"""
import json
import os
import time
import warnings
from dotenv import load_dotenv
load_dotenv()

from rag_test import run_rag_pipeline

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
    )

from openai import OpenAI
from ragas.llms import llm_factory

# ─────────────────────────────────────────────────────────────────────────────
# LLM — choose ONE of the three options below, comment out the others
# ─────────────────────────────────────────────────────────────────────────────

# ── Option A: Groq with a smaller/cheaper model ───────────────────────────
# llama-3.1-8b-instant uses ~8x fewer tokens than llama-3.3-70b-versatile.
# Free tier limit: 500k TPD for 8b vs 100k for 70b.
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
ragas_llm = llm_factory(
    model="llama-3.1-8b-instant",   # ← switched from llama-3.3-70b-versatile
    client=groq_client,
)

# ── Option B: OpenAI gpt-4o-mini (cheap, no daily token cap) ─────────────
# Uncomment if you have an OpenAI key. gpt-4o-mini costs ~$0.002 per eval run.
#
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)

# ── Option C: Ollama (fully local, no API key, no rate limits) ────────────
# Requires Ollama running locally: https://ollama.com
# Pull a model first:  ollama pull llama3.2
#
# local_client = OpenAI(
#     api_key="ollama",
#     base_url="http://localhost:11434/v1",
# )
# ragas_llm = llm_factory(model="llama3.2", client=local_client)

# ─────────────────────────────────────────────────────────────────────────────
# Inject LLM onto singletons
# ─────────────────────────────────────────────────────────────────────────────
faithfulness.llm      = ragas_llm
context_precision.llm = ragas_llm
context_recall.llm    = ragas_llm

metrics = [faithfulness, context_precision, context_recall]

print("\nMetrics:")
for m in metrics:
    print(f"  {type(m).__name__:25s} llm={m.llm is not None}")

# ─────────────────────────────────────────────────────────────────────────────
# Build eval dataset — with per-record delay to avoid mid-run rate limits
# ─────────────────────────────────────────────────────────────────────────────
with open("golden_dataset_merchant_venice.json") as f:
    golden_data = json.load(f)

records = golden_data["records"]

# Adjust BATCH_SIZE and DELAY_SECONDS if you keep hitting 429s.
# With llama3-8b on Groq free tier, 5 records per batch with 5s gap is safe.
BATCH_SIZE     = 5      # process N records, then pause
DELAY_SECONDS  = 5      # seconds to wait between batches

eval_data = []

for i, sample in enumerate(records, 1):
    question = sample["question"]
    print(f"[{i:02d}/{len(records)}] {question[:70]}...")

    result = run_rag_pipeline(question)

    eval_data.append({
        "question":     question,
        "answer":       result["answer"],
        "contexts":     result["contexts"],
        "ground_truth": sample["ground_truth_answer"],
    })

    # Pause between batches to stay within rate limits
    if i % BATCH_SIZE == 0 and i < len(records):
        print(f"  ⏸  Pausing {DELAY_SECONDS}s to respect rate limits...")
        time.sleep(DELAY_SECONDS)

# ─────────────────────────────────────────────────────────────────────────────
# Run RAGAS — also wrapped in retry for 429s that slip through
# ─────────────────────────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate

dataset = Dataset.from_list(eval_data)
print("\nRunning RAGAS evaluation...\n")

MAX_RETRIES = 3
for attempt in range(1, MAX_RETRIES + 1):
    try:
        result = evaluate(dataset=dataset, metrics=metrics)
        break
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            wait = 60 * attempt   # 60s, 120s, 180s
            print(f"  ⚠️  Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                  f"Waiting {wait}s before retry...")
            time.sleep(wait)
        else:
            raise   # re-raise anything that isn't a rate limit error
else:
    print("  ❌ All retries exhausted. Wait for your quota to reset and re-run.")
    raise SystemExit(1)

print(result)

df = result.to_pandas()
df.to_csv("ragas_results.csv", index=False)
print("\nPer-record results saved → ragas_results.csv")