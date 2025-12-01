import os
import json
import time
from openai import OpenAI
import tiktoken

# --- CONFIG ---
INPUT_FILE = "additional_prompts.txt"   # your uploaded JSON array
BATCH_JSONL = "batch_tasks.jsonl"
RAW_RESULTS_FILE = "raw_batch_results.jsonl"
ENRICHED_OUTPUT = "additional_results_gpt4o.json"
MODEL = "gpt-4o"       # model to use
MAX_TOKENS = 512
POLL_INTERVAL = 10     # seconds between batch status polls
# ---------------

# init client (SDK reads OPENAI_API_KEY from env)
client = OpenAI()

# helper: token counting
def count_tokens(text, model_name=MODEL):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

# 1) Load the input JSON array
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    items = json.load(f)   # expects a list of objects with PromptText

# 2) All prompts are allowed (no moderation step)
allowed = [{"index": idx, "obj": obj} for idx, obj in enumerate(items)]

# 3) Build JSONL tasks for the Batch API
tasks = []
for a in allowed:
    orig_index = a["index"]
    o = a["obj"]
    custom_id = f"{o.get('TaskID','task')}-{orig_index}"
    task = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": o.get("PromptText", "")}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2
        }
    }
    tasks.append(task)

# write tasks to jsonl
with open(BATCH_JSONL, "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"Wrote {len(tasks)} tasks to {BATCH_JSONL}")

# 4) Upload file for batch purpose
batch_file = client.files.create(file=open(BATCH_JSONL, "rb"), purpose="batch")
print("Uploaded batch file:", batch_file.id)

# 5) Create batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print("Batch job created:", batch_job.id)

# 6) Poll for completion
job_id = batch_job.id
while True:
    job = client.batches.retrieve(job_id)
    status = job.status
    print("Batch status:", status)
    if status in ("completed", "failed", "cancelled"):
        break
    time.sleep(POLL_INTERVAL)

if job.status != "completed":
    print("Batch finished with status:", job.status)
    raise SystemExit(1)

# 7) Download the result file
result_file_id = job.output_file_id
result_bytes = client.files.content(result_file_id).content
with open(RAW_RESULTS_FILE, "wb") as f:
    f.write(result_bytes)
print("Saved raw results to", RAW_RESULTS_FILE)

# 8) Parse results and enrich with ResponseText & ResponseLength
enriched = []
orig_map = { f"{a['obj'].get('TaskID','task')}-{a['index']}": a['obj'] for a in allowed }

with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        custom_id = rec.get("custom_id") or rec.get("id")
        response_body = None
        if "response" in rec and isinstance(rec["response"], dict):
            response_body = rec["response"].get("body")
        elif "modelOutput" in rec:
            response_body = rec.get("modelOutput")
        else:
            response_body = rec.get("response")

        response_text = None
        response_token_count = None

        try:
            choices = response_body.get("choices", [])
            if choices:
                response_text = choices[0]["message"]["content"]
        except Exception:
            response_text = None

        try:
            usage = response_body.get("usage", {})
            response_token_count = usage.get("total_tokens") or usage.get("output_tokens")
        except Exception:
            response_token_count = None

        if response_token_count is None and response_text is not None:
            response_token_count = count_tokens(response_text, MODEL)

        original = orig_map.get(custom_id, {})
        enriched_obj = dict(original)
        enriched_obj["ResponseText"] = response_text
        enriched_obj["ResponseLength"] = int(response_token_count or 0)
        enriched.append(enriched_obj)

with open(ENRICHED_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print(f"Wrote enriched results ({len(enriched)}) to {ENRICHED_OUTPUT}")
print("Done.")
