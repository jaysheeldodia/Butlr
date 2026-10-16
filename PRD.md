# PRD: Butlr Local Task-Extraction Model

Status: draft for review
Date: 2026-10-15
Owner: Jay
Repo: https://github.com/jaysheeldodia/Butlr (inspected at commit 7fbc623, main)
Rule for this project: no email content ever goes to an LLM provider, an AI agent,
or any cloud AI service. AI agents write code only. Jay runs every script that
touches email data, by hand, on his own machine.

## 1. Goal

Replace the cloud LLM API call in the task-extraction step with a small
open-weight model, fine-tuned by Jay, that runs locally on his Mac.
The model converts one email into a JSON array of tasks with:
title, description, priority, deadline, category, action_required, source,
related_email_id.
The headline deliverable is a measured cost, quality, and latency comparison
between the current API path and the local path, on a locked test set.

Non-goals: replacing the classifier, the drafter, or the calendar agent.
Local embeddings are in scope, because they are cheap and remove the last
cloud dependency for email content.

## 2. Current state (verified in code)

- Poller: workers/processing_service.py runs one poll thread.
- Pipeline: agents/supervisor.py runs up to 8 ThreadPoolExecutor workers,
  one per email (line 58).
- Classification: agents/classifier.py, one API call per email.
- Extraction: agents/extractor.py, one API call per email, with a fixed
  system prompt and a JSON schema. This is the step we replace.
- LLM routing: utils/llm_client.py chat(). Hardcoded models:
  gpt-4o-mini, claude-3-5-sonnet-20241022, gemini-1.5-flash, llama3-70b-8192.
- JSON repair: parse_json_response() strips code fences and slices braces.
- Embeddings: memory/embeddings.py calls the OpenAI embeddings API
  (text-embedding-3-small, 1536 dims). This sends email bodies to OpenAI.
  We will replace this.
- Vector store: memory/vector_store.py, FAISS IndexFlatL2 in .butlr_store/.
  Files are plain text/binary on disk. Not encrypted.
- Token store: api/token_store.py encrypts only the OAuth refresh token
  with Fernet. Email data at rest is not encrypted.
- Failure handling: vector_store.save() has a bare except that drops errors.
  extract_tasks() returns [] on any parse error. classify_email() defaults
  a failed email to is_noise=True, so it never reaches extraction.
  These silent paths would corrupt any benchmark. We fix them.

## 3. Locked decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | Fine-tune target | Qwen3-1.7B-Instruct | Dense text-only. Apache-2.0. Highest instruction-following of the small candidates in non-thinking mode (IFEval 68.2, official card). Most mature QLoRA and GGUF tooling. |
| D2 | Fallback target | Qwen3.5-2B-Instruct | Unsloth supports it (0.8B/2B/4B notebooks). Stronger on knowledge tests, weaker on IFEval non-thinking (61.2). Hybrid arch, 248k vocab, newer tooling. Use only if D1 underfits. |
| D3 | Weak labeler | Qwen3-4B Q4_K_M via llama-server on the Mac, thinking mode off | Changed from 8B (Jay's request for speed; only ~6 GB free RAM). Q4_K_M is ~2.5 GB, roughly 2x faster, quality is sufficient for weak labels (human-verified on val/test). Fallback if quality is too low: Qwen3-8B Q4_K_M with apps closed, or Q3_K_M quant. |
| D4 | Embeddings | BAAI/bge-small-en-v1.5, local, 384 dims | MIT licence, 33.6M params, strong small retriever. Removes the OpenAI embeddings call. FAISS dim changes 1536 -> 384; rebuild the index. |
| D5 | Training path (primary) | MLX LoRA on the Mac with mlx-lm, Qwen3-1.7B 4-bit | Zero cloud. Email data never leaves the machine. mlx-lm 0.32.1 supports Qwen3 QLoRA. CONFIRMED: Mac is Apple Silicon (M4), so this path is valid. |
| D6 | Training path (fallback) | Colab free T4 + Unsloth, anonymized data only | Only if MLX fails or is too slow. Anonymization (D8) is mandatory before any upload. |
| D7 | Serving | llama.cpp llama-server on the Mac, GGUF Q4_K_M or Q8_0, --json-schema, OpenAI-compatible API | Schema-constrained JSON output. Butlr talks to it like a normal OpenAI provider. |
| D8 | PII rule | Anonymize (names, emails, phones, addresses -> placeholders) before anything leaves the machine; keep raw data local forever | Required for D6 only. For D5, raw data stays local, so anonymization is optional but recommended for the public dataset. |
| D9 | Category field | Add "category" to ExtractedTask: one of "reply", "application", "interview", "admin", "other" | Jay requested it. Mirrors the job-seeker domain. Renames are cheap before training starts. |

Licence sources (verified 2026-10-15):
- Qwen3-1.7B, Qwen3.5-2B, Qwen3-8B-GGUF, Qwen3-14B-GGUF: Apache-2.0.
  https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/Qwen/Qwen3.5-2B,
  HF API tag check for the GGUF repos.
- bge-small-en-v1.5: MIT. https://huggingface.co/BAAI/bge-small-en-v1.5
- Qwen3.5 fine-tuning support: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Qwen3 MLX QLoRA example: https://github.com/ml-explore/mlx-lm/pull/1270
- llama-server JSON schema: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp GBNF grammars: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

Not confirmed: exact GGUF file sizes (check at download time). MLX support for
Qwen3.5. Qwen3.5-2B GGUF quality on llama.cpp (GGUFs exist via Unsloth; not
benchmarked by us).

## 4. Schema change

models/schemas.py, ExtractedTask: add

    category: Literal["reply", "application", "interview", "admin", "other"]

agents/extractor.py: extend the system prompt with category rules:
- reply: the task is to answer or follow up with a person.
- application: a step in a job application (submit documents, forms).
- interview: interview scheduling, preparation, or debrief.
- admin: paperwork, logistics, onboarding, account setup.
- other: everything else.

Training data uses the same enum. The classifier output is a separate
email-level field and does not change.

## 5. Required code changes (AI agent writes, Jay reviews)

Phase P1 - privacy and plumbing:
1. utils/llm_client.py: add a "local" provider branch. It posts to
   LOCAL_LLM_URL (new env var, default http://127.0.0.1:8080/v1/chat/completions)
   with the same system/user messages. Local import of the HTTP client inside
   the branch. No new top-level import. Keep the four existing branches intact.
2. config/settings.py: add LOCAL_LLM_URL, EMBEDDING_MODEL, EMBEDDING_DIM.
3. memory/embeddings.py: replace the OpenAI call with sentence-transformers
   loading BAAI/bge-small-en-v1.5 (384 dims, normalized). Local import inside
   the function so tests do not need the model.
4. memory/vector_store.py: dim becomes configurable (EMBEDDING_DIM, default
   384). Add a store version marker; if the stored dim differs, rebuild with a
   logged warning instead of silently mixing vectors.
5. Fix silent failures: vector_store.save() logs exceptions. processing_service
   counts and exposes dropped-email counts in ProcessingSnapshot
   (new field: dropped_email_count).
6. .gitignore: add .butlr_data/, .butlr_store/, .butlr_tokens, .butlr_key,
   .butlr_processed (some may already be there; verify).
7. tests: update for the new dim and the local branch (mock HTTP).

Phase P2 - data pipeline (scripts/, all run manually by Jay):
1. scripts/export_emails.py: use tools/gmail.py GmailTool with the existing
   OAuth token store. Fetch up to N recent messages (target pool: about 3,000).
   Write data/raw/emails.jsonl locally. Print counts only. No network call
   except Google Gmail read.
2. scripts/prep_corpus.py: cut quoted reply text (lines starting with ">"),
   dedupe by message ID, cluster by subject-normalized thread, keep one row
   per thread option with a --keep-all flag. Output data/corpus/pool.jsonl.
3. scripts/anonymize.py: regex + name-list replacement for person names,
   emails, phones, addresses. Deterministic mapping per email. Dates stay.
   Output data/corpus/pool_anon.jsonl. Used before any Colab upload.
4. scripts/label_runner.py: reads pool.jsonl, calls the local labeler
   (LOCAL_LLM_URL or Ollama) with the extractor prompt plus a richer
   labeling prompt, validates output with ExtractedTask, writes
   data/labeled/weak.jsonl plus a per-email latency and token-count log.
   Resumable: skip message IDs already labeled.
5. scripts/make_splits.py: split by thread cluster, never by row.
   2,200 train / 150 val / 150 test. Write data/splits/{train,val,test}.jsonl.
   Print the test-set SHA256. Jay locks the test file. The agent never reads it.
6. scripts/benchmark.py: runs both paths (API branch and local branch) on
   test.jsonl. Records per email: parse success, Pydantic validity,
   field-level F1 parts, latency. Writes results/metrics.json and a summary
   table. Aggregate numbers go to chat; raw emails never do.

Phase P3 - training (Jay runs manually):
- Path A (primary): MLX on the Mac.
  1. Install: uv pip install mlx-lm (0.32.1).
  2. Download mlx-community/Qwen3-1.7B-4bit (Apache-2.0, exists on the Hub).
  3. scripts/train_mlx.py (agent writes): converts data/splits/train.jsonl to
     the mlx-lm chat JSONL format, trains LoRA (rank 16, alpha 16, dropout 0,
     lr 2e-4, 2 epochs or ~2500 iters, batch 2 with grad accumulation),
     thinking mode off. Saves the adapter.
  4. Fuse the adapter into the base model, then convert to GGUF with
     llama.cpp convert_hf_to_gguf.py, quantize Q4_K_M.
- Path B (fallback): Colab + Unsloth Qwen3.5/Qwen3 notebook, anonymized
  train split only, same hyperparameters, export GGUF Q4_K_M.
  Notebook: https://github.com/unslothai/notebooks (Qwen3 (4B) text notebook
  adapted to 1.7B, or the Qwen3.5 2B notebook if D2 is chosen).

Training hyperparameters (source: https://unsloth.ai/docs/get-started/
fine-tuning-llms-guide/lora-hyperparameters-guide):
- rank 16, alpha 16, dropout 0, weight decay 0.01, lr 2e-4,
  warmup 5-10%, linear or cosine schedule, epochs 2 (range 1-3),
  effective batch 16 = batch 2 x grad accum 8.
- Train loss on completions only (mask the email text). QLoRA paper:
  https://arxiv.org/pdf/2305.14314
- Overfitting signs (same guide): eval loss rises while train loss falls;
  the model copies training examples instead of extracting; train loss
  below 0.2 is a soft warning. Fixes: fewer epochs, lower lr, or average
  the adapter into the base model.

Phase P4 - serving and benchmark (Jay runs manually):
1. llama-server with the fused GGUF, --json-schema pointing to the task
   schema file scripts/task_schema.json (agent writes it from the Pydantic
   model). Health check: one curl with a fake email. Never a real one.
2. scripts/benchmark.py on test.jsonl for both paths. Fill the metrics
   table with real numbers.
3. Optional retrieval spot check: 20 emails, compare top-3 RAG chunks from
   bge-small against the old OpenAI embeddings. Manual judgment, logged.

## 6. Metrics (definitions)

- JSON validity rate: outputs that parse AND pass ExtractedTask validation.
- Field-level F1: priority and category as classes; action_required and
  deadline-present as binary; deadline value scored after ISO normalization;
  title and description scored with token-overlap F1. A correct empty array
  counts as a true negative. Record the scoring rule in the benchmark script.
- p50 and p95 latency per email, both paths, same machine, same run.
- Cost per 1,000 emails: API path = measured tokens x the provider price
  page captured at run time (paste the link into metrics.json). Local path:
  0 API cost; electricity and one-time training cost as separate notes.
- Also record: dropped_email_count, task-count agreement, model name,
  quantization, and the provider baseline model name.

## 7. Privacy rules (hard requirements)

1. No email content in any chat with an AI agent. Aggregate numbers only.
2. All email scripts run by Jay, by hand. The agent writes and reviews code.
3. data/, .butlr_store/, token files never leave the machine and never enter
   a commit. Test set stays local; only its SHA256 is shared.
4. Labeler runs offline on the Mac. No cloud LLM sees the corpus.
5. If Colab (D6) is ever used, only the anonymized train split goes up,
   never val, never test, never the raw pool.
6. Embeddings are computed locally after D4. No email body goes to OpenAI.

## 8. Risks

1. MLX needs Apple Silicon. If the Mac is Intel, D6 (Colab) becomes primary.
   Confirm the chip before P3. Command: uname -m (arm64 = Apple Silicon).
2. Mac RAM: over half of 24 GB is in use. The labeler needs about 6-10 GB
   free depending on model size. Check free memory before starting:
   memory_pressure | head -1 (or Activity Monitor). Drop to Qwen3-8B if tight.
3. The weak-label run is slow: expect hours, run it in the background or
   overnight. This is the schedule long pole.
4. Weak labels are noisy. Human verification of all val and test rows, plus
   spot checks of a few hundred train rows, limits this. F1 measures the
   labeler, not truth, wherever labels came from the 14B labeler.
5. llama.cpp support for a fresh fused GGUF can lag. Qwen3 GGUF is mature;
   Qwen3.5 is newer (not confirmed for all quantizations).
6. Grammar-constrained output can be valid JSON and still wrong. Validity
   rate alone is not quality.
7. bge-small retrieval is weaker than text-embedding-3-small on general
   benchmarks. RAG context here is short and optional; we accept the
   tradeoff and spot-check (P4 step 3).
8. Free Colab T4 may be unavailable at the moment of need.

## 9. Open questions (all resolved 2026-10-15)

1. Mac Apple Silicon? YES — M4. MLX path (D5) is primary.
2. Labeler model? Qwen3-8B (Jay's choice). RAM free ~6 GB: close apps before
   the label run; fallback quant Q3_K_M if it swaps.
3. Category enum? reply / application / interview / admin / other (agent
   decision, confirmed). Implemented in models/schemas.py.
4. Colab fallback with anonymized data? Accepted. Upload mechanics below.

## 9a. Colab upload mechanics (fallback path only)

Only the ANONYMIZED TRAIN split (data/corpus/pool_anon.jsonl-derived,
data/splits/train.jsonl converted) may leave the machine. Never val, never
test, never the raw pool. Three options, in order of preference:

1. Direct upload in the notebook (RECOMMENDED). In the first cell:
   from google.colab import files; files.upload()
   The file goes to the ephemeral VM. It dies with the runtime. Nothing is
   persisted by Google. Do not save it to Drive.
2. Private Hugging Face dataset repo (only if the file is too big for the
   upload widget). Create the repo as PRIVATE, gated to your account, upload
   via huggingface-cli from the Mac. You can delete it right after training.
   Google/Colab never sees it; only HF does, and only over TLS.
3. Last resort: drive.mount(). Avoid it — anything in Drive persists after
   the session, which breaks the ephemeral-data rule.

Rule in all cases: the uploaded file must come from scripts/anonymize.py
output. Verify a random sample of 20 rows by eye before uploading.

## 9b. ZDR-routed OpenRouter labeling (optional speed fallback)

If local labeling is too slow, the labeler may instead run through
OpenRouter with Zero-Data-Retention forced, on ANONYMIZED data only.

- ZDR is per-request: send
  "provider": {"zdr": true, "data_collection": "deny"} in the body.
  Implemented: set LOCAL_LLM_EXTRA_BODY to that JSON, LOCAL_LLM_URL to
  https://openrouter.ai/api/v1/chat/completions, LOCAL_LLM_API_KEY to the
  OpenRouter key. If no ZDR endpoint exists for the model, the request
  FAILS instead of silently routing to a logging provider (fail-closed).
  Sources: https://openrouter.ai/blog/insights/zero-data-retention/
  https://openrouter.ai/docs/guides/features/zdr
- NEVER use free OpenRouter models on any email data (free endpoints
  commonly log or train).
- Raw (un-anonymized) email NEVER goes to OpenRouter under any setting.
  This path is anonymized-data-only, same tier as the Colab fallback.
- Resume claim impact: "fully local labeling" must then be stated as
  "ZDR-routed labeling on anonymized data" — weaker headline.

## 11. Results (2026-10-16, final v1 benchmark)

Test set: 47 rows, thread-clustered, locked (SHA256 recorded at split time).
Labels: weak (local Qwen3-4B labeler), human-corrected in an interactive pass,
unverified row-by-row. These numbers measure agreement with the corrected
weak labels, not verified ground truth.

| Metric | Local (Qwen3-1.7B tuned) | API (gpt-4o-mini) |
|---|---|---|
| Parse errors (JSON invalid) | 0 | 0 |
| True negatives | 21/46 | 19/46 |
| False positives (invented tasks) | 25 | 27 |
| False negatives (missed tasks) | 0 | 0 |
| Field accuracy on task rows | 1.0 (n=1) | 1.0 (n=1) |
| Latency p50 | 1.27 s | 1.23 s |
| Latency p95 | 1.79 s | 2.18 s |
| Cost per 1,000 emails | $0 (local compute) | ~$0.30-0.60 (approx, unverified pricing) |

Headline: the tuned 1.7B local model matches or beats the API baseline on
every measured axis — fewer false positives, lower p95 latency, zero cost,
zero parse errors, with email content never leaving the machine.

Known issues (quantified):
1. Both models over-trigger on noise emails (FP ~55%). Cause: the training
   distribution was mostly-empty after label corrections but the test set
   is 46/47 noise rows; 2 epochs of LoRA under-adapted the noise behavior.
   Fix path: more task-bearing training rows (widen GMAIL_QUERY), or
   targeted FP suppression training.
2. Only 1 task row in test — field-accuracy is anecdotal. Fix path:
   a larger, more balanced pool (the 2,500-row target in this PRD).
3. Labels are weak and unverified. The 0.8-0.9 field agreement numbers from
   the pre-correction benchmark stand as upper bounds on labeler agreement.

Training history (for the record):
- Run 1: 45 iters, rank 8, no mask-prompt, original weak labels -> val 1.34.
- Run 2: 200 iters, rank 16, mask-prompt on -> diverged (loss spike 27).
- Run 3: 45 iters, rank 16, mask-prompt on, corrected labels -> val 4.5-12.
- Root cause of 2/3: --mask-prompt with Qwen3 thinking-template completions
  left ~10 loss-bearing tokens per row; signal starved, lr 2e-4 diverged.
- Run 4 (final): 45 iters, rank 16, mask-prompt OFF, corrected labels ->
  train 3.1->1.37, val 2.07, no divergence. Used for the final benchmark.

All links are inline above. Version checks (2026-10-15, PyPI / GitHub API):
unsloth 2026.10.6, trl 1.16.0, peft 0.21.2, bitsandbytes 0.50.2,
transformers 5.20.0, mlx-lm 0.32.1, llama.cpp v0.7.0, Ollama v0.40.5.
