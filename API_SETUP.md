# NVIDIA NIM API Setup Guide 🔑

This guide walks you through obtaining NVIDIA NIM API keys and configuring
them for SmartDocs AI. NVIDIA NIM is **completely free** — no credit card required.

---

## Why NVIDIA NIM?

SmartDocs AI uses two separate NVIDIA NIM API keys:

| Key | Used By | Purpose |
|---|---|---|
| `NVIDIA_CHAT_API_KEY` | `qa_engine.py` | LLM chat completions (Llama 3.3 70B + Nemotron 70B) |
| `NVIDIA_EMBEDDING_API_KEY` | `embeddings.py` | Document + query embeddings (nv-embedqa-e5-v5) |

Both keys are free with a NVIDIA developer account.

---

## Step 1: Create a NVIDIA Developer Account

1. Go to [https://build.nvidia.com](https://build.nvidia.com)
2. Click **Sign In** (top right)
3. Click **Create Account** if you don't have one
4. Register with your email — no credit card required

---

## Step 2: Get Your Chat API Key (for LLM calls)

1. Once logged in, go to [https://build.nvidia.com](https://build.nvidia.com)
2. Search for **"llama-3.3-70b-instruct"** or browse **NIM Catalog**
3. Click on the model → click **Get API Key**
4. Click **Generate Key**
5. **Copy the key immediately** — it starts with `nvapi-`

```
nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> This key is used for all LLM chat completions in `qa_engine.py`.

---

## Step 3: Get Your Embedding API Key (for embeddings)

You can use the **same key** for both, or generate a separate one:

1. Go to [https://build.nvidia.com](https://build.nvidia.com)
2. Search for **"nv-embedqa-e5-v5"**
3. Click on the model → click **Get API Key**
4. Click **Generate Key** and copy

> **Tip:** Using separate keys lets you track usage independently
> and avoids hitting rate limits on one key for both tasks.

---

## Step 4: Configure Keys in Your Project

### Option A: Using the `.env` file (Recommended)

1. Copy the template from project root:

```bash
cp .env.template .env
```

2. Open `.env` and fill in both keys:

```env
NVIDIA_CHAT_API_KEY=nvapi-your-chat-key-here
NVIDIA_EMBEDDING_API_KEY=nvapi-your-embedding-key-here
```

3. Save the file. **Never commit `.env` to Git** — it is already in `.gitignore`.

### Option B: System Environment Variable

**Windows (PowerShell):**
```powershell
$env:NVIDIA_CHAT_API_KEY = "nvapi-your-chat-key-here"
$env:NVIDIA_EMBEDDING_API_KEY = "nvapi-your-embedding-key-here"
```

**macOS / Linux:**
```bash
export NVIDIA_CHAT_API_KEY="nvapi-your-chat-key-here"
export NVIDIA_EMBEDDING_API_KEY="nvapi-your-embedding-key-here"
```

---

## Step 5: Verify the Connection

Run the NVIDIA smoke test from the project root (with venv active):

```bash
python tests/test_nvidia.py
```

Expected output:

```
Testing chat model (meta/llama-3.3-70b-instruct)...
  OK  Chat model working: OK

Testing embedding model (nvidia/nv-embedqa-e5-v5)...
  OK  Embedding model working: dimension = 1024

Done.
```

If both lines show `OK` — you are fully configured.

---

## 🚨 Common Errors & Solutions

| Error | Cause | Solution |
|---|---|---|
| `NVIDIA_CHAT_API_KEY not found` | `.env` file missing or key not set | Copy `.env.template` → `.env` and fill in keys |
| `NVIDIA_EMBEDDING_API_KEY not found` | Embedding key missing | Add `NVIDIA_EMBEDDING_API_KEY` to `.env` |
| `❌ Invalid NVIDIA API key` | Wrong or expired key | Regenerate at build.nvidia.com |
| `⚠️ Rate limit exceeded` | Too many requests | Wait 60 seconds and retry |
| `🌐 Network error` | No internet connection | Check your internet connection |
| `dimension = 768` (wrong) | Wrong embedding model | Ensure model is `nvidia/nv-embedqa-e5-v5` in `config.py` |

---

## 💰 Pricing

NVIDIA NIM is **free for developers** on the hosted API tier.

| Model | Cost |
|---|---|
| `meta/llama-3.3-70b-instruct` | Free (rate-limited) |
| `nvidia/llama-3.1-nemotron-70b-instruct` | Free (rate-limited) |
| `nvidia/nv-embedqa-e5-v5` | Free (rate-limited) |

> Check [https://build.nvidia.com](https://build.nvidia.com) for current
> rate limits and any updates to the free tier.

---

## 🔒 Security Best Practices

- **Never share your API keys** publicly, in code, or in screenshots
- **Never commit `.env`** to version control (already in `.gitignore`)
- **Use separate keys** for chat and embeddings to isolate rate limits
- **Rotate your key** immediately if you suspect it has been compromised
- Run `git status` before every push and confirm `.env` is NOT listed

---

## 📌 Models Used in This Project

| Purpose | Model | Key Used |
|---|---|---|
| Primary LLM | `meta/llama-3.3-70b-instruct` | `NVIDIA_CHAT_API_KEY` |
| Secondary LLM (fallback) | `nvidia/llama-3.1-nemotron-70b-instruct` | `NVIDIA_CHAT_API_KEY` |
| Embeddings | `nvidia/nv-embedqa-e5-v5` (1024-dim) | `NVIDIA_EMBEDDING_API_KEY` |

All models are configured in `config.py`. To change a model, update the
`PRIMARY_MODEL`, `SECONDARY_MODEL`, or `EMBEDDING_MODEL` constants there.
