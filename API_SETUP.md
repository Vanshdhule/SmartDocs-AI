# OpenAI API Setup Guide 🔑

This guide walks you through obtaining an OpenAI API key and configuring it for SmartDocs AI.

---

## Step 1: Create an OpenAI Account

1. Go to [https://platform.openai.com](https://platform.openai.com)
2. Click **Sign Up** (or **Log In** if you already have an account)
3. Complete the registration process (email verification required)

---

## Step 2: Add a Payment Method (Required for API Access)

1. Once logged in, navigate to **Settings → Billing**
2. Click **Add payment method**
3. Enter your credit/debit card details
4. Set a **usage limit** (recommended: $5–$10 for development)

> **Note:** OpenAI requires a payment method even when using free credits.
> New accounts may receive free trial credits ($5 USD as of 2024).

---

## Step 3: Generate an API Key

1. Navigate to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **+ Create new secret key**
3. Give it a name, e.g. `SmartDocs-AI-Dev`
4. Click **Create secret key**
5. **Copy the key immediately** — it will NOT be shown again!

Your key will look like:

```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Step 4: Configure the Key in Your Project

### Option A: Using the `.env` file (Recommended)

1. In the project root directory, copy the template:

   ```bash
   cp .env.template .env
   ```

2. Open `.env` in a text editor and replace the placeholder:

   ```env
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

3. Save the file. **Do not commit this file to Git.**

### Option B: Set as a System Environment Variable (Alternative)

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = "sk-proj-your-actual-key-here"
```

**macOS / Linux:**

```bash
export OPENAI_API_KEY="sk-proj-your-actual-key-here"
```

---

## Step 5: Verify the Connection

Run the test script from the project root (with venv active):

```bash
python test_openai.py
```

Expected output:

```
🔑 API Key loaded successfully
✅ OpenAI API connection successful!

🤖 OpenAI Response:
Yes, I'm working! How can I assist you today?
```

---

## 🚨 Common Errors & Solutions

| Error                       | Cause                              | Solution                                    |
| --------------------------- | ---------------------------------- | ------------------------------------------- |
| `OPENAI_API_KEY not found`  | `.env` file missing or key not set | Create `.env` from `.env.template`          |
| `❌ Invalid OpenAI API key` | Wrong or expired key               | Regenerate a new key at platform.openai.com |
| `⚠️ Rate limit exceeded`    | Too many requests                  | Wait 60 seconds and retry                   |
| `💳 Insufficient credits`   | No balance on account              | Add payment method / purchase credits       |
| `🌐 Network error`          | No internet connection             | Check your internet connection              |

---

## 🔒 Security Best Practices

- **Never share your API key** publicly, in code, or in screenshots
- **Never commit `.env`** to version control (it is already in `.gitignore`)
- **Set usage limits** in the OpenAI dashboard to avoid unexpected charges
- **Rotate your key** immediately if you suspect it has been compromised

---

## 💰 Pricing Reference (as of 2024)

| Model                    | Input                | Output              |
| ------------------------ | -------------------- | ------------------- |
| `gpt-3.5-turbo`          | $0.0015 / 1K tokens  | $0.002 / 1K tokens  |
| `gpt-4o-mini`            | $0.00015 / 1K tokens | $0.0006 / 1K tokens |
| `text-embedding-3-small` | $0.00002 / 1K tokens | —                   |

> Check [https://openai.com/pricing](https://openai.com/pricing) for the latest pricing.

---

## 📌 Model Used in This Project

This project defaults to **`gpt-3.5-turbo`** for Q&A responses.
You can change the model in `backend/openai_helper.py`.
