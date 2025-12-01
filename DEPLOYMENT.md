# Deployment Guide - API Keys Configuration

## Setting Up API Keys in Streamlit Cloud

Since `keys.json` is excluded from git for security, you need to set API keys as **environment variables** in your Streamlit Cloud deployment.

### Step 1: Access Streamlit Cloud Settings

1. Go to your Streamlit Cloud dashboard: https://share.streamlit.io/
2. Select your app: `capstone-dashboard`
3. Click on **"⚙️ Settings"** (or "Manage app" → "Settings")

### Step 2: Add Environment Variables

In the **"Secrets"** section, add the following environment variables:

#### Required Environment Variables:

```
OPENAI_API_KEY=sk-proj-your-actual-openai-key-here
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-anthropic-key-here
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key-here
GEMINI_API_KEY=your-actual-gemini-key-here
GOOGLE_API_KEY=your-actual-google-key-here
```

**OR** you can use the TOML format in the secrets editor:

```toml
[secrets]
OPENAI_API_KEY = "sk-proj-your-actual-openai-key-here"
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-anthropic-key-here"
DEEPSEEK_API_KEY = "sk-your-actual-deepseek-key-here"
GEMINI_API_KEY = "your-actual-gemini-key-here"
GOOGLE_API_KEY = "your-actual-google-key-here"
```

### Step 3: Get New API Keys

Since your previous keys were exposed in git and revoked:

#### OpenAI (GPT)
1. Go to: https://platform.openai.com/account/api-keys
2. Click **"Create new secret key"**
3. Copy the key (starts with `sk-proj-` or `sk-`)
4. Set as `OPENAI_API_KEY`

#### Anthropic (Claude)
1. Go to: https://console.anthropic.com/settings/keys
2. Click **"Create Key"**
3. Copy the key (starts with `sk-ant-api03-`)
4. Set as `ANTHROPIC_API_KEY`

#### DeepSeek
1. Go to: https://platform.deepseek.com/api_keys
2. Create a new API key
3. Set as `DEEPSEEK_API_KEY`

#### Google (Gemini)
1. Go to: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Set as `GEMINI_API_KEY` (or `GOOGLE_API_KEY` - both work)

### Step 4: Redeploy

After adding the environment variables:
1. Click **"Save"** in the settings
2. Streamlit Cloud will automatically redeploy your app
3. Check the logs to verify the API keys are being read correctly

### Verification

After deployment, check the logs. You should see:
```
API keys status: {'OPENAI_API_KEY': '✓', 'ANTHROPIC_API_KEY': '✓', ...}
```

If you see `✗` for any key, that means the environment variable is not set correctly.

## How the Code Works

The code checks for API keys in this order:
1. **First**: Tries to load from `data/keys.json` (if file exists locally)
2. **Then**: Falls back to environment variables (for deployment)

The providers automatically read from environment variables:
- **GPT**: `OpenAI()` SDK reads from `OPENAI_API_KEY`
- **Claude**: Explicitly reads `ANTHROPIC_API_KEY`
- **DeepSeek**: Reads `DEEPSEEK_API_KEY`
- **Gemini**: Reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Troubleshooting

### Error: "401 - Incorrect API key"
- The API key is invalid or expired
- Generate a new key from the provider's dashboard
- Make sure you copied the entire key (no spaces, no line breaks)

### Error: "403 - API key was reported as leaked"
- The key was exposed publicly (like in git)
- Generate a completely new API key
- Never commit API keys to git again

### Error: "invalid x-api-key" (Claude)
- Check that `ANTHROPIC_API_KEY` is set correctly
- Make sure there are no extra spaces or quotes
- Verify the key starts with `sk-ant-api03-`

### Keys not working after setting environment variables
- Wait a few minutes for Streamlit Cloud to redeploy
- Check the deployment logs for the "API keys status" message
- Verify the variable names match exactly (case-sensitive!)

