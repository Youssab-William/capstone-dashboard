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

### Step 3: Set Up GitHub Token for Data Persistence

To persist run data across deployments, you need to set up a GitHub Personal Access Token:

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name like "Streamlit Cloud Data Persistence"
4. Select scopes: **`repo`** (full control of private repositories)
5. Click **"Generate token"**
6. Copy the token (starts with `ghp_`)

Then add it to Streamlit Cloud secrets:

```toml
[secrets]
GITHUB_TOKEN = "ghp_your-token-here"
GITHUB_REPO_OWNER = "Youssab-William"  # Your GitHub username
GITHUB_REPO_NAME = "capstone-dashboard"  # Your repository name
```

**Note:** The `GITHUB_REPO_OWNER` and `GITHUB_REPO_NAME` are optional - the code will try to auto-detect them from git, but you can set them explicitly if needed.

### Step 4: Get New API Keys

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

### Step 5: Redeploy

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

## How Data Persistence Works

### The Problem
Streamlit Cloud uses an **ephemeral filesystem** - any files written during runtime are lost when:
- The app shuts down due to inactivity
- The app is redeployed
- The container restarts

### The Solution
The app now automatically commits run data to GitHub after each run completes. This means:
- ✅ Run data persists across deployments
- ✅ Data is versioned in git
- ✅ Multiple deployments can share the same data
- ✅ Data survives app restarts

### What Gets Committed
After each run completes, the following files are automatically committed to GitHub:
- `data/completions/{run_id}.jsonl` - LLM responses
- `data/metrics/{run_id}.jsonl` - Computed metrics
- `data/analysis/{run_id}.jsonl` - Analysis results
- `data/logs/run_{run_id}.json` - Run progress metadata

### How It Works
1. When you start a new run, data is written to the local filesystem
2. After the run completes successfully, the app commits the data files to GitHub
3. On the next deployment/restart, the app reads data from the GitHub repository
4. The dashboard displays all runs that exist in the repository

### Troubleshooting Data Persistence

#### Data not appearing after deployment
- Check that `GITHUB_TOKEN` is set correctly
- Verify the token has `repo` scope permissions
- Check the logs for GitHub commit errors
- Ensure the repository name matches your actual repo

#### "GITHUB_TOKEN not set" warning
- This is normal if you haven't set up the token yet
- Runs will still work, but data won't persist across deployments
- Set up the token as described above to enable persistence

#### GitHub commit fails
- Check that the token is valid and not expired
- Verify the repository owner/name are correct
- Check GitHub's rate limits (you shouldn't hit them with normal usage)
- The run will still complete successfully even if the commit fails

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
