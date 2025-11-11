# INFO 5940 
Welcome to the INFO 5940 repository. You will complete your work using [**GitHub Codespaces**](#about-github-codespaces) and save your progress in your own GitHub repository. This guide will walk you through setting up the development environment and running the test notebook.  

## Getting Started 

### Step 1: Fork this repository 
1. Click the **Fork** button (top right of this page).
2. This will create a copy of the repo under **your own GitHub account**.

Forking creates a personal copy of the repo under **your** GitHub account.  
- You can commit, push, and experiment freely.  
- Your work stays separate from the official class materials.

### Step 2: Open your forked repo Codespace
1. Go to **your forked repo**.
2. Click the green **Code** button and switch to the **Codespaces** tab.  
3. Select **Create Codespace**.
4. Wait a few minutes for the environment to finish setting up.

### Step 3: Verify your environment 
Once the Codespace is ready: 
1. If you are in `<your-file-name>.ipynb` in your codespace.
2. Install the Python 3.11.13 Kernel.  In the top-right corner, click **Select Kernel**.
    1. If **Install/Enable suggested extensions Python + Jupyter** appears, select it, and wait for the install to finish before moving on to the next step.
    2. Select **Python Environments** choose **Python 3.11.13 (first option)**.
3. Run the code block to check your setup. 

## About GitHub Codespaces

[Codespaces](https://docs.github.com/en/codespaces) is a complete software development and execution environment, running in the cloud, with its primary interface being a VSCode instance running in your browser.

Codespaces is not free, but their per-month [free quota](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces#free-quota) is generous.  Codespaces is free under the [GitHub Student Developer Pack](https://education.github.com/pack#github-codespaces).

### Codespaces Tips

* Codespaces keep running even when you close your browser (but will time out and stop after a while)
* Unless you're on a free plan, or within your free quota, costs acrue while the codespace is running, whether or not you have it open in your browser or are working on it
* You can control when it's running, and the space it takes up.  Check out [GitHub's codespaces lifecycle documentation](https://docs.github.com/en/codespaces/about-codespaces/understanding-the-codespace-lifecycle)

## Sync Updates 
To make sure your personal forked repository stays up to date with the original class repository, please follow these steps:
1. Open your forked repo.
2. At the top of the page, you should see a banner or menu option that shows whether your fork is behind the original repo.
3. Click the **Sync fork** button.
4. In the dropdown, choose **Update branch** to pull the latest changes from the original repo into your fork.

Optionally, you can also follow these steps to create a new branch on your fork:
1. Open your **forked repository** on GitHub.  
2. At the top of the page, next to the branch dropdown, click the **Branches** button.  
3. In the **Branches** view, click the green **New Branch** button.  
4. In the popup window, enter a branch name.  
   - You can use any name you like, but it’s recommended to match the branch name used in class for better organization.  
```markdown
# INFO 5940 — Assignment 1
This repository contains a Streamlit Retrieval-Augmented Generation (RAG) app that lets users upload `.txt` and `.pdf` documents, indexes them with Chroma, and asks a conversational LLM questions grounded in those documents.

## Quick start (run the app)

1. Open the Codespace or workspace and ensure dependencies are installed (the repository includes `requirements.txt`).

```bash
# (optional) create a venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the app
streamlit run chat_with_pdf.py
```

2. In Codespaces, after `streamlit` starts you'll see an **Open in Browser** link — use it to interact with the UI.

3. Upload files via the "Add documents" section (supports `.txt` and `.pdf`), click **Ingest files**, then ask questions in the chat box.

## Features

- Upload and ingest `.txt` and `.pdf` files (multiple files allowed).
- Documents are chunked (default chunk size 800 chars, overlap 120) and indexed in Chroma.
- Retrieval uses top-K (default 4) with optional MMR.
- Conversational chat interface via Streamlit that returns grounded answers and lists source documents/pages.
- Safe upload handling (sanitizes filenames and avoids accidental overwrites by adding a timestamp suffix when needed).

## Configuration & Environment Variables

The app reads the following environment variables (you can set them in your Codespace or export them locally):

- `API_KEY` or `OPENAI_API_KEY` — OpenAI/Cornell proxy API key
- `BASE_URL` or `OPENAI_BASE_URL` — Base URL for the API (example: `https://api.ai.it.cornell.edu/`)
- `OPENAI_MODEL` — Chat/completion model (default: `openai.gpt-5`)
- `OPENAI_EMBED_MODEL` — Embedding model (default: `openai.text-embedding-ada.002`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `USE_MMR` — numeric runtime defaults

The devcontainer sets these values in `.devcontainer/devcontainer.json` for the Codespace environment. Example export lines if you prefer to set them in your shell:

```bash
export API_KEY="<your_key>"
export OPENAI_API_KEY="$API_KEY"
export BASE_URL="https://api.ai.it.cornell.edu/"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_MODEL="openai.gpt-5"
export OPENAI_EMBED_MODEL="openai.text-embedding-ada.002"
```

## Notable implementation details / changes from template

- Simplified embedding/model selection to use fixed model names compatible with the Cornell proxy.
- Removed a complex embedding discovery flow to keep the app deterministic in the course environment.
- Fixed upload handling: `save_uploads` now sanitizes filenames and prevents path traversal.
- Updated Chroma usage: removed a deprecated `persist()` call and adapted retriever usage for the installed LangChain/Chroma versions.
- Adjusted chat response display to clearly label the assistant's answer.

## File structure (relevant)

- `chat_with_pdf.py` — main Streamlit app (upload, ingest, chat UI).
- `data/uploads/` — saved uploaded files (created at runtime).
- `data/chroma/` — Chroma persistence directory (created at runtime).
- `requirements.txt` — Python dependencies.
- `.devcontainer/devcontainer.json` — Codespace environment variables (API keys and model defaults).

## Troubleshooting

- If you get an "Invalid model name" 400 error, run the following quick check to list available models for your key:

```bash
python3 - <<'PY'
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('API_KEY'), base_url=os.getenv('BASE_URL'))
for m in client.models.list().data:
    print(m.id)
PY
```

- If Chroma raises a readonly DB error, delete or reset `data/chroma/` (you can clear vectors from the sidebar). Ensure the current user has write permissions to the `data/` directory.

## How this fulfills the assignment criteria

- Uses the provided Codespace/devcontainer configuration.
- Supports `.txt` uploads, `.pdf` parsing, multiple documents, chunking, retrieval, and conversational UI.
- Uses Chroma for vector storage and LangChain/OpenAI for LLM generation.

---
Please see `ref-log.md` for a record of tools and external sources used while implementing this assignment.
```


## Troubleshooting
- The Jupyter extension should install automatically. If you still cannot select a Python kernel on Jupyter Notebook: Go to the left sidebar >> **Extensions** >> search for **Jupyter** >> reload window (or reinstall it).   
