# Reference Log

External tools, libraries and resources used
- OpenAI API (via Cornell proxy): model endpoints for embeddings and chat
- LangChain (langchain-core / langchain-openai) — orchestration and prompt wiring
- Chroma (chromadb / langchain-chroma) — vector store for embeddings
- Streamlit — web UI for upload and chat interface
- pypdf — PDF parsing
- Python standard library (pathlib, os, re, logging)

Local files and config changed
- `chat_with_pdf.py` — main application. Changes made to ensure stable behavior in the Codespace environment:
	- Simplified embedding model selection and set defaults to `openai.text-embedding-ada.002`.
	- Sanitized and timestamped uploaded filenames via `save_uploads`.
	- Removed deprecated `persist()` usage for Chroma and adapted retriever usage.
	- Fixed retriever API call to work with installed LangChain (uses `invoke` where appropriate).
	- Formatted assistant responses for clearer display.
- `.devcontainer/devcontainer.json` — updated `OPENAI_MODEL` and `OPENAI_EMBED_MODEL` defaults to provider-prefixed model IDs (e.g., `openai.gpt-5`, `openai.text-embedding-ada.002`) so the Codespace environment uses correct model names for the Cornell proxy.

GenAI usage (this assignment)
- I used an AI assistant during development to:
	- Review and suggest fixes to `chat_with_pdf.py` (upload handling, Chroma usage, retriever API, model naming).
	- Draft user-facing README content and troubleshooting steps.
	- Propose small refactors to make the app simpler and more robust.

Rationale for GenAI use
- The assistant accelerated debugging of integration issues (invalid model names, LangChain/Chroma API mismatches) and helped generate clear documentation for running the app in Codespaces. All suggestions were reviewed and tested locally before being applied.

External references consulted
- Streamlit docs: https://docs.streamlit.io
- Chroma docs / TryChroma telemetry messages
- OpenAI API docs (model listing and embeddings): https://platform.openai.com/docs

If any external package updates are required, please see `requirements.txt` and re-run `pip install -r requirements.txt`.

Date: 2025-11-08
