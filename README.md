# ARQ Voice AI Agent

Prototype of a Marathi-first voice-enabled RAG system built with Whisper, Ollama, and Chroma.

## Code flow (current state)
- CLI voice -> RAG (`voice_rag_mic_to_rag.py`): records 6-second mono audio via `sounddevice`, writes `mic_input.wav`, transcribes with `faster-whisper` (CPU int8), runs LangChain `RetrievalQA` backed by the Chroma store and Ollama model from `settings.ollama_model`, prints a Marathi answer. Requires the knowledge base to be built first.
- STT smoke test (`voice_rag/stt_mic_test.py`): same recording pipeline but only prints the Whisper transcript (Marathi forced), useful for checking the mic and STT before running RAG.
- Streamlit text chat (`voice_rag/app_streamlit.py`): web UI to upload PDFs (saved under `voice_rag/data/pdfs`), build or refresh the vector store, and chat in Marathi/English (answers in Marathi) using RetrievalQA; keeps a simple chat history in session; right column shows KB status/debug info and a quick Chroma load test.
- RAG backend (`voice_rag/rag/*`): `ingest.py` loads PDFs with `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter`, embeds with HuggingFace multilingual MiniLM, and stores or loads Chroma at `voice_rag/data/chroma_store`; `qa.py` builds a `ChatOllama` + `RetrievalQA` chain with a Marathi-focused prompt; other files are placeholders for prompt, embedding, and retriever customization.
- Config and paths (`voice_rag/config/settings.py`): defines project root, data directories, ensures `voice_rag/data/{pdfs,chroma_store}` exist, and sets embedding plus Ollama model names.
- LangGraph scaffolding (`voice_rag/graph/*`): placeholders for future graph nodes (transcription, language router, retrieval, answer, TTS) and graph builder/state wiring.
- Tests (`voice_rag/tests/*`): `test_ollama_chat.py` sanity-checks the local Ollama endpoint; `test_end_to_end.py` is a manual harness to build a KB from `voice_rag/data/pdfs/test.pdf` and ask a sample Marathi question once embeddings and Chroma are available.
- Data and assets: `voice_rag/data` holds user PDFs and the persisted Chroma DB; `mic_input.wav` at repo root and under `voice_rag/` are sample recordings left by recent runs.

## Running it
- Install deps: `pip install -r requirements.txt` (Whisper downloads weights on first run; pull the Ollama model you want, default is `llama3`).
- Build the knowledge base:
  - Streamlit path: `streamlit run voice_rag/app_streamlit.py`, upload PDFs, click Build/Refresh (stores vectors in `voice_rag/data/chroma_store`).
  - CLI path: place PDFs in `voice_rag/data/pdfs` and call `build_vector_store_from_pdfs` from `voice_rag.rag.ingest` in a short script (the Streamlit flow shows the sequence).
- Chat by text: use the Streamlit app after the KB is ready.
- Voice to RAG (terminal): ensure KB exists, then run `python voice_rag_mic_to_rag.py` and speak when prompted.
- STT only: run `python voice_rag/stt_mic_test.py`.

## Folder map
- `voice_rag/app_streamlit.py` - Streamlit UI.
- `voice_rag_mic_to_rag.py`, `voice_rag/stt_mic_test.py` - CLI record and transcribe flows.
- `voice_rag/config/` - settings and data directory creation.
- `voice_rag/rag/` - ingest and QA pipeline (Chroma + Ollama) plus placeholders for prompt or embedding tweaks.
- `voice_rag/graph/` - LangGraph scaffolding for the future voice pipeline.
- `voice_rag/data/` - `pdfs` (input docs), `chroma_store` (persisted vector DB), `audio` (scratch recordings).
- `voice_rag/tests/` - manual smoke scripts for Ollama and end-to-end KB/QA.
