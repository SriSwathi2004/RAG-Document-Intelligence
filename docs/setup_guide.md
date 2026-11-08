# Setup guide
See README.md. Short version (Windows PowerShell):

    py -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python scripts/initialize_project.py
    streamlit run app/streamlit_app.py

Local LLM (local machine only, not Colab): install Ollama from ollama.com, then `ollama pull llama3.2:3b`
(or `llama3.2:1b` for low RAM). Without Ollama the app falls back to extractive answers.
Google Colab: Ollama is awkward there; use `llm.provider: extractive` or run Ollama locally. Streamlit needs a tunnel in Colab.
