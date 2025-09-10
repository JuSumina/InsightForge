# InsightForge — AI-Powered Business Intelligence Assistant

Turn your raw sales data into clear, actionable insights. InsightForge combines pandas analytics, interactive Plotly charts, and an LLM + RAG pipeline (LangChain + OpenAI) with conversational memory in a clean Streamlit UI.


## ✨ Highlights

Plug-and-play data: Upload a CSV or use generated sample data.

Rich analytics: Sales trends, product & regional breakdowns, demographics, statistics.

Interactive visuals: Plotly charts for trends, distributions, correlations.

LLM insights (RAG): Natural-language Q&A over your data using a Chroma vector store (persistent) with OpenAI embeddings.

Executive summaries: One-click, C-suite-ready writeups.

Memory: Chat retains conversation history for follow-ups and context-aware answers.

Model eval: Quick smoke tests with QAEvalChain (when available).


## 🧱 Architecture
```
InsightForge/
├─ main.py                  # Streamlit entrypoint
├─ config.py                # Central config (UI, model, paths, etc.)
├─ data_handler.py          # Load CSV/generate sample data + compute analytics
├─ visualizations.py        # Plotly figures
├─ ai_system.py             # RAG system (Chroma + OpenAI), prompts, memory, eval
├─ ui_components.py         # Streamlit UI: sidebar, charts, insights, chat, eval
├─ sales_data.csv           # (optional) sample/input data
├─ .env                     # (not committed) holds OPENAI_API_KEY
└─ chroma_db/               # (git-ignored) persistent Chroma vector store
```

**Key flows**

**Data pipeline (data_handler.py)**

Loads CSV (or generates realistic sample data).

Standardizes columns, derives features (month/quarter, age buckets, etc.).

Produces rich aggregates: sales performance, product/region analysis, demographics, satisfaction, temporal patterns, statistics.

Note: Period indexes (Month/Quarter) are converted to str to avoid JSON serialization issues.

**RAG pipeline (ai_system.py)**

Converts processed analytics to documents.

Splits into chunks and embeds with OpenAI Embeddings (text-embedding-3-small).

Stores in Chroma (persistent) at ./chroma_db (or fallback to SKLearnVectorStore if configured).

ConversationalRetrievalChain + ConversationBufferMemory enables chat with memory.

Executive summaries via LLMChain + prompt templates.

**UI (ui_components.py)**

Sidebar: data source & display toggles.

Dashboard: KPIs + interactive charts.

AI-Powered Insights: Executive summary, model evaluation, and a chat pane with predefined questions + follow-ups.

Session state stores the RAG instance and chat history across interactions.


## 🛠 Requirements

Recommended Python: 3.10–3.11 (3.13 is bleeding-edge and some libs may lag)

OpenAI API key (for embeddings + model)

requirements.txt (example):

streamlit==1.32.2
pandas==2.2.2
numpy==1.26.4
plotly==5.22.0
langchain>=0.3.0,<0.4
langchain-community>=0.3.0,<0.4
langchain-openai>=0.2.0,<0.3
openai>=1.44.0
tiktoken==0.7.0
python-dotenv==1.0.1
scikit-learn==1.5.2
langchain-text-splitters>=0.3.0,<0.4
regex>=2024.5.15
pyarrow==16.1.0
chromadb>=0.5.5


Why Chroma? We default to Chroma for vector search to avoid FAISS install headaches on Windows. Chroma persists locally and works out of the box with OpenAI embeddings.


## 🚀 Quickstart
1) Clone
git clone https://github.com/<your-username>/InsightForge
cd insightforge

2) Create & activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate


(macOS/Linux)

python3 -m venv .venv
source .venv/bin/activate

3) Install deps
pip install -r requirements.txt

4) Add your OpenAI key

Create a .env file in the project root:

OPENAI_API_KEY=sk-********************************

5) Run the app
streamlit run main.py


Open the URL Streamlit prints (usually http://localhost:8501)


## ⚙️ Configuration

**config.py centralizes settings:**

**OPENAI_API_KEY:** loaded from .env via python-dotenv.

**Model & generation:**

model_name = "gpt-4o-mini"

llm_temperature = 0.1

max_tokens = 2000

**Embeddings & RAG:**

chunk_size = 1000

chunk_overlap = 200

chroma_path = "./chroma_db" (persistent vector store directory)

**Charts:** chart_height, color_palette

**Sample data controls:** sample_data_size, demo categories.


## 🖥 Using InsightForge

Choose data source

Upload your sales_data.csv (columns like Date, Product, Region, Sales, Customer_Age, Customer_Gender, Customer_Satisfaction)

or use the Sample Data generator.

Explore the dashboard

KPIs: total records, total sales, average sale, product count.

Visuals: time trends, product & region performance, demographics, satisfaction, correlation heatmap.

AI-Powered Insights

Executive Summary: click to generate a structured, C-level overview.

Model Evaluation: run quick checks over a few predefined questions.

Chat with Memory:

Pick a predefined question and click Use Selected, or type your own.

The assistant remembers prior Q&A to enable context-aware follow ups.


## 🧠 Prompts & Memory

**Analysis Prompt guides the model to:**

Use retrieved context + prior chat history

Produce well-formatted insights (with spacing rules to avoid “1.38million” issues)

Provide trends, recommendations, and risks

**Executive Summary Prompt:**

Delivers an exec-friendly structure with bullet points and clean formatting.

**Memory:**

ConversationBufferMemory keeps chat_history so responses become more relevant across turns.


## 🧪 Model Evaluation

Optional QAEvalChain support (depends on your LangChain build).

A small harness runs 2–3 test questions and reports a simple success rate—good for quick health checks.


## 🧩 Troubleshooting

“OpenAI API key is not set…”
Make sure .env exists and your venv is activated:

Run python -c "import os; print(os.getenv('OPENAI_API_KEY'))" → should print a key.

Restart Streamlit after changes.

“Could not import faiss…”
We don’t use FAISS by default. The code uses Chroma. Ensure:

chromadb is installed

You’re not forcing a FAISS backend elsewhere.

Chroma “onnxruntime not installed”
If Chroma tries to auto-select local embedding models, install:

pip install onnxruntime


(Not required if you pass OpenAIEmbeddings as in this project.)

Weird spacing like 1.38million
The prompts and a lightweight normalization step reduce this. If you paste model output manually elsewhere, formatting may vary—prefer the in-app rendering.

LangChain deprecation: Chain.__call__
We use .invoke({...}) on chains (e.g., conversational_chain.invoke({"question": ...})). If you add new chains, avoid .run() or direct calls.

Wrong environment
Most “module not found” errors come from installing packages in one shell/venv and running in another. Always:

Activate venv before installing and running.

which python / where python to confirm you’re using the venv interpreter.

Resetting the vector store
Delete the chroma_db/ folder (while the app is stopped) to rebuild from scratch.


## 📈 Roadmap

Forecasting + scenario analysis (and render predicted lines on charts)

Parameter controls for retrieval (k, similarity threshold) in the UI

Better long-term memory (summarization, persistence)

Multi-file/document ingestion

Role-based prompts (analyst vs exec vs marketer)

Exportable reports (PDF/HTML)


## 🔒 Notes

Do not commit secrets: add .env and chroma_db/ to .gitignore.

This app sends prompts and (summarized) context to OpenAI. Review your data governance requirements before uploading sensitive data.

.gitignore suggestion

.venv/
.env
__pycache__/
chroma_db/
.streamlit/


## 🙌 Acknowledgements

Streamlit
 — rapid data apps

Plotly
 — interactive charts

LangChain
 — LLM orchestration, RAG chains

Chroma
 — local, persistent vector store

OpenAI
 — chat + embeddings