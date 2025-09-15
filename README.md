# Generative Ad Copy Chatbot for Paid Media Using Knowledge Graphs and GraphRAG

## Marketify AI – GraphRAG for Paid Media Ad Copy  
Grounded answers about ad copy performance using a Neo4j knowledge graph + a guarded LLM-Cypher translator, with a simple Streamlit UI.

---

### What It Does
This project lets you ask questions like:

- **Best performing ads for [industry] in [region], top 5**  
- **Top keywords for [industry] in [region]**  
- **Explain ad by ID** (show creative text, metrics, industry/region/keyword links)  
- Free-form **“Ask Anything”** → LLM → Cypher (validated) → results with provenance  

✅ All answers are backed by explicit nodes/edges in Neo4j (no hallucinated sources).  
📊 Dataset is **synthetic** and safe to publish.  

---

### 🚀 TL;DR (Setup in 5 Steps)
```bash
# 1. Clone + install
git clone https://github.com/yourusername/Generative-Ad-Copy-Chatbot-for-Paid-Media-Using-Knowledge-Graphs-and-GraphRAG.git
cd Generative-Ad-Copy-Chatbot-for-Paid-Media-Using-Knowledge-Graphs-and-GraphRAG
python -m venv .venv

# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials

# 3. Generate synthetic data
python data_generation/generate_data.py

# 4. Preprocess data
python data_preprocessing/preprocess_data.py

# 5. Run the application
streamlit run graph_app/streamlit_app.py

## ✨ Features

1. GraphRAG Implementation → Combines knowledge graphs with retrieval-augmented generation

2. Multi-LLM Support → Works with Groq, Gemini, and OpenRouter models

3. Neo4j Integration → All data stored and queried through a knowledge graph

4. Streamlit UI → Clean, interactive interface for exploring ad performance

5. Synthetic Data → Safe, generated dataset for research purposes
