# Marketify AI — GraphRAG for Paid-Media Ad Copy
> Grounded answers about ad copy performance using a Neo4j knowledge graph + a guarded LLM→Cypher translator, with a simple Streamlit UI.

This project lets you ask questions like:

- “Best performing ads for **Gym** in **Berlin**, top 5”
- “Top **keywords** for **Italian Restaurant** in **NYC**”
- “Explain ad **by ID** (show creative text, metrics, industry/region/keyword links)”
- Free-form “Ask Anything” → **LLM→Cypher** (validated) → results with provenance

All answers are backed by explicit nodes/edges in Neo4j (no hallucinated sources).  
Dataset is **synthetic** and safe to publish.

---

## TL;DR (copy-paste setup)

```bash
# 1) Clone + install
git clone https://github.com/Bhargav141922/Generative-Ad-Copy-Chatbot-for-Paid-Media-Using-Knowledge-Graphs-and-GraphRAG.git
cd Generative-Ad-Copy-Chatbot-for-Paid-Media-Using-Knowledge-Graphs-and-GraphRAG
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure
cp .env.example .env
# → open .env and fill in NEO4J_* and (optionally) OPENAI_* keys

# 3) Create graph constraints & indexes (idempotent)
# Use whichever script exists in your repo; try in this order:
python -m graph_app.setup_constraints  || \
python graph_app/setup_constraints.py  || \
python scripts/setup_constraints.py

# 4) Import data into Neo4j
# If you already have node/relation CSVs:
python -m graph_app.import_graph --nodes data/exports/nodes --rels data/exports/rels  || \
python graph_app/import_graph.py --nodes data/exports/nodes --rels data/exports/rels

# If you only have a clean snapshot (e.g., data/processed/ads_clean.parquet), try the helper:
python -m data_preprocessing.export_graph  ||  python data_preprocessing/export_graph.py
# (It will write node/rel CSVs under data/exports/)

# 5) Run the UI
streamlit run graph_app/app.py  ||  streamlit run app.py
