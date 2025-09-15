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
```
---

 ### ✨ Features

- GraphRAG Implementation → Combines knowledge graphs with retrieval-augmented generation
- Multi-LLM Support → Works with Groq, Gemini, and OpenRouter models
- Neo4j Integration → All data stored and queried through a knowledge graph
- Streamlit UI → Clean, interactive interface for exploring ad performance
- Synthetic Data → Safe, generated dataset for research purposes

---
## 📂 Project Structure
```
├── data_generation/
│   ├── __init__.py
│   └── generate_data.py
├── data_preprocessing/
│   ├── __init__.py
│   └── preprocess_data.py
├── graph_app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   └── requirements.txt
├── data/
│   ├── raw/
│   ├── clean/
│   ├── import/
│   └── import_bulk/
├── assets/
│   └── logo.png
├── .env.example
└── README.md
```
---
### ⚙️ Setup Details

- Environment Setup → Create a virtual environment and install dependencies
- API Configuration → Add your API keys to the .env file
- Data Generation → Create synthetic ad performance data
- Data Processing → Prepare data for Neo4j import
- Neo4j Setup → Import processed data and create indexes
- Application Launch → Start the Streamlit interface
---
### 🔎 Usage Examples
#### Find Top Performing Ads:

- "Show me the best performing gym ads in Berlin"
- "Top 5 hair salon ads in New York"

#### Keyword Analysis:

- "What keywords are most effective for Italian restaurants?"
- "Show keyword usage for digital marketing agencies"

#### Ad Analysis:

- "Explain ad with ID [ad_id]"
- "Show me ads containing free trial"

---
### 📌 Supported Industries
- Gym / Fitness Centers
- Hair Salons
- Roofing Companies
- Online Clothing Boutiques
- Italian Restaurants
- Digital Marketing Agencies
- Local Coffee Shops
- Plumbing Services
- Yoga Studios
- Pet Grooming Services

---
### 🛠️ Technical Stack

- **Backend** → Python, Neo4j Graph Database
- **Frontend** → Streamlit
- **LLM Integration** → Google Gemini, Groq, OpenRouter
- **Data Processing** → Pandas, NumPy
- **Visualization** → Matplotlib, Seaborn

### 📜 License

This project is part of academic research.
Please contact the author for usage permissions.
