# streamlit_app.py
import os, re, textwrap, base64
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from neo4j import GraphDatabase

# (optional) load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------- Paths & Helpers ---------------------------
LOGO_PATH = "../assets/logo.png"
HAS_LOGO = os.path.exists(LOGO_PATH)

# --------------------------- Page & Theme ---------------------------
st.set_page_config(
    page_title="Marketify AI — GraphRAG",
    page_icon=(LOGO_PATH if HAS_LOGO else "🧩"),
    layout="wide",
)

def inject_css():
    st.markdown("""
    <style>
      section.main > div { padding-top: 0.6rem; }

      /* Compact flex header (no anchors) */
      .app-header{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1rem 0; }
      .app-header .logo{ width:36px; height:auto; display:block; }
      .app-header .emoji{ font-size:28px; line-height:1; }

      .app-title {
        font-size: clamp(1.8rem, 2.2vw + 1rem, 2.6rem);
        font-weight: 800; line-height: 1.1; margin: 0;
      }
      .app-title span {
        background: linear-gradient(90deg,#a288f6,#62b3ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }

      /* Polished dataframe look */
      div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
      div[data-testid="stDataFrame"] thead tr th {
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
      }
      div[data-testid="stDataFrame"] tbody tr td {
        border-bottom: 1px dashed rgba(255,255,255,0.06) !important;
      }

      .chip { display:inline-block; padding:.18rem .55rem; border-radius:999px;
              background:#1f2430; border:1px solid #2a3040; font-size:.82rem; }
      .chip.badge-high{ background:#10331f; border-color:#1f6f3a; color:#9ce3b3 }
      .chip.badge-mid { background:#23220e; border-color:#736b1f; color:#e9da86 }
      .chip.badge-low { background:#361c1c; border-color:#7a2727; color:#f3a5a5 }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    if HAS_LOGO:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f"<img class='logo' src='data:image/png;base64,{b64}' alt='logo'/>"
    else:
        logo_html = "<div class='logo emoji'>🧩</div>"

    st.markdown(
        f"""
        <div class="app-header">
          {logo_html}
          <div class="app-title" role="heading" aria-level="1">
            <span>Marketify AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

inject_css()
render_header()

# --------------------------- Small helpers ---------------------------
def tier_class(tier:str) -> str:
    t = (tier or "").lower()
    if t=="high": return "badge-high"
    if t=="mid":  return "badge-mid"
    return "badge-low"

def ctr_percent(value) -> float:
    """Return CTR as percent points (e.g., 12.34). If input is 0..1, convert by *100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    v = v*100.0 if 0.0 <= v <= 1.0 else v
    return max(0.0, v)

def normalize_ad_row(r: Dict[str, Any]) -> Dict[str, Any]:
    h1  = r.get("h1")  or r.get("headline_1") or r.get("a.headline_1") or ""
    d1  = r.get("d1")  or r.get("description_1") or r.get("a.description_1") or ""
    ctr_val = r.get("ctr", r.get("simulated_ctr", r.get("a.simulated_ctr", 0)))
    z   = float(r.get("z",  r.get("ctr_z_final",   r.get("a.ctr_z_final",   0))) or 0)
    tier= r.get("tier", r.get("tier_final", r.get("a.tier_final","")))
    idv = r.get("id") or r.get("ad_id") or r.get("a.ad_id") or r.get("elementId") or "—"
    return {
        "ID": str(idv),
        "Headline": h1,
        "Description": d1,
        "CTR (%)": round(ctr_percent(ctr_val), 2),   # percent value
        "z": round(z, 4),
        "Tier": (str(tier).title() if tier else "—"),
    }

def render_ads_table(rows: List[Dict[str, Any]], key: Optional[str]=None):
    if not rows:
        st.warning("No results.")
        return
    norm = [normalize_ad_row(r) for r in rows]
    df = pd.DataFrame(norm)

    st.dataframe(
        df[["Headline", "Description", "CTR (%)", "z", "Tier", "ID"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Headline": st.column_config.TextColumn("Headline", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "CTR (%)": st.column_config.NumberColumn(
                "CTR (%)",
                help="Click-through rate",
                format="%.2f%%",
            ),
            "z": st.column_config.NumberColumn("z-score", format="%.3f"),
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "ID": st.column_config.TextColumn("ID", width="medium"),
        },
        key=key,
    )
    st.caption(f"{len(df)} rows")

# --------------------------- Sidebar / Secrets ---------------------------
st.sidebar.title("⚙️ Settings")
if HAS_LOGO: st.sidebar.image(LOGO_PATH, width=140)

NEO4J_URI  = st.secrets.get("neo4j_uri",  os.getenv("NEO4J_URI",  st.sidebar.text_input("Neo4j Aura URI (neo4j+s://…)", value="")))
NEO4J_USER = st.secrets.get("neo4j_user", os.getenv("NEO4J_USER", st.sidebar.text_input("Neo4j user", value="neo4j")))
NEO4J_PASS = st.secrets.get("neo4j_pass", os.getenv("NEO4J_PASS", st.sidebar.text_input("Neo4j password", type="password")))

provider = st.sidebar.selectbox("LLM provider (free tiers)", ["groq", "gemini", "openrouter"], index=0)

# expose secrets to env
if "groq_api_key" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["groq_api_key"]
if "gemini_api_key" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = st.secrets["gemini_api_key"]
if "openrouter_api_key" in st.secrets and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = st.secrets["openrouter_api_key"]

# --------------------------- Neo4j Connection (persistent) ---------------------------
@st.cache_resource(show_spinner=False)
def get_driver(uri: str, user: str, pwd: str):
    return GraphDatabase.driver(uri, auth=(user, pwd))

if "neo_driver" not in st.session_state:
    st.session_state.neo_driver = None
if "neo_connected" not in st.session_state:
    st.session_state.neo_connected = False

colA, colB = st.sidebar.columns(2)
if colA.button("🔌 Connect"):
    try:
        st.session_state.neo_driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
        with st.session_state.neo_driver.session() as s:
            s.run("RETURN 1")
        st.session_state.neo_connected = True
        st.sidebar.success("Connected ✅")
    except Exception as e:
        st.session_state.neo_driver = None
        st.session_state.neo_connected = False
        st.sidebar.error(f"Neo4j connection failed: {e}")

if colB.button("🔴 Disconnect") and st.session_state.neo_driver:
    try:
        st.session_state.neo_driver.close()
    except Exception:
        pass
    st.session_state.neo_driver = None
    st.session_state.neo_connected = False
    st.sidebar.info("Disconnected.")

driver = st.session_state.neo_driver

def run_cypher(cypher: str, **params) -> List[Dict[str, Any]]:
    if not st.session_state.neo_connected or driver is None:
        raise RuntimeError("Not connected to Neo4j. Click 'Connect'.")
    with driver.session() as s:
        with st.spinner("Running query…"):
            return [r.data() for r in s.run(cypher, **params)]

# --------------------------- Templates (deterministic) ---------------------------
TEMPLATES = {
"best_ads_fulltext": """
CALL db.index.fulltext.queryNodes('region_text_idx', $regionText) YIELD node AS r, score
MATCH (i:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
OPTIONAL MATCH (a)-[:Mentions]->(kw:Keywords)
WITH a, kw
WHERE $keyword IS NULL OR kw.lemma = $keyword
RETURN coalesce(a.ad_id, elementId(a)) AS id, a.headline_1 AS h1, a.description_1 AS d1,
       a.simulated_ctr AS ctr, a.ctr_z_final AS z, a.tier_final AS tier
ORDER BY coalesce(a.ctr_z_final, a.simulated_ctr) DESC
LIMIT $k
""",
"best_ads_contains": """
MATCH (i:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r:Regions)
WHERE $regionText IS NULL OR $regionText = "" OR
      toLower(coalesce(r.name,'')) CONTAINS toLower($regionText) OR
      toLower(coalesce(r.code,'')) = toLower($regionText)
OPTIONAL MATCH (a)-[:Mentions]->(kw:Keywords)
WITH a, kw
WHERE $keyword IS NULL OR kw.lemma = $keyword
RETURN coalesce(a.ad_id, elementId(a)) AS id, a.headline_1 AS h1, a.description_1 AS d1,
       a.simulated_ctr AS ctr, a.ctr_z_final AS z, a.tier_final AS tier
ORDER BY coalesce(a.ctr_z_final, a.simulated_ctr) DESC
LIMIT $k
""",
"top_keywords_fulltext": """
CALL db.index.fulltext.queryNodes('region_text_idx', $regionText) YIELD node AS r, score
MATCH (:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
MATCH (a)-[:Mentions]->(k:Keywords)
RETURN k.lemma AS lemma, count(*) AS uses
ORDER BY uses DESC
LIMIT $topN
""",
"top_keywords_contains": """
MATCH (:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r:Regions)
WHERE $regionText IS NULL OR $regionText = "" OR
      toLower(coalesce(r.name,'')) CONTAINS toLower($regionText) OR
      toLower(coalesce(r.code,'')) = toLower($regionText)
MATCH (a)-[:Mentions]->(k:Keywords)
RETURN k.lemma AS lemma, count(*) AS uses
ORDER BY uses DESC
LIMIT $topN
""",
"explain_ad": """
// Accept either the business id (a.ad_id) or the elementId(a)
MATCH (a:AdCopy)
WHERE ($adId IS NOT NULL) AND (a.ad_id = $adId OR elementId(a) = $adId)
OPTIONAL MATCH (a)-[:Belongs_to]->(i:Industries)
OPTIONAL MATCH (a)-[:Targets]->(r:Regions)
OPTIONAL MATCH (a)-[:Mentions]->(k:Keywords)
RETURN
  coalesce(a.ad_id, elementId(a))                         AS id,
  a.headline_1                                            AS h1,
  a.headline_2                                            AS h2,
  a.description_1                                         AS d1,
  a.description_2                                         AS d2,
  a.description_3                                         AS d3,
  a.simulated_ctr                                         AS ctr,
  a.ctr_z_final                                           AS z,
  a.tier_final                                            AS tier,
  head(collect(DISTINCT i.name))                          AS industry,
  collect(DISTINCT r.name)                                AS regions,
  collect(DISTINCT k.lemma)                               AS keywords
"""
}

def best_ads(industry: str, region_text: str, k=5, keyword=None, use_fulltext=True):
    q = "best_ads_fulltext" if (use_fulltext and region_text) else "best_ads_contains"
    return run_cypher(TEMPLATES[q], industry=industry, regionText=region_text, k=k, keyword=keyword)

def top_keywords(industry: str, region_text: str, topN=15, use_fulltext=True):
    q = "top_keywords_fulltext" if (use_fulltext and region_text) else "top_keywords_contains"
    return run_cypher(TEMPLATES[q], industry=industry, regionText=region_text, topN=topN)

def explain_ad(ad_id: str):
    return run_cypher(TEMPLATES["explain_ad"], adId=ad_id)

# --------------------------- LLM → Cypher (fallback) ---------------------------
SCHEMA_PRIMER = """
You write Cypher for Neo4j using THIS schema (use exact casing):

Node labels:
- AdCopy(ad_id, headline_1, headline_2, description_1, description_2, description_3,
         simulated_ctr, ctr_z_final, tier_final)
- Industries(ind_id, name)
- Regions(code, name)
- Keywords(lemma)

Relationships (direction matters):
- (:AdCopy)-[:Belongs_to]->(:Industries)
- (:AdCopy)-[:Targets]->(:Regions)
- (:AdCopy)-[:Mentions]->(:Keywords)

Rules:
- Prefer ranking by coalesce(a.ctr_z_final, a.simulated_ctr) DESC.
- Always LIMIT results (<= 50).
- Use parameters where possible: $industry, $regionText, $keyword, $k, $topN, $adId.
- For region lookup, prefer: CALL db.index.fulltext.queryNodes('region_text_idx', $regionText) …
- **Do NOT include any [:Mentions]/Keywords filters unless the user explicitly asks for a keyword (e.g., 'mentioning X', 'keyword X').**
- Never CREATE/MERGE/SET/DELETE. Read-only queries only.
Return only a Cypher string, nothing else.
""".strip()

FEW_SHOTS = [
{
 "user": "best performing ads for Berlin in Gym mentioning free trial, top 5",
 "cypher": """
CALL db.index.fulltext.queryNodes('region_text_idx', $regionText) YIELD node AS r, score
MATCH (i:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
MATCH (a)-[:Mentions]->(:Keywords {lemma:$keyword})
RETURN a.ad_id, a.headline_1, a.description_1, a.simulated_ctr, a.ctr_z_final, a.tier_final
ORDER BY coalesce(a.ctr_z_final, a.simulated_ctr) DESC
LIMIT $k
"""
},
{
 "user": "best performing ads for Gym in London, top 5",
 "cypher": """
CALL db.index.fulltext.queryNodes('region_text_idx', $regionText) YIELD node AS r, score
MATCH (i:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
RETURN a.ad_id, a.headline_1, a.description_1, a.simulated_ctr, a.ctr_z_final, a.tier_final
ORDER BY coalesce(a.ctr_z_final, a.simulated_ctr) DESC
LIMIT $k
"""
}
]

def _build_messages(question: str):
    msgs = [{"role": "system", "content": SCHEMA_PRIMER}]
    for ex in FEW_SHOTS:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["cypher"].strip()})
    msgs.append({"role": "user", "content": question})
    return msgs

def llm_to_cypher(question: str, provider_choice: str) -> str:
    if provider_choice == "groq":
        from openai import OpenAI
        client = OpenAI(base_url="https://api.groq.com/openai/v1",
                        api_key=os.environ.get("GROQ_API_KEY", ""))
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=_build_messages(question),
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    elif provider_choice == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel("gemini-2.5-flash")
        fewshot = "\n\n".join(
            f"User: {ex['user']}\nAssistant:\n{ex['cypher'].strip()}" for ex in FEW_SHOTS
        )
        prompt = f"{SCHEMA_PRIMER}\n\n{fewshot}\n\nUser: {question}\nAssistant:"
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return (resp.text or "").strip()
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ.get("OPENROUTER_API_KEY", ""))
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite:free",
            messages=_build_messages(question),
            temperature=0.2,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Marketify-AI"},
        )
        return resp.choices[0].message.content.strip()

def validate_cypher(c: str):
    cu = re.sub(r"\s+", " ", c).upper()
    if any(k in cu for k in [" CREATE ", " MERGE ", " DELETE ", " DETACH ", " REMOVE ", " SET "]):
        raise ValueError("Unsafe (write) clause found.")
    if " LIMIT " not in cu:
        raise ValueError("Missing LIMIT.")

# -------- Post-processing (regions, keyword, fulltext fallback) --------
def soften_region_filter(cy: str) -> str:
    cy = cy.replace("$topN", "$k")
    if "(r:Regions {name:$regionText})" in cy:
        cy = cy.replace("(r:Regions {name:$regionText})", "(r:Regions)")
        if re.search(r"\bWHERE\b", cy, flags=re.IGNORECASE):
            cy = re.sub(
                r"\bRETURN\b",
                "AND (toLower(coalesce(r.name,'')) CONTAINS toLower($regionText) "
                "OR toLower(coalesce(r.code,'')) = toLower($regionText))\nRETURN",
                cy, count=1, flags=re.IGNORECASE
            )
        else:
            cy = re.sub(
                r"\bRETURN\b",
                "WHERE (toLower(coalesce(r.name,'')) CONTAINS toLower($regionText) "
                "OR toLower(coalesce(r.code,'')) = toLower($regionText))\nRETURN",
                cy, count=1, flags=re.IGNORECASE
            )
    return cy

def make_keyword_optional(cy: str) -> str:
    cy = re.sub(
        r"MATCH\s*\((\w+)\)\s*-\s*\[:Mentions\]\s*->\s*\(:\s*Keywords\s*\{\s*lemma\s*:\s*\$keyword\s*\}\)",
        r"OPTIONAL MATCH (\1)-[:Mentions]->(kw:Keywords)\nWHERE $keyword IS NULL OR toLower(kw.lemma)=toLower($keyword)",
        cy, flags=re.IGNORECASE
    )
    cy = re.sub(
        r"MATCH\s*\((\w+)\)\s*-\s*\[:Mentions\]\s*->\s*\((\w+)\s*:\s*Keywords\s*\{\s*lemma\s*:\s*\$keyword\s*\}\)",
        r"OPTIONAL MATCH (\1)-[:Mentions]->(\2:Keywords)\nWHERE $keyword IS NULL OR toLower(\2.lemma)=toLower($keyword)",
        cy, flags=re.IGNORECASE
    )
    return cy

def fulltext_exists(name: str) -> bool:
    try:
        rows = run_cypher("SHOW INDEXES YIELD name WHERE name = $n RETURN count(*) AS c", n=name)
        return rows and rows[0]["c"] > 0
    except Exception:
        return False

def replace_fulltext_with_contains(cy: str) -> str:
    pat = r"CALL\s+db\.index\.fulltext\.queryNodes\('region_text_idx',\s*\$regionText\)\s+YIELD\s+node\s+AS\s+r,\s*score"
    if re.search(pat, cy, flags=re.IGNORECASE):
        repl = (
            "MATCH (r:Regions)\n"
            "WHERE toLower(coalesce(r.name,'')) CONTAINS toLower($regionText) "
            "OR toLower(coalesce(r.code,'')) = toLower($regionText)"
        )
        cy = re.sub(pat, repl, cy, flags=re.IGNORECASE)
    return cy

# --------------------------- Param guessing ---------------------------
def _list_industries() -> List[str]:
    rows = run_cypher("MATCH (i:Industries) RETURN i.name AS n")
    return [r["n"] for r in rows]

def _list_regions() -> List[Dict[str, Any]]:
    rows = run_cypher("MATCH (r:Regions) RETURN coalesce(r.name,'') AS name, coalesce(r.code,'') AS code")
    return rows

def extract_keyword_from_question(q: str) -> Optional[str]:
    ql = q.lower()
    m = re.search(r"(mentioning|keyword|contains|containing)\s+['\"]([^'\"]+)['\"]", ql)
    if m: return m.group(2)
    m = re.search(r"(mentioning|keyword|contains|containing)\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,2})", ql)
    return m.group(2) if m else None

def guess_params_from_question(q: str) -> Dict[str, Any]:
    qlow = q.lower()
    params: Dict[str, Any] = {}

    # k
    m = re.search(r"\btop\s+(\d{1,3})\b", qlow)
    params["k"] = int(m.group(1)) if m else 5

    # industry
    inds = _list_industries()
    params["industry"] = None
    for ind in sorted(inds, key=len, reverse=True):
        if ind.lower() in qlow:
            params["industry"] = ind
            break
    if not params["industry"]:
        params["industry"] = "Gym"

    # region (text after ' in ')
    params["regionText"] = None
    m = re.search(r"\bin\s+([A-Za-z\-\s,]+)", q)
    if m:
        candidate = m.group(1).strip().rstrip("?.,").lower()
        if candidate:
            params["regionText"] = candidate

    # normalize region to what exists
    if params["regionText"]:
        reg_rows = _list_regions()
        cand = params["regionText"]
        for r in reg_rows:
            name = (r["name"] or "").lower()
            code = (r["code"] or "").lower()
            if (cand and cand in name) or (code and cand == code):
                params["regionText"] = r["name"] or r["code"]
                break

    # keyword ONLY if asked
    params["keyword"] = extract_keyword_from_question(q)
    return params

# --------------------------- Prompted Generation ---------------------------
def build_prompt(industry: str, region_text: str, examples: List[Dict[str, Any]], keyword=None):
    bullets_lines = []
    for e in examples:
        ctr_text = f"{ctr_percent(e.get('ctr', e.get('simulated_ctr', 0))):.2f}%"
        h1 = e.get('h1') or e.get('headline_1') or e.get('a.headline_1')
        d1 = e.get('d1') or e.get('description_1') or e.get('a.description_1')
        bullets_lines.append(f"- {h1} — {d1} (CTR {ctr_text})")
    bullets = "\n".join(bullets_lines)

    must = f" Include the keyword '{keyword}'." if keyword else ""
    return textwrap.dedent(f"""
        You are an ad-copy writer.

        Context:
        - Industry: {industry}
        - Region: {region_text or '(any)'}
        - High-performing examples:
        {bullets}

        Task:
        Write ONE new headline and ONE description.{must}
        Tone: upbeat and trustworthy. Constraints: headline ≤ 6 words, no exclamation marks.

        Return:
        Headline: …
        Description: …
    """).strip()

def llm_generate_text(prompt: str, provider_choice: str) -> str:
    if provider_choice == "groq":
        from openai import OpenAI
        client = OpenAI(base_url="https://api.groq.com/openai/v1",
                        api_key=os.environ.get("GROQ_API_KEY", ""))
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    elif provider_choice == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt, generation_config={"temperature": 0.7})
        return (resp.text or "").strip()
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1",
                        api_key=os.environ.get("OPENROUTER_API_KEY", ""))
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "Marketify-AI"},
        )
        return resp.choices[0].message.content.strip()

# --------------------------- UI ---------------------------
if not st.session_state.neo_connected:
    st.info("Enter credentials in the sidebar and click **Connect**.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Best Ads", "Top Keywords", "Explain Ad", "Ask Anything (LLM→Cypher)", "Setup"]
    )

    # ---------- Best Ads ----------
    with tab1:
        st.subheader("Best performing ads (Templates, optional Mentions)")
        industry = st.text_input("Industry", value="Gym")
        region_text = st.text_input("Region (free text)", value="")
        keyword = st.text_input("Keyword filter (optional)")
        k = st.number_input("How many results", min_value=1, max_value=50, value=5, step=1)
        use_fulltext = st.toggle("Use full-text region index", value=False)

        # Region picker (existing regions for this industry)
        try:
            opts = run_cypher("""
                MATCH (:Industries {name:$industry})<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r:Regions)
                RETURN DISTINCT coalesce(r.name,'') AS name, coalesce(r.code,'') AS code
                ORDER BY name, code
                LIMIT 100
            """, industry=industry)
            pretty = [f"{o['name']} ({o['code']})" if o['name'] else o['code'] for o in opts if o['name'] or o['code']]
            picked = st.selectbox("Pick a region that exists (optional)", [""] + pretty, index=0)
            if picked and picked != "":
                region_text = picked.split(" (")[0] if " (" in picked else picked
        except Exception as e:
            st.caption(f"Region picker unavailable: {e}")

        colA, colB, colC = st.columns(3)
        run_template = colA.button("Run (Template)")
        run_hybrid   = colB.button("Run (Hybrid: template → fallback)")
        gen_button   = colC.button("Generate ad from results")
        st.caption("Rels: AdCopy —[:Belongs_to]→ Industries, AdCopy —[:Targets]→ Regions, (optional) AdCopy —[:Mentions]→ Keywords")

        rows: List[Dict[str, Any]] = []
        if run_template or run_hybrid:
            try:
                rows = best_ads(industry, region_text, k=k, keyword=(keyword or None), use_fulltext=use_fulltext)
            except Exception as e:
                st.error(e)

            if (not rows) and run_hybrid:
                q = f"best performing ads for {industry}" + (f" in {region_text}" if region_text else "") + f", top {k}"
                try:
                    cy = llm_to_cypher(q, provider)
                    cy = soften_region_filter(cy)
                    cy = make_keyword_optional(cy)
                    if "db.index.fulltext.queryNodes('region_text_idx'" in cy and not fulltext_exists("region_text_idx"):
                        cy = replace_fulltext_with_contains(cy)
                    st.code(cy, language="cypher")
                    validate_cypher(cy)
                    params = {"industry":industry, "regionText":region_text, "keyword":(keyword or None), "k":k}
                    rows = run_cypher(cy, **params)
                except Exception as e:
                    st.error(f"Fallback failed: {e}")

            if rows:
                render_ads_table(rows, key="bestads")
            else:
                st.warning("No results. Try a region from the picker or turn OFF full-text (uses CONTAINS).")

        if gen_button:
            try:
                rows = best_ads(industry, region_text, k=k, keyword=(keyword or None), use_fulltext=use_fulltext)
                if rows:
                    prompt = build_prompt(industry, region_text or "(any)", rows, keyword=(keyword or None))
                    st.code(prompt)
                    out = llm_generate_text(prompt, provider)
                    st.markdown("### ✍️ Generated Copy")
                    st.write(out)
                else:
                    st.warning("No rows to build a prompt. Run a retrieval first.")
            except Exception as e:
                st.error(e)

        with st.expander("🔎 Diagnostics"):
            try:
                c_ad   = run_cypher("MATCH (n:AdCopy) RETURN count(n) AS c")[0]["c"]
                c_ind  = run_cypher("MATCH (n:Industries) RETURN count(n) AS c")[0]["c"]
                c_reg  = run_cypher("MATCH (n:Regions) RETURN count(n) AS c")[0]["c"]
                c_kw   = run_cypher("MATCH (n:Keywords) RETURN count(n) AS c")[0]["c"]
                a,b,c,d = st.columns(4)
                a.metric("AdCopy", c_ad)
                b.metric("Industries", c_ind)
                c.metric("Regions", c_reg)
                d.metric("Keywords", c_kw)
                inds = run_cypher("MATCH (i:Industries) RETURN i.name AS name ORDER BY name LIMIT 20")
                regs = run_cypher("MATCH (r:Regions) RETURN r.name AS name, r.code AS code ORDER BY name LIMIT 20")
                st.write("Industries sample:", [x["name"] for x in inds])
                st.write("Regions sample:", [f'{x["name"]} ({x["code"]})' for x in regs])
            except Exception as e:
                st.error(e)

    # ---------- Top Keywords ----------
    with tab2:
        st.subheader("Top Keywords in an Industry–Region (Templates)")
        industry2 = st.text_input("Industry ", value="Gym", key="ind2")
        region_text2 = st.text_input("Region (free text) ", value="", key="reg2")
        topN = st.number_input("How many keywords", min_value=5, max_value=50, value=15, step=1)
        use_fulltext2 = st.toggle("Use full-text region index ", value=False, key="ft2")
        if st.button("Run (Template)", key="kwbtn"):
            try:
                rows = top_keywords(industry2, region_text2, topN=topN, use_fulltext=use_fulltext2)
                if rows:
                    df = pd.DataFrame(rows).rename(columns={"lemma":"Keyword", "uses":"Uses"})
                    total = int(df["Uses"].sum()) or 1
                    df["Share (%)"] = (df["Uses"] / total * 100).round(2)
                    st.dataframe(
                        df[["Keyword", "Uses", "Share (%)"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Keyword": st.column_config.TextColumn("Keyword", width="large"),
                            "Uses": st.column_config.NumberColumn("Uses", format="%d"),
                            "Share (%)": st.column_config.ProgressColumn(
                                "Share",
                                help="Share of total keyword mentions for this result set",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                    )
                else:
                    st.warning("0 rows. Try a different region (use the picker on the Best Ads tab).")
            except Exception as e:
                st.error(e)

    # ---------- Explain Ad ----------
    with tab3:
        st.subheader("Explain an Ad (robust)")
        ad_id = st.text_input("Paste the ad ID you saw in the results table (either ad_id or elementId)")
        colx, coly = st.columns([1,1])
        find_text = colx.text_input("…or find by text (headline/description CONTAINS)", value="")
        find_k = coly.number_input("Show", 1, 50, 10, 1)

        if st.button("Find candidates"):
            if not find_text.strip():
                st.warning("Enter some text to search.")
            else:
                try:
                    rows = run_cypher(
                        """
                        MATCH (a:AdCopy)
                        WHERE toLower(coalesce(a.headline_1,''))    CONTAINS toLower($q)
                           OR toLower(coalesce(a.headline_2,''))    CONTAINS toLower($q)
                           OR toLower(coalesce(a.description_1,'')) CONTAINS toLower($q)
                           OR toLower(coalesce(a.description_2,'')) CONTAINS toLower($q)
                           OR toLower(coalesce(a.description_3,'')) CONTAINS toLower($q)
                        RETURN
                          coalesce(a.ad_id, elementId(a)) AS id,
                          a.headline_1 AS h1,
                          a.description_1 AS d1,
                          a.simulated_ctr AS ctr,
                          a.ctr_z_final AS z,
                          a.tier_final AS tier
                        ORDER BY coalesce(a.ctr_z_final, a.simulated_ctr) DESC
                        LIMIT $k
                        """,
                        q=find_text, k=int(find_k)
                    )
                    if rows:
                        render_ads_table(rows, key="findcandidates")
                        st.caption("Copy an **ID** from the last column and paste it above, then click Explain.")
                    else:
                        st.warning("No candidates. Try a different phrase.")
                except Exception as e:
                    st.error(e)

        if st.button("Explain"):
            if not ad_id.strip():
                st.warning("Paste an id first (from your results table or from 'Find candidates').")
            else:
                try:
                    rows = explain_ad(ad_id.strip())
                    if rows:
                        r = rows[0]
                        df = pd.DataFrame([{
                            "ID": r.get("id","—"),
                            "Industry": r.get("industry") or "—",
                            "Headline 1": r.get("h1") or "—",
                            "Headline 2": r.get("h2") or "—",
                            "Description 1": r.get("d1") or "—",
                            "Description 2": r.get("d2") or "—",
                            "Description 3": r.get("d3") or "—",
                            "CTR (%)": round(ctr_percent(r.get("ctr",0)), 2),
                            "z-score": round(float(r.get("z",0) or 0), 3),
                            "Tier": (str(r.get("tier","")).title() or "—"),
                            "Regions": ", ".join(r.get("regions") or []) or "—",
                            "Keywords": ", ".join(r.get("keywords") or []) or "—",
                        }])
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "CTR (%)": st.column_config.NumberColumn("CTR (%)", format="%.2f%%"),
                                "z-score": st.column_config.NumberColumn("z-score", format="%.3f"),
                            },
                        )
                    else:
                        st.warning("No ad found for that id.")
                except Exception as e:
                    st.error(e)

    # ---------- Ask Anything ----------
    with tab4:
        st.subheader("Ask Anything (LLM → Cypher fallback)")
        question = st.text_input("Question", value="best performing ads for gym in London, top 5")
        if st.button("Ask"):
            try:
                cy = llm_to_cypher(question, provider)
                cy = soften_region_filter(cy)
                cy = make_keyword_optional(cy)
                if "db.index.fulltext.queryNodes('region_text_idx'" in cy and not fulltext_exists("region_text_idx"):
                    cy = replace_fulltext_with_contains(cy)
                st.code(cy, language="cypher")

                guessed = guess_params_from_question(question)
                if "keyword" not in guessed:
                    guessed["keyword"] = None  # never missing

                validate_cypher(cy)
                rows = run_cypher(cy, **guessed)
                if rows:
                    render_ads_table(rows, key="askanything")
                    st.caption(f"Params used: {guessed}")
                else:
                    st.warning(f"The query ran but returned 0 rows with params={guessed}. "
                               f"Pick a region from Diagnostics or ask without a city.")
            except Exception as e:
                st.error(f"Generated Cypher rejected/run failed: {e}")

    # ---------- Setup ----------
    with tab5:
        st.subheader("Setup")
        st.caption("Create constraints/indexes (including full-text for regions) and compute IR-normalised metrics.")
        col1, col2 = st.columns(2)
        if col1.button("Create constraints & indexes"):
            try:
                ops = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:AdCopy)     REQUIRE a.ad_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industries) REQUIRE i.ind_id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Regions)    REQUIRE r.code  IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keywords)   REQUIRE k.lemma IS UNIQUE",
                    "CREATE INDEX  IF NOT EXISTS FOR (a:AdCopy) ON (a.simulated_ctr)",
                    "CREATE FULLTEXT INDEX region_text_idx IF NOT EXISTS FOR (r:Regions) ON EACH [r.name, r.code]",
                ]
                for q in ops: run_cypher(q)
                st.success("Constraints & indexes ensured ✅")
            except Exception as e:
                st.error(e)

        minN = col2.number_input("Min ads per (Industry×Region)", 5, 100, 10, 1)
        if st.button("Compute IR metrics"):
            blocks = [
                f"""
                MATCH (i:Industries)<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r:Regions)
                WITH i, r, avg(a.simulated_ctr) AS mu, stDev(a.simulated_ctr) AS sigma, count(*) AS n
                WHERE n >= {int(minN)}
                MATCH (i)<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
                SET a.ctr_z_ir = CASE WHEN sigma = 0 THEN 0 ELSE (a.simulated_ctr - mu)/sigma END
                """,
                f"""
                MATCH (i:Industries)<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r:Regions)
                WITH i, r,
                     percentileCont(a.simulated_ctr, 0.33) AS p33,
                     percentileCont(a.simulated_ctr, 0.66) AS p66,
                     percentileCont(a.simulated_ctr, 0.75) AS q3,
                     count(*) AS n
                WHERE n >= {int(minN)}
                MATCH (i)<-[:Belongs_to]-(a:AdCopy)-[:Targets]->(r)
                SET a.ctr_topq_ir = a.simulated_ctr >= q3,
                    a.tier_ir =
                      CASE
                        WHEN a.simulated_ctr <  p33 THEN 'Low'
                        WHEN a.simulated_ctr <= p66 THEN 'Mid'
                        ELSE 'High'
                      END
                """,
                """
                MATCH (i:Industries)<-[:Belongs_to]-(a:AdCopy)
                WHERE a.ctr_z_ir IS NULL
                WITH i, avg(a.simulated_ctr) AS mu, stDev(a.simulated_ctr) AS sigma
                MATCH (i)<-[:Belongs_to]-(a:AdCopy)
                WHERE a.ctr_z_ir IS NULL
                SET a.ctr_z_industry = CASE WHEN sigma = 0 THEN 0 ELSE (a.simulated_ctr - mu)/sigma END
                """,
                """
                MATCH (i:Industries)<-[:Belongs_to]-(a:AdCopy)
                WITH i,
                     percentileCont(a.simulated_ctr, 0.75) AS q3,
                     percentileCont(a.simulated_ctr, 0.33) AS p33,
                     percentileCont(a.simulated_ctr, 0.66) AS p66
                MATCH (i)<-[:Belongs_to]-(a:AdCopy)
                WHERE a.ctr_topq_ir IS NULL
                SET a.ctr_topq_industry = a.simulated_ctr >= q3,
                    a.tier_industry =
                      CASE
                        WHEN a.simulated_ctr <  p33 THEN 'Low'
                        WHEN a.simulated_ctr <= p66 THEN 'Mid'
                        ELSE 'High'
                      END
                """,
                """
                MATCH (a:AdCopy)
                SET a.ctr_z_final    = coalesce(a.ctr_z_ir, a.ctr_z_industry),
                    a.ctr_topq_final = coalesce(a.ctr_topq_ir, a.ctr_topq_industry),
                    a.tier_final     = coalesce(a.tier_ir, a.tier_industry)
                """
            ]
            try:
                for b in blocks: run_cypher(b)
                st.success("IR metrics computed ✅")
            except Exception as e:
                st.error(f"Failed computing metrics: {e}")
