#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid, pycountry, pandas as pd, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import shutil, datetime as dt

def preprocess_data():
    # Create directories if they don't exist
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/clean', exist_ok=True)
    os.makedirs('../data/import', exist_ok=True)
    os.makedirs('../data/import_bulk', exist_ok=True)
    
    df = pd.read_csv('../data/raw/synthetic_ad_data_relevant_combos.csv')

    df[df['simulated_conversion_rate']>100]

    ## EDA

    print("Descriptive Statistics for Simulated Performance Metrics:")
    print(df[['simulated_impressions', 'simulated_clicks', 'simulated_conversions',
              'simulated_cost', 'simulated_ctr', 'simulated_conversion_rate',
              'simulated_cpc', 'simulated_cpa']].describe())

    # 2. Visualize Distributions using Histograms
    numerical_cols_for_hist = [
        'simulated_impressions', 'simulated_clicks', 'simulated_conversions',
        'simulated_cost', 'simulated_ctr', 'simulated_conversion_rate'
    ]

    # Create subplots for histograms
    plt.figure(figsize=(18, 12))
    for i, col in enumerate(numerical_cols_for_hist):
        plt.subplot(2, 3, i + 1) # 2 rows, 3 columns
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.suptitle('Distributions of Key Simulated Performance Metrics', y=1.02, fontsize=16)
    plt.savefig('../data/clean/distributions_of_metrics.png')
    print("\nGenerated 'distributions_of_metrics.png' showing metric distributions.")

    # Next: Analyze performance across tiers.
    # Let's look at average CTR and Conversion Rate per performance tier
    print("\nAverage CTR and Conversion Rate by Performance Tier:")
    tier_performance = df.groupby('performance_tier')[['simulated_ctr', 'simulated_conversion_rate', 'simulated_impressions', 'simulated_clicks', 'simulated_conversions', 'simulated_cost']].mean().sort_values(by='simulated_ctr', ascending=False)
    print(tier_performance)

    # Visualize Performance by Tier (e.g., CTR by Tier)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='performance_tier', y='simulated_ctr', data=df, order=['high', 'mid', 'low'])
    plt.title('Simulated CTR by Performance Tier')
    plt.xlabel('Performance Tier')
    plt.ylabel('Simulated CTR (%)')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('../data/clean/ctr_by_performance_tier.png')
    print("\nGenerated 'ctr_by_performance_tier.png' showing CTR distribution per tier.")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='performance_tier', y='simulated_conversion_rate', data=df, order=['high', 'mid', 'low'])
    plt.title('Simulated Conversion Rate by Performance Tier')
    plt.xlabel('Performance Tier')
    plt.ylabel('Simulated Conversion Rate (%)')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('../data/clean/conversion_rate_by_performance_tier.png')
    print("\nGenerated 'conversion_rate_by_performance_tier.png' showing Conversion Rate distribution per tier.")


    RAW_CSV   = Path("../data/raw/synthetic_ad_data_relevant_combos.csv")
    CLEAN_PAR = Path("../data/clean/ads_clean.parquet")
    IMP_DIR   = Path("../data/import")
    IMP_DIR.mkdir(parents=True, exist_ok=True)

    ## Dedup

    # Cell 2 – Load & drop duplicates (extended)
    df = pd.read_csv(RAW_CSV)

    TEXT = ["headline_1", "headline_2",
            "description_1", "description_2", "description_3"]

    # Make NaN comparable
    df[TEXT] = df[TEXT].fillna("")

    df = df.drop_duplicates(subset=TEXT)

    print(f"After duplicates drop: {len(df):,} rows remain")
    df[TEXT].head(3)


    df = df[df["simulated_ctr"].between(0.1, 10)]
    df.describe(include="number").T

    ## Harmonise industry and region

    import uuid, re, pycountry

    # 1️⃣  INDUSTRY canonicalisation
    # ----------------------------------------------------------------
    # Keys      = lowercase variants you might meet in raw data
    # Values    = canonical label that will become the 1-per-node value
    industry_map = {
        # Fitness / Gym
        "gym": "Gym",
        "fitness studio": "Gym",
        "fitness-studio": "Gym",
        "fitness centre": "Gym",
        # Hair / Barber
        "hair salon": "Hair Salon",
        "barber": "Hair Salon",
        "barber shop": "Hair Salon",
        # Roofing
        "roofing company": "Roofing Company",
        "roofer": "Roofing Company",
        # Fashion e-commerce
        "online clothing boutique": "Online Clothing Boutique",
        "clothing store": "Online Clothing Boutique",
        "fashion boutique": "Online Clothing Boutique",
        # Italian food
        "italian restaurant": "Italian Restaurant",
        "pizzeria": "Italian Restaurant",
        # Marketing agencies
        "digital marketing agency": "Digital Marketing Agency",
        "marketing agency": "Digital Marketing Agency",
        "marketing firm": "Digital Marketing Agency",
        "ad agency": "Digital Marketing Agency",
        # Coffee / Café
        "local coffee shop": "Local Coffee Shop",
        "coffee shop": "Local Coffee Shop",
        "cafe": "Local Coffee Shop",
        # Plumbing
        "plumbing services": "Plumbing Services",
        "plumber": "Plumbing Services",
        # Yoga
        "yoga studio": "Yoga Studio",
        "yoga centre": "Yoga Studio",
        # Pets
        "pet grooming": "Pet Grooming",
        "pet grooming salon": "Pet Grooming",
        "pet groomer": "Pet Grooming",
    }

    def canon_ind(val: str) -> str:
        """Lower-case + strip + map via industry_map; fall back to Title Case."""
        key = val.strip().lower()
        return industry_map.get(key, val.strip().title())

    df["industry"] = df["industry"].map(canon_ind)

    # 2️⃣  REGION  → ISO-3166 alpha-2
    # ----------------------------------------------------------------
    # Helper dict for quick look-up of country synonyms not in pycountry.
    country_alias = {
        "usa": "US",
        "u.s.a.": "US",
        "us": "US",
        "uk": "GB",
        "u.k.": "GB",
    }

    def to_iso_alpha2(region_val: str) -> str:
        """
        Extract the country part (everything after the last comma);
        map it to an ISO-3166-1 alpha-2 code using pycountry or aliases.
        """
        if not isinstance(region_val, str):
            return "XX"                       # unknown
        # take text after last comma → country fragment
        country_fragment = region_val.split(",")[-1].strip()
        key = country_fragment.lower()
        # 1) alias dict
        if key in country_alias:
            return country_alias[key]
        # 2) pycountry lookup
        try:
            return pycountry.countries.lookup(country_fragment).alpha_2
        except LookupError:
            return country_fragment.upper()[:2]   # crude fallback

    df["region_code"] = df["region"].map(to_iso_alpha2)

    # 3️⃣  ADD STABLE UUIDs
    # ----------------------------------------------------------------
    df["ad_id"] = [uuid.uuid4().hex for _ in range(len(df))]
    assert df["ad_id"].is_unique

    print("✅  Harmonisation done — unique industries:", df["industry"].nunique(),
          "| regions:", df["region_code"].unique())
    df.head(3)[["ad_id","industry","region","region_code"]]

    ## Persist the clean snapshot

    from pathlib import Path
    CLEAN_PAR = Path("../data/clean/ads_clean.parquet")
    CLEAN_PAR.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(CLEAN_PAR, index=False)
    print("✅  Clean Parquet stored →", CLEAN_PAR)

    ## Build node tables

    IMP_DIR = Path("../data/import")
    IMP_DIR.mkdir(parents=True, exist_ok=True)

    # Industry nodes
    industries = (df[["industry"]].drop_duplicates()
                  .rename(columns={"industry": "name"}))
    industries["ind_id"] = industries["name"].factorize()[0] + 1
    industries[["ind_id","name"]].to_csv(IMP_DIR/"Industry_nodes.csv", index=False)

    # Region nodes
    regions = (df[["region_code","region"]]
               .drop_duplicates()
               .rename(columns={"region_code":"code","region":"name"}))
    regions.to_csv(IMP_DIR/"Region_nodes.csv", index=False)

    # AdCopy nodes (keep *all* creative text)
    ad_cols = ["ad_id",
               "headline_1","headline_2",
               "description_1","description_2","description_3",
               "simulated_ctr","simulated_cpc","simulated_cpa"]
    df[ad_cols].to_csv(IMP_DIR/"AdCopy_nodes.csv", index=False)

    print("✅  Node CSVs written in", IMP_DIR)

    industries

    regions

    ## Build relationship tables

    # BELONGS_TO (AdCopy → Industry)
    bel = df.merge(industries, left_on="industry", right_on="name")
    bel[["ad_id","ind_id"]].to_csv(
        IMP_DIR/"BELONGS_TO_rels.csv", index=False,
        header=["ad_id:START_ID(AdCopy)", "ind_id:END_ID(Industry)"])

    # TARGETS (AdCopy → Region)
    tar = df.merge(regions, left_on="region_code", right_on="code")
    tar[["ad_id","code"]].to_csv(
        IMP_DIR/"TARGETS_rels.csv", index=False,
        header=["ad_id:START_ID(AdCopy)", "code:END_ID(Region)"])

    print("✅  Relationship CSVs written")

    ## Sanity checks

    print("Rows per industry:")
    display(df["industry"].value_counts())

    # Counts must match between Parquet and node CSV
    assert len(df) == pd.read_csv(IMP_DIR/"AdCopy_nodes.csv").shape[0]
    print("✔ Counts consistent!")

    ## Zip the import folder

    import shutil, datetime as dt
    zip_path = f"../data/ads_import_{dt.date.today()}.zip"
    shutil.make_archive(zip_path.replace(".zip",""), "zip", root_dir=IMP_DIR)
    print("📦  Zipped import folder →", zip_path)

    # Create bulk-import versions with typed ID headers
    import pandas as pd
    from pathlib import Path

    SRC = Path("../data/import")
    DST = Path("../data/import_bulk")
    DST.mkdir(parents=True, exist_ok=True)

    # --- Nodes -----------------------------------------------------
    ad = pd.read_csv(SRC / "AdCopy_nodes.csv")
    # Type numeric columns explicitly (helps LOAD CSV too)
    for col in ["simulated_ctr", "simulated_cpc", "simulated_cpa"]:
        if col in ad.columns:
            ad[col] = pd.to_numeric(ad[col], errors="coerce")

    ad = ad.rename(columns={"ad_id": "ad_id:ID(AdCopy)"})
    ad.to_csv(DST / "AdCopy_nodes.csv", index=False)

    ind = pd.read_csv(SRC / "Industry_nodes.csv").rename(
        columns={"ind_id": "ind_id:ID(Industry)"}
    )
    ind.to_csv(DST / "Industry_nodes.csv", index=False)

    reg = pd.read_csv(SRC / "Region_nodes.csv").rename(
        columns={"code": "code:ID(Region)"}
    )
    reg.to_csv(DST / "Region_nodes.csv", index=False)

    # --- Relationships ---------------------------------------------
    bel = pd.read_csv(SRC / "BELONGS_TO_rels.csv")
    bel.columns = ["ad_id:START_ID(AdCopy)", "ind_id:END_ID(Industry)"]
    bel.to_csv(DST / "BELONGS_TO_rels.csv", index=False)

    tar = pd.read_csv(SRC / "TARGETS_rels.csv")
    tar.columns = ["ad_id:START_ID(AdCopy)", "code:END_ID(Region)"]
    tar.to_csv(DST / "TARGETS_rels.csv", index=False)

    print("✅ Bulk-import files written to data/import_bulk/")

    import pandas as pd
    from pathlib import Path

    CLEAN = Path("../data/clean/ads_clean.parquet")
    DST   = Path("../data/import_bulk")   # write alongside the bulk files

    df_kw = pd.read_parquet(CLEAN)[["ad_id", "keywords"]].copy()
    kw = (df_kw.assign(keywords=df_kw["keywords"].fillna("").str.split(";"))
                  .explode("keywords")
                  .assign(keywords=lambda d: d["keywords"].str.strip().str.lower())
                  .query("keywords != ''"))

    # Keyword nodes
    kw_nodes = kw[["keywords"]].drop_duplicates().rename(columns={"keywords":"lemma"})
    kw_nodes.to_csv(DST/"Keyword_nodes.csv", index=False)

    # MENTIONS relationships
    kw_rels = kw.merge(kw_nodes, left_on="keywords", right_on="lemma")
    kw_rels[["ad_id","lemma"]].to_csv(
        DST/"MENTIONS_rels.csv", index=False,
        header=["ad_id:START_ID(AdCopy)", "lemma:END_ID(Keyword)"])

    print("✅ Keyword_nodes.csv and MENTIONS_rels.csv written")

    ## Quick Integrity Check for IDS

    import pandas as pd
    from pathlib import Path

    bulk = Path("../data/import_bulk")

    ad = pd.read_csv(bulk/"AdCopy_nodes.csv")["ad_id:ID(AdCopy)"]
    bel = pd.read_csv(bulk/"BELONGS_TO_rels.csv")["ad_id:START_ID(AdCopy)"]
    tar = pd.read_csv(bulk/"TARGETS_rels.csv")["ad_id:START_ID(AdCopy)"]

    # Every relationship must reference an existing ad_id
    assert set(bel).issubset(set(ad)), "BELONGS_TO has unknown ad_id(s)"
    assert set(tar).issubset(set(ad)), "TARGETS has unknown ad_id(s)"
    print("✅ All relationship ad_ids resolve to AdCopy nodes")

if __name__ == "__main__":
    preprocess_data()
