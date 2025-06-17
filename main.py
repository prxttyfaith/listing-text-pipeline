import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Load environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "5432")

# Create engine
DB_URL = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Test connection
def test_connection():
    try:
        with DB_URL.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Connection successful:", result.fetchone())
    except Exception as e:
        print("Connection failed:", e)
        
# load_data_from_db:
def load_data_from_db(sample_size: int = 5_000):
    print(f"Loading raw data (sample_size={sample_size})...")
    os.makedirs("raw_subset",    exist_ok=True)
    # sample the listings table
    sample_q = f"""
    SELECT id, shop_id, title
      FROM global.listings
      TABLESAMPLE SYSTEM (1)
    LIMIT {sample_size};
    """
    print("  - Fetching listings...")
    ilistings_df = pd.read_sql(sample_q, DB_URL)
    ilistings_df.to_csv("raw_subset/ilistings.csv", index=False)

    # filter listing_tags on just those listing_ids
    listing_ids = ilistings_df["id"].tolist()
    ltags_q = text("""
    SELECT shop_id, listing_id, tag_id
      FROM global.listing_tags
     WHERE listing_id = ANY(:ids)
    """)
    print("  - Fetching listing_tags...")
    ilisting_tags_df = pd.read_sql(ltags_q, DB_URL, params={"ids": listing_ids})
    ilisting_tags_df.to_csv("raw_subset/ilisting_tags.csv", index=False)

    # filter tags on just those tag_ids
    tag_ids = ilisting_tags_df["tag_id"].unique().tolist()
    tags_q = text("""
    SELECT id, name
      FROM global.tags
     WHERE id = ANY(:ids)
    """)
    print("  - Fetching tags...")
    itags_df = pd.read_sql(tags_q, DB_URL, params={"ids": tag_ids})
    itags_df.to_csv("raw_subset/itags.csv", index=False)

    return ilistings_df, ilisting_tags_df, itags_df

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("Cleaning listings...")
        df = df.dropna(subset=["title"])
        df["clean_title"] = (
            df["title"].astype(str)
            .str.lower()
            .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
            .str.strip()
        )
        df.drop_duplicates(subset=["title"])
        print("Listings cleaned successfully.")
        return df
    except Exception as e:
        print(f"Error cleaning listings: {e}")
        return df


def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("Cleaning tags...")
        df = df.dropna(subset=["name"])
        df["clean_name"] = (
            df["name"].astype(str)
               .str.lower()
               .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
               .str.strip()
        )
        df.drop_duplicates(subset=["name"])
        print("Tags cleaned successfully.")
        return df
    except Exception as e:
        print(f"Error cleaning tags: {e}")
        return df

def clean_listing_tags(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["shop_id", "listing_id", "tag_id"])

# listing with tags
def build_listing_with_tags(ilistings_clean_df, ilisting_tags_clean_df, itags_clean_df):
    try: 
        print("Building listing_with_tags DataFrame...")
        # prepare listings and tags for merge
        df_listings = ilistings_clean_df[["id", "shop_id", "clean_title"]].rename(
            columns={"id": "listing_id", "clean_title": "listing_title"}
        )
        df_tags = itags_clean_df[["id", "clean_name"]].rename(
            columns={"id": "tag_id", "clean_name": "tag_name"}
        )

        # merge into one “listing_with_tags” table
        listing_with_tags_df = (
            df_listings
            .merge(ilisting_tags_clean_df, on="listing_id", how="inner")
            .merge(df_tags,                 on="tag_id",       how="inner")
        )

        return listing_with_tags_df
    except Exception as e:
        print(f"Error building listing_with_tags DataFrame: {e}")
        return pd.DataFrame()

# pairwise features based on listing_with_tags_df
def create_pairwise_features(listing_with_tags_df, sample_n: int = 200):
    try:
        print(f"Creating pairwise features (sample_n={sample_n})...")
        # sample 200 unique listings
        sampled_ids = (
            listing_with_tags_df["listing_id"]
            .drop_duplicates()
            .sample(sample_n, random_state=42)
            .tolist()
        )
        sub = listing_with_tags_df[
            listing_with_tags_df.listing_id.isin(sampled_ids)
        ]

        # build lookups
        titles = (
            sub.drop_duplicates(subset=["listing_id"])
            .set_index("listing_id")["listing_title"]
            .loc[sampled_ids]
            .to_dict()
        )
        tagsets = sub.groupby("listing_id").tag_name.apply(set).to_dict()

        # TF-IDF + cosine
        tfidf = TfidfVectorizer().fit_transform([titles[i] for i in sampled_ids])
        cosmat = cosine_similarity(tfidf)

        rows = []
        for i, j in combinations(range(len(sampled_ids)), 2):
            id1, id2 = sampled_ids[i], sampled_ids[j]
            s1, s2   = tagsets[id1], tagsets[id2]
            shared   = s1 & s2
            union    = s1 | s2
            rows.append({
                "listing_id1":  id1,
                "listing_id2":  id2,
                "cosine_sim":   cosmat[i, j],
                "jaccard_sim":  (len(shared) / len(union)) if union else 0,
                "shared_tags":  len(shared),
                "name_len_diff": abs(len(titles[id1]) - len(titles[id2]))
            })

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error creating pairwise features: {e}")
        return pd.DataFrame()

def main():
    test_connection()
    
    # LOAD
    ilistings_df, ilisting_tags_df, itags_df = load_data_from_db(sample_size=5000)
 
    # CLEAN
    ilistings_clean_df     = clean_listings(ilistings_df)
    ilisting_tags_clean_df = clean_listing_tags(ilisting_tags_df)
    itags_clean_df         = clean_tags(itags_df)
    
    # listing with tags
    listing_with_tags_df = build_listing_with_tags(
        ilistings_clean_df,
        ilisting_tags_clean_df,
        itags_clean_df
    )
    
    # EXPORT FINAL .CSV
    os.makedirs("final",          exist_ok=True)
    listing_with_tags_df.to_csv("final/listing_with_tags.csv", index=False)
    print(f"Saved listing_with_tags.csv ({listing_with_tags_df.shape})")

    ilistings_clean_df    .to_csv("final/listings.csv",      index=False)
    ilisting_tags_clean_df.to_csv("final/listing_tags.csv", index=False)
    itags_clean_df        .to_csv("final/tags.csv",          index=False)
    print("Exported: listings.csv, listing_tags.csv, tags.csv")
    
    # pairwise features based on listing_with_tags_df
    pairs_df = create_pairwise_features(listing_with_tags_df, sample_n=200)
    pairs_df.to_csv("final/pairwise_features.csv", index=False)
    print(f" Saved pairwise_features.csv ({pairs_df.shape})")

if __name__ == "__main__":
    main()
