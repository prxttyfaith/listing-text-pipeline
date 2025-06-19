import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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
def load_data_from_db():
    print(f"Loading raw data ...")
    os.makedirs("raw_subset",    exist_ok=True)
    # sample the listings table
    sample_q = f"""
    SELECT id, shop_id, title
      FROM interns.ilistings_clean
    """
    print("  - Fetching ilistings_clean...")
    ilistings_df = pd.read_sql(sample_q, DB_URL)
    ilistings_df.to_csv("raw_subset/ilistings.csv", index=False)
    # # filter listing_tags on just those listing_ids
    listing_ids = ilistings_df["id"].tolist()
    
    ltags_q = text("""
    SELECT shop_id, listing_id, tag_id
      FROM interns.ilisting_tags_clean
    """)
    print("  - Fetching ilisting_tags_clean...")
    ilisting_tags_df = pd.read_sql(ltags_q, DB_URL, params={"ids": listing_ids})
    ilisting_tags_df.to_csv("raw_subset/ilisting_tags.csv", index=False)

    # # filter tags on just those tag_ids
    print("  - Fetching itags_clean")
    tags_q = text("""
    SELECT id, name
      FROM interns.itags_clean
    """)
    itags_df = pd.read_sql(tags_q, DB_URL)
    itags_df.to_csv("raw_subset/itags.csv", index=False)

    # return ilistings_df
    return ilistings_df, ilisting_tags_df, itags_df

# listing with tags
def build_listing_with_tags(ilistings_clean_df, ilisting_tags_clean_df, itags_clean_df):
    try: 
        print("Building listing_with_tags DataFrame...")
        # prepare listings and tags for merge
        df_listings = ilistings_clean_df[["id", "shop_id", "title"]].rename(
            columns={"id": "listing_id", "title": "listing_title"}
        )
        df_tags = itags_clean_df[["id", "name"]].rename(
            columns={"id": "tag_id", "name": "tag_name"}
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

    
def split_and_label(pairs_df, cos_threshold=0.5, jac_threshold=0.3, random_state=42):

    # Label = 1 if (cosine_sim >= cos_threshold) AND (jaccard_sim >= jac_threshold),
    # then stratified split into train (60%), val (20%), test (20%).

    df = pairs_df.copy()
    df['label'] = (
        (df['cosine_sim'] >= cos_threshold) &
        (df['jaccard_sim'] >= jac_threshold)
    ).astype(int)

    # 20% hold-out for test
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=df['label']
    )
    # of the remaining 80%, 25% → val (so val=20% total, train=60% total)
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=random_state, stratify=train_val['label']
    )
    return train, val, test

def main():
    test_connection()
    
    # load
    ilistings_clean_df, ilisting_tags_clean_df, itags_clean_df = load_data_from_db()

    # create listing with tags
    listing_with_tags_df = build_listing_with_tags(
        ilistings_clean_df,
        ilisting_tags_clean_df,
        itags_clean_df
    )
    # export final csv files
    os.makedirs("final",          exist_ok=True)
    listing_with_tags_df.to_csv("final/listing_with_tags.csv", index=False)
    print(f"Saved listing_with_tags.csv ({listing_with_tags_df.shape})")

    # pairwise features based on listing_with_tags_df
    pairs_df = create_pairwise_features(listing_with_tags_df, sample_n=200)
    pairs_df.to_csv("final/pairwise_features.csv", index=False)
    print(f" Saved pairwise_features.csv ({pairs_df.shape})") 
    
    # ilistings_clean_df.to_csv("final/listings.csv", index=False)
    # ilisting_tags_clean_df.to_csv("final/listing_tags.csv", index=False)
    # itags_clean_df.to_csv("final/tags.csv",          index=False)
    # print("Exported: listings.csv, listing_tags.csv, tags.csv")

if __name__ == "__main__":
    main()
