import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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

    # sample sample_size listings on the Postgres side (no full table scan in Python)
    # pull only the matching listing_tags for those listings
    # pull only the matching tags for those tags

    # sample the listings table
    sample_q = f"""
    SELECT id, shop_id, title
      FROM global.listings
      TABLESAMPLE SYSTEM (1)
    LIMIT {sample_size};
    """
    ilistings_df = pd.read_sql(sample_q, DB_URL)

    # filter listing_tags on just those listing_ids
    listing_ids = ilistings_df["id"].tolist()
    ltags_q = text("""
    SELECT shop_id, listing_id, tag_id
      FROM global.listing_tags
     WHERE listing_id = ANY(:ids)
    """)
    ilisting_tags_df = pd.read_sql(ltags_q, DB_URL, params={"ids": listing_ids})

    # filter tags on just those tag_ids
    tag_ids = ilisting_tags_df["tag_id"].unique().tolist()
    tags_q = text("""
    SELECT id, name
      FROM global.tags
     WHERE id = ANY(:ids)
    """)
    itags_df = pd.read_sql(tags_q, DB_URL, params={"ids": tag_ids})

    return ilistings_df, ilisting_tags_df, itags_df


# def clean_data (df):
# - remove null, lowercase, remove punctuations and special characters, remove stop words, remove duplicates based on title/name

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title"])
    df["clean_title"] = (
        df["title"].astype(str)
           .str.lower()
           .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
           .str.strip()
    )
    return df.drop_duplicates(subset=["id"])


def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["name"])
    df["clean_name"] = (
        df["name"].astype(str)
           .str.lower()
           .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
           .str.strip()
    )
    return df.drop_duplicates(subset=["id"])

def clean_listing_tags(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["shop_id", "listing_id", "tag_id"])


# From your listing with tags, create a subset of 200 random entries.  (200 * 199) / 2 pairs

    
def main():
    test_connection()
    
    # load subset data from the database
    ilistings_df, ilisting_tags_df, itags_df = load_data_from_db(sample_size=5000)

    ilistings_df .to_csv("raw_subset/ilistings.csv",         index=False)
    ilisting_tags_df.to_csv("raw_subset/ilisting_tags.csv", index=False)
    itags_df     .to_csv("raw_subste/itags.csv",             index=False)
    print(f"📦 Raw saved: ilistings={ilistings_df.shape}, ilisting_tags={ilisting_tags_df.shape}, itags={itags_df.shape}")

    # clean data. pass df to functions 
    ilistings_clean_df     = clean_listings(ilistings_df)
    ilisting_tags_clean_df = clean_listing_tags(ilisting_tags_df)
    itags_clean_df         = clean_tags(itags_df)

    ilistings_clean_df     .to_csv("cleaned_subset/ilistings.csv",         index=False)
    ilisting_tags_clean_df .to_csv("cleaned_subset/ilisting_tags.csv", index=False)
    itags_clean_df         .to_csv("cleaned_subset/itags.csv",         index=False)
    
    # listing with tags
    df_l = ilistings_clean_df[["id","shop_id","clean_title"]].rename(
        columns={"id":"listing_id","clean_title":"listing_title"}
    )
    df_t = itags_clean_df[["id","clean_name"]].rename(
        columns={"id":"tag_id","clean_name":"tag_name"}
    )
    listing_with_tags = (
        df_l
        .merge(ilisting_tags_clean_df, on="listing_id", how="inner")
        .merge(df_t,                   on="tag_id",       how="inner")
    )
    # final exports
    listing_with_tags.to_csv("final/listing_with_tags.csv", index=False)
    print(f"🗂 listing_with_tags saved: {listing_with_tags.shape}")

    ilistings_clean_df    .to_csv("final/listings.csv",      index=False)
    ilisting_tags_clean_df.to_csv("final/listing_tags.csv", index=False)
    itags_clean_df        .to_csv("final/tags.csv",          index=False)
    print(" Exported: listings.csv, listing_tags.csv, tags.csv")

if __name__ == "__main__":
    main()
