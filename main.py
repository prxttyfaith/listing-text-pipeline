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
# - remove null, lowercase, remove punctuations and special characters, remove stop words, remove duplicates title/name



    
def main():
    test_connection()
    ilistings_df, ilisting_tags_df, itags_df = load_data_from_db(sample_size=5000)

    # ilistings_df, itags_df, ilisting_tags_df = load_data_from_db()
    # clean data. pass df to functions 
    
    

if __name__ == "__main__":
    main()
