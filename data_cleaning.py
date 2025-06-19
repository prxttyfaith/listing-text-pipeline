import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except:
        return False

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
    
    sample_q = f"""
    SELECT id, shop_id, title
      FROM interns.ilistings
    """
    print("  - Fetching listings...")
    ilistings_df = pd.read_sql(sample_q, DB_URL)
    ilistings_df.to_csv("raw_subset/ilistings.csv", index=False)

    # # filter listing_tags on just those listing_ids
    listing_ids = ilistings_df["id"].tolist()
    
    ltags_q = text("""
    SELECT shop_id, listing_id, tag_id
      FROM interns.ilisting_tags_clean
    """)
    print("  - Fetching listing_tags...")
    ilisting_tags_df = pd.read_sql(ltags_q, DB_URL, params={"ids": listing_ids})
    ilisting_tags_df.to_csv("raw_subset/ilisting_tags.csv", index=False)

    # # filter tags on just those tag_ids
    print("  - Fetching tags via JOIN on cleaned listing_tags")
    tags_q = text("""
    SELECT DISTINCT
      t.id,
      t.name
    FROM interns.itags AS t
    JOIN interns.ilisting_tags_clean AS ltc
      ON t.id = ltc.tag_id
    """)
    itags_df = pd.read_sql(tags_q, DB_URL)
    itags_df.to_csv("raw_subset/itags.csv", index=False)

    # return ilistings_df
    return ilistings_df, ilisting_tags_df, itags_df

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    STOP_WORDS = {"a","an","and","the","in","on","for","with","to","of","is","it","this","that","as","at"}
    try:
        print(" - Cleaning listings...")
        print(f"    rows: {len(df)}")     
        # 1) Drop real NaNs
        print("     > dropping NaN and empty titles")
        df = df.dropna(subset=["title"])
        df = df[df["title"].astype(str).str.strip() != ""].copy()
        print(f"    rows: {len(df)}")    
        print("     > normalizing titles -> clean_title")
        df["clean_title"] = (
            df["title"].astype(str)
            .str.lower()
            .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .apply(lambda txt: ' '.join(word for word in txt.split() if word not in STOP_WORDS))
        )
        # !!!! DROP AGAIN THE EMPTY CLEANED TITLE
        print("     > dropping empty cleaned_titles")
        df = df[df["clean_title"].astype(str).str.strip() != ""].copy()
        print(f"    rows: {len(df)}")    

        #language filter
        print("     > Filtering for English titles…")
        df = df[df["clean_title"].apply(is_english)]
        print(f"    rows: {len(df)}")     
        
        # 2) Drop duplicates based on the cleaned title
        print("     > dropping duplicates based on clean_title")
        df = df.drop_duplicates(subset=['clean_title'], keep='first')
        print(f"    rows: {len(df)}")    
        
        print("Listings cleaned successfully.")
        return df
    except Exception as e:
        print(f"Error cleaning listings: {e}")
        return df

def save_clean_listings_to_db(clean_df: pd.DataFrame):
    # select & rename
    df_to_load = (
        clean_df
        .loc[:, ["id", "shop_id", "clean_title"]]
        .rename(columns={"clean_title": "title"})
    )

    # write to Postgres
    df_to_load.to_sql(
        name="ilistings_clean",
        schema="interns",
        con=DB_URL,
        index=False
    )

    # (optional) add a primary key or index for performance
    with DB_URL.connect() as conn:
        conn.execute(text("""
            ALTER TABLE interns.ilistings_clean
            ADD PRIMARY KEY (id);
        """))
    print("✅ interns.ilistings_clean written to database.")

def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    STOP_WORDS = {"a","an","and","the","in","on","for","with","to","of","is","it","this","that","as","at"}
    try:
        print(" - Cleaning tags...")
        print(f"    rows: {len(df)}")
        
        # 1) Drop NaNs and empty names
        print("     > dropping NaN and empty names")
        df = df.dropna(subset=["name"])
        df = df[df["name"].astype(str).str.strip() != ""].copy()
        print(f"    rows: {len(df)}")
        
        # 2) Normalize name → clean_name
        print("     > normalizing names → clean_name")
        df["clean_name"] = (
            df["name"].astype(str)
              .str.lower()
              .str.replace(r"[^A-Za-z0-9\s]", "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
              .apply(lambda txt: " ".join(w for w in txt.split() if w not in STOP_WORDS))
        )
        
        # 3) Drop any rows where clean_name became empty
        print("     > dropping empty cleaned names")
        df = df[df["clean_name"].astype(str).str.strip() != ""].copy()
        print(f"    rows: {len(df)}")
        
        # 4) Language filter
        print("     > Filtering for English names…")
        df = df[df["clean_name"].apply(is_english)]
        print(f"    rows: {len(df)}")
        
        # 5) Drop duplicates, keep first
        print("     > dropping duplicates based on clean_name")
        df = df.drop_duplicates(subset=["clean_name"], keep="first")
        print(f"    rows: {len(df)}")
        
        print("Tags cleaned successfully.")
        return df

    except Exception as e:
        print(f"Error cleaning tags: {e}")
        return df

def save_clean_tags_to_db(clean_df: pd.DataFrame):
    # select & rename so that clean_name → name
    df_to_load = (
        clean_df
        .loc[:, ["id", "clean_name"]]
        .rename(columns={"clean_name": "name"})
    )

    # write to Postgres
    df_to_load.to_sql(
        name="itags_clean",
        schema="interns",
        con=DB_URL,
        index=False
    )

    # (optional) add a primary key for speed
    with DB_URL.connect() as conn:
        conn.execute(text("""
            ALTER TABLE interns.itags_clean
            ADD PRIMARY KEY (id);
        """))
    print("✅ interns.itags_clean written to database.")

def clean_listing_tags(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["shop_id", "listing_id", "tag_id"])


def main():
    test_connection()
    
    # LOAD
    ilistings_df, ilisting_tags_df, itags_df = load_data_from_db()
 
    # CLEAN LISTINGS
    # ilistings_clean_df     = clean_listings(ilistings_df)
    # save_clean_listings_to_db(ilistings_clean_df)
    
    # ilisting_tags_clean_df = clean_listing_tags(ilisting_tags_df)
    
    # CLEAN TAGS
    # itags_clean_df         = clean_tags(itags_df)
    # save_clean_tags_to_db(itags_clean_df)
    
    # # # EXPORT FINAL .CSV
    # os.makedirs("final",          exist_ok=True)
    # ilistings_clean_df.to_csv("final/listings.csv", index=False)
    # ilisting_tags_clean_df.to_csv("final/listing_tags.csv", index=False)
    # itags_clean_df.to_csv("final/tags.csv",          index=False)
    # print("Exported: listings.csv, listing_tags.csv, tags.csv")
    

if __name__ == "__main__":
    main()
