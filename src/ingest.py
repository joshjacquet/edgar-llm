import os

os.environ["HF_HUB_OFFLINE"] = "1"  # this is a local machine networking workaround...

import datasets
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, explode, lit
from pyspark.sql.types import ArrayType, StringType
from sentence_transformers import SentenceTransformer
from config import get_spark_session
from constants import EMBEDDING_MODEL_NAME, CIK, YEAR, OUTPUT_PATH


def fetch_target_10k(
    spark: SparkSession = get_spark_session(), year: str = YEAR, cik: str = CIK
) -> DataFrame:
    """
    Return a Spark dataframe containing the specified company's 10k from a given year.
    """
    dataset = datasets.load_dataset(
        path="eloukas/edgar-corpus",
        name=f"year_{year}",
        split="train",
        trust_remote_code=True,
    ).filter(lambda x: x["cik"] == cik)

    return spark.createDataFrame(dataset)


def split_10k_sections(
    df: DataFrame,
    sections: list = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> DataFrame:
    """
    Split the specified section columns of a 10K filing DataFrame into chunks using LangChain TextSplitter.
    Returns a new Spark DataFrame containing: cik, filename, year, section, and chunk_text.
    """

    # Identify target sections (all columns starting with section_)
    if sections is None:
        sections = [c for c in df.columns if c.startswith("section_")]

    # Define UDF for text splitting to avoid PySpark serialization errors
    @udf(returnType=ArrayType(StringType()))
    def split_text_udf(text):
        if not text:
            return []
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    split_dfs = []
    for section in sections:
        if section in df.columns:
            # Select key metadata columns and the specific section text
            sec_df = df.select(
                "cik", "filename", "year", col(section).alias("section_text")
            ).withColumn("section", lit(section))

            # Apply UDF to split the text
            sec_split = sec_df.withColumn("splits", split_text_udf("section_text"))

            # Explode the splits into individual rows
            sec_exploded = sec_split.withColumn("chunk_text", explode("splits"))

            # Select final columns matching our schema
            sec_final = sec_exploded.select(
                "cik", "filename", "year", "section", "chunk_text"
            )
            split_dfs.append(sec_final)

    if not split_dfs:
        raise ValueError("No sections found to split.")

    # Union all section DataFrames
    result_df = split_dfs[0]
    for next_df in split_dfs[1:]:
        result_df = result_df.union(next_df)

    return result_df


def generate_embeddings(
    df: DataFrame, column: str, embeddings_model: str = EMBEDDING_MODEL_NAME
) -> DataFrame:
    """
    Take a Spark DataFrame, add a field containing embeddings.
    """

    # Collecting everything back to the driver due to local spark worker issues generating embeddings
    # Generate embeddings then load back to spark dataframe
    pd_df = df.toPandas()
    model = SentenceTransformer(embeddings_model)
    texts = pd_df[column].fillna("").tolist()
    embeddings = model.encode(texts)
    pd_df["embeddings"] = [emb.tolist() for emb in embeddings]

    spark = df.sparkSession
    return spark.createDataFrame(pd_df)


if __name__ == "__main__":
    import pandas as pd
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Fetch 10-K and collect to driver
    df = fetch_target_10k()
    raw = df.toPandas()

    # Split each section into chunks on the driver (Spark workers are unreliable locally)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    sections = [c for c in raw.columns if c.startswith("section_")]

    rows = []
    for section in sections:
        text = raw[section].iloc[0]
        if not text:
            continue
        print(f"Splitting {section}...")
        for chunk in splitter.split_text(text):
            rows.append({
                "cik": raw["cik"].iloc[0],
                "filename": raw["filename"].iloc[0],
                "year": raw["year"].iloc[0],
                "section": section,
                "chunk_text": chunk,
            })

    combined = pd.DataFrame(rows)

    # Generate embeddings in one pass
    print(f"Generating embeddings for {len(combined)} chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = combined["chunk_text"].fillna("").tolist()
    embeddings = model.encode(texts)
    combined["embeddings"] = [emb.tolist() for emb in embeddings]

    combined.to_json(os.path.join(OUTPUT_PATH, "embedded_chunks.json"), orient="records", indent=2)
    print(f"Wrote {len(combined)} chunks to {OUTPUT_PATH}/embedded_chunks.json")
