import os

os.environ["HF_HUB_OFFLINE"] = "1"  # this is a local machine networking workaround...

import datasets
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, explode, lit
from pyspark.sql.types import ArrayType, StringType
from sentence_transformers import SentenceTransformer
from config import get_spark_session
from constants import EMBEDDING_MODEL_NAME, CIK, YEAR


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
    df = fetch_target_10k()
    df_split = split_10k_sections(df, sections=["section_1"])
    df_embedded = generate_embeddings(df_split, "chunk_text")
    print(df_embedded.head(5))
