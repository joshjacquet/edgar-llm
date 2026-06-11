from pyspark.sql.functions import col, split, explode, monotonicity_increasing_id, instr

# GOAL: From chunks, retrieve relevant data based on metric

def retrieve_context_chunks(df_spark, target_section, keyword):
    """
    Splits a massive section column into paragraphs and retrieves 
    the specific text chunk containing the keyword.
    """
    # 1. Break the massive block string down by double-newlines (paragraphs)
    df_split = df_spark.withColumn("paragraphs", split(col(target_section), "\n\n"))
    
    # 2. Explode the array column so every paragraph becomes an independent Row
    df_exploded = df_split.withColumn("chunk_text", explode(col("paragraphs")))
    
    # 3. Add a unique row identifier to represent "virtual pages/blocks"
    df_indexed = df_exploded.withColumn("chunk_id", monotonicity_increasing_id())
    
    # 4. Filter strictly down to chunks matching your retrieval query
    df_matches = df_indexed.filter(instr(col("chunk_text"), keyword) > 0)
    
    return df_matches.select("chunk_id", "chunk_text")

if __name__ == '__main__':
    from pyspark.sql import SparkSession
    import os
    import sys

    # Basic environment variable setup; force 
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName("EDGAR_Retrieval") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.