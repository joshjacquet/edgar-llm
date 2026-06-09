from pyspark.sql.types import ArrayType, DoubleType

# something like this? TODO: Scrap later?
@F.pandas_udf(returnType=ArrayType(DoubleType()))
def mpnet_encode(x: pd.Series) -> pd.Series:
    # Import and instantiate model inside the UDF
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    model.max_seq_length = 256
    return pd.Series(model.encode(x.tolist(), batch_size=128).tolist())