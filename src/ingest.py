from pyspark.sql.types import ArrayType, DoubleType
from src.constants import EMBEDDING_MODEL_NAME

### GOAL: Load, split into chunks, generate embeddings.


# something like this? TODO: Scrap later?
@.pandas_udf(returnType=ArrayType(DoubleType()))
def embed(x: pd.Series) -> pd.Series:
    # Import and instantiate model inside the UDF
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = 256
    return pd.Series(model.encode(x.tolist(), batch_size=128).tolist())