from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, DoubleType

def get_spark_session(app_name="AIG_RAG_Pipeline"):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )