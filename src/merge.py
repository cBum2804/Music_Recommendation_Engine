from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, col
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Window
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("Hybrid_system").getOrCreate()
als_df = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/als_recommendations.parquet"
)
content_df = spark.read.csv(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/top_recommendations_with_scores.csv",
    header=True,
    inferSchema=True
)


# Normalize ALS
als_min, als_max = als_df.agg(min("rating"), max("rating")).first()
als_df = als_df.withColumn("als_score_norm", 
    (col("rating") - als_min) / (als_max - als_min)
)

# Normalize Content-Based
content_min, content_max = content_df.agg(min("cosine_similarity"), max("cosine_similarity")).first()
content_df = content_df.withColumn("content_score_norm", 
    (col("cosine_similarity") - content_min) / (content_max - content_min)
)

hybrid_df = als_df.join(
    content_df,
    on=["user_id", "song_id"],
    how="outer"
).fillna(0)  # Fill missing scores with 0


weight_als = 0.6
weight_content = 0.4

hybrid_df = hybrid_df.withColumn(
    "hybrid_score",
    (col("als_score_norm") * weight_als) + (col("content_score_norm") * weight_content)
)

# Mapping song_id with name
# Load your original user-song interaction data
songs_meta_df = (
    spark.read.csv(
        "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/data/common_song.csv",
        header=True,
        inferSchema=True
    )
    .select("song_id", "artist_name", "track_name")
)


hybrid_df = hybrid_df.join(
    songs_meta_df, on="song_id", how="left"
)



windowSpec = Window.partitionBy("user_id").orderBy(F.desc("hybrid_score"))

final_hybrid = (
    hybrid_df
    .withColumn("rank", F.row_number().over(windowSpec))
    .filter(F.col("rank") <= 10)
)
output_path = r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/hybrid_recommendations.parquet"
final_hybrid.write.mode("overwrite").parquet(output_path)

final_hybrid.show(10, truncate = False)
