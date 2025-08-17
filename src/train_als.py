from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

# Start Spark
spark = SparkSession.builder.appName("ALS_Recommendation").getOrCreate()

# Load the prepared parquet with indexes already present
user_interactions = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"
)

# Check columns (important step)
print("Schema:", user_interactions.printSchema())
print("Sample rows:")
user_interactions.show(5, truncate=False)


# If you only have user_id and song_id (not indexed yet) → create indexes
if "userIndex" not in user_interactions.columns or "songIndex" not in user_interactions.columns:
    user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex", handleInvalid="skip")
    song_indexer = StringIndexer(inputCol="song_id", outputCol="songIndex", handleInvalid="skip")

    interactions = user_indexer.fit(user_interactions).transform(user_interactions)
    interactions = song_indexer.fit(interactions).transform(interactions)
else:
    interactions = user_interactions


# Train ALS model
als = ALS(
    userCol="userIndex",
    itemCol="songIndex",
    ratingCol="interaction",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True
)

model = als.fit(interactions)

# Generate top 10 recommendations for each user
user_recs = model.recommendForAllUsers(10)

# Explode recommendations into separate rows
from pyspark.sql.functions import explode
user_recs = user_recs.withColumn("rec", explode(col("recommendations"))).select(
    col("userIndex").alias('user_id'),
    col("rec.songIndex").alias("song_id"),
    col("rec.rating").alias("rating")
)

# Save recommendations as Parquet
output_path = r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/als_recommendations.parquet"
user_recs.write.mode("overwrite").parquet(output_path)

print(f"Recommendations saved to: {output_path}")

# Optional: Preview saved recommendations
user_recs.show(10, truncate=False)
