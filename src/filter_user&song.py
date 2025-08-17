from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, when
from pyspark.ml.feature import StringIndexer
import os

# Hadoop setup
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] += os.pathsep + "C:/hadoop/bin"

# Spark session
spark = SparkSession.builder.appName("SpotifyRecommendation").getOrCreate()
spark.conf.set("spark.hadoop.io.native.lib.available", "false")

# Read CSV
df = spark.read.csv(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/data/updated_user_data.csv",
    header=True,
    inferSchema=True
)

# Add interaction flag
df = df.withColumn("interaction", when(col("listen_count") >= 1, 1).otherwise(0))

# Filter users/songs with very few interactions
min_song_interactions = 5
min_user_interactions = 5

user_interactions = df.groupBy("user_id").agg(count("song_id").alias("interaction_count"))
song_interactions = df.groupBy("song_id").agg(count("user_id").alias("interaction_count"))

active_users = user_interactions.filter(col("interaction_count") >= min_user_interactions)
popular_songs = song_interactions.filter(col("interaction_count") >= min_song_interactions)

df = df.join(active_users, "user_id").join(popular_songs, "song_id")
df = df.select("user_id", "song_id", "interaction")

# Create indexes for ALS
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex")
song_indexer = StringIndexer(inputCol="song_id", outputCol="songIndex")

df = user_indexer.fit(df).transform(df)
df = song_indexer.fit(df).transform(df)

# Save to parquet
output_path = "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"
df.write.mode("overwrite").parquet(output_path)

# Read back and show sample
df_read = spark.read.parquet(output_path)
df_read.show(10, truncate=False)
