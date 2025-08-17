from pyspark.sql import SparkSession
#from pyspark.sql.functions import udf, col, row_number, broadcast
import numpy as np
import sys, os
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import numpy as np


# Note: The following lines of are of no use. This file is not recommended as the operations on the large
# dataset takes large amount of time(I dont know how much because my laptop just got crashed in 20 min)

# I used GPU and RAM and some   
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("ContentBasedRecommendation") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Read data
user_profiles = spark.read.parquet("outputs/user_profiles.parquet").repartition(16, "user_id")
song_features = spark.read.parquet("outputs/vectorized_features.parquet").repartition(16).cache()


# Step 1: Collect and broadcast song features
song_features_broadcast = spark.sparkContext.broadcast(
    song_features.select("song_id", "features").rdd
    .map(lambda row: (row.song_id, row.features.toArray()))
    .collect()
)

# Step 2: Define cosine similarity
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# Step 3: Recommend top-N songs for each user using RDD map
def recommend_songs_for_user(row, top_n=10):
    user_id = row.user_id
    user_vec = row.user_profile.toArray()

    scores = []
    for song_id, song_vec in song_features_broadcast.value:
        sim = cosine_similarity(user_vec, song_vec)
        scores.append((user_id, song_id, float(sim)))

    # Sort by similarity and take top N
    top_scores = sorted(scores, key=lambda x: -x[2])[:top_n]
    return top_scores

# Step 4: Parallelize and run the recommendation engine
top_n = 10  
recommendations_rdd = user_profiles.rdd.flatMap(lambda row: recommend_songs_for_user(row, top_n))

# Step 5: Convert RDD back to DataFrame
recommendations_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("song_id", StringType(), True),
    StructField("similarity", FloatType(), True),
])

recommendations_df = spark.createDataFrame(recommendations_rdd, schema=recommendations_schema)

# Step 6: Optionally, join with song metadata
final_recommendations = recommendations_df.join(song_features.select("song_id", "track_name", "artist_name"), on="song_id", how="left")

# Display
final_recommendations.show(truncate=False)

