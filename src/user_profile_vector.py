from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler

# Create Spark session
spark = SparkSession.builder \
    .appName("RobustUserProfileVectors") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# Load interactions & features
interactions = spark.read.parquet(
    "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"
)
features = spark.read.parquet(
    "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/vectorized_features.parquet"
)

# Join on song_id
user_song_features = interactions.join(features, on="song_id").filter(col("features").isNotNull())

# Convert vector to array
user_song_features = user_song_features.withColumn("features_array", vector_to_array("features"))

# Get vector size dynamically
vector_size = len(user_song_features.select("features_array").first()["features_array"])

# Split features into separate columns
for i in range(vector_size):
    user_song_features = user_song_features.withColumn(f"f_{i}", col("features_array")[i])

# Average features per user
avg_exprs = [avg(f"f_{i}").alias(f"f_{i}") for i in range(vector_size)]
user_profiles = user_song_features.groupBy("user_id").agg(*avg_exprs)

# Assemble back into vector
assembler = VectorAssembler(inputCols=[f"f_{i}" for i in range(vector_size)], outputCol="user_profile")
user_profiles = assembler.transform(user_profiles).select("user_id", "user_profile")

# Save user profiles
user_profiles.write.mode("overwrite").parquet(
    "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_profiles.parquet"
)


user_profiles.show(10, truncate= False)
spark.stop()