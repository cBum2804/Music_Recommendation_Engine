from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("try").getOrCreate()

# Load the prepared parquet with indexes already present
user_interactions = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"
)
als_reco = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/als_recommendations.parquet"
)
hybrid = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/hybrid_recommendations.parquet"
)
user_profiles = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_profiles.parquet"
)
vectorized_features = spark.read.parquet(
    r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/vectorized_features.parquet"
)

print(" ")
user_interactions.show(10, truncate = False)

print(" ")
als_reco.show(10, truncate = False)

print(" ")
hybrid.show(10, truncate = False)

print(" ")
user_profiles.show(10, truncate = False)

print(" ")
vectorized_features.show(10, truncate = False)
