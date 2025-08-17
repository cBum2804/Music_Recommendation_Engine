from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder.appName("SpotifyRecommendation").getOrCreate()

# Load CSV properly
df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    r"C:\Users\dhruv\OneDrive\Desktop\Music Recommendation project\data\Spotifydataset.csv"
)

