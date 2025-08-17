from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("SpotifyRecommendation").getOrCreate()
song_df = spark.read.csv("C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/data/common_song.csv", header=True, inferSchema=True)
song_df.show(5)


columns_to_cast = ['popularity', 'danceability', 'energy', 
                   'key', 'loudness', 'mode', 'speechiness', 
                   'acousticness', 'liveness', 'tempo', 'time_signature']
for col_name in columns_to_cast:
    song_df = song_df.withColumn(col_name, col(col_name).cast(DoubleType()))



indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
encoder = OneHotEncoder(inputCol="genreIndex", outputCol="genreVec")

indexed = indexer.fit(song_df).transform(song_df)
encoded = encoder.fit(indexed).transform(indexed)   


feature_columns = ['popularity', 'year', 'danceability', 'energy', 
                   'key', 'loudness', 'mode', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 
                   'valence', 'tempo', 'duration_ms', 
                   'time_signature', 'genreVec']

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
final_df = assembler.transform(encoded)

final_df.select("features").show(truncate=False)
final_df.write.mode("overwrite").parquet("C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/vectorized_features.parquet")
