from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, row_number
from pyspark.sql import Window
import pandas as pd

class HybridRecommender:
    def __init__(self,
                 als_model_path: str,
                 vectorized_features_path: str | None = None,
                 user_profiles_path: str | None = None,
                 interactions_parquet_path: str = r"C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"):
        # Spark
        self.spark = SparkSession.builder.appName("HybridRecommender").getOrCreate()

        # ALS model (trained with userIndex/songIndex)
        self.als_model = ALSModel.load(als_model_path)

        # Load the interactions parquet that contains user_id/song_id + userIndex/songIndex
        # (schema should include: user_id, song_id, interaction, userIndex, songIndex)
        self.interactions = self.spark.read.parquet(interactions_parquet_path).cache()

        # Build small mapping DFs
        self.user_map = self.interactions.select("user_id", "userIndex").dropDuplicates().cache()
        self.item_map = self.interactions.select("song_id", "songIndex").dropDuplicates().cache()

        # Optional extras (not used in this minimal ALS path)
        self.vectorized_features = None
        self.user_profiles = None
        if vectorized_features_path:
            self.vectorized_features = self.spark.read.parquet(vectorized_features_path)
        if user_profiles_path:
            self.user_profiles = self.spark.read.parquet(user_profiles_path)

    def _user_id_to_index(self, user_id: str):
        row = (self.user_map
               .filter(col("user_id") == user_id)
               .select("userIndex")
               .limit(1)
               .collect())
        if not row:
            return None
        return row[0]["userIndex"]

    def recommend_existing_user(self, user_id: str, num_recs: int = 10) -> pd.DataFrame:
        # 1) Map user_id -> userIndex
        uidx = self._user_id_to_index(user_id)
        if uidx is None:
            # No mapping found for this user
            return pd.DataFrame(columns=["rank", "user_id", "song_id", "rating"])

        # 2) Call ALS with a DataFrame that has column name exactly "userIndex"
        subset = self.spark.createDataFrame([(uidx,)], ["userIndex"])
        als_recs = (self.als_model
                    .recommendForUserSubset(subset, num_recs)
                    .select("userIndex", explode(col("recommendations")).alias("rec"))
                    .select(
                        col("userIndex"),
                        col("rec.songIndex").alias("songIndex"),
                        col("rec.rating").alias("rating"))
                    )

        # 3) Map songIndex -> song_id (join with item_map)
        recs_with_ids = (als_recs
                         .join(self.item_map, on="songIndex", how="left")
                         .join(self.user_map.select("user_id", "userIndex"),
                               on="userIndex", how="left")
                         .select("user_id", "song_id", "rating"))

        # 4) Add rank
        w = Window.orderBy(col("rating").desc())
        ranked = recs_with_ids.withColumn("rank", row_number().over(w)).select("rank", "user_id", "song_id", "rating")

        # 5) Return as Pandas for Streamlit
        return ranked.toPandas()

    # Keep the external API unchanged
    def recommend(self, user_id: str, num_recs: int = 10) -> pd.DataFrame:
        return self.recommend_existing_user(user_id, num_recs)
