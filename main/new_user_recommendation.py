from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
from typing import List, Tuple


class NewUserContentRecommender:
    def __init__(self,
                 spark: SparkSession = None,
                 vectorized_features_path: str = None,
                 metadata_csv: str | None = None):
        
        if spark is None:
            spark = SparkSession.builder.appName("NewUserRecommender").getOrCreate()
        self.spark = spark
        
        vectorized_features_path = vectorized_features_path or 'C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/vectorized_features.parquet'
        metadata_csv= metadata_csv or 'C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/data/common_song.csv'
        
        self.spark = spark
        self.features_df = (
            spark.read.parquet(vectorized_features_path)
                 .select("song_id", vector_to_array("features").alias("feat"))
                 .filter(col("song_id").isNotNull())
                 .filter(col("feat").isNotNull())
        )

        # Pull to pandas/numpy once for super fast inference in UI
        pdf = self.features_df.toPandas()
        self.song_ids: List[str] = pdf["song_id"].tolist()
        self.X: np.ndarray = np.stack(pdf["feat"].values).astype(float)  # (N, d)

        # Pre-normalize matrix for cosine = dot of normalized vectors
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.Xn = self.X / norms

        # Fast lookup: song_id -> row index
        self.id2row = {sid: i for i, sid in enumerate(self.song_ids)}

        # Optional metadata (artist/title)
        self.meta = None
        if metadata_csv:
            meta = pd.read_csv(metadata_csv)
            # Expect columns: song_id, artist_name, track_name
            use_cols = [c for c in ["song_id", "artist_name", "track_name"] if c in meta.columns]
            self.meta = meta[use_cols].drop_duplicates("song_id") if use_cols else None

    def recommend(self, selected_song_ids: List[str], k: int = 5) -> pd.DataFrame:
        """
        Build a user profile from selected songs and return top-k recommendations.
        Returns a pandas DataFrame with columns: song_id, score, [artist_name, track_name]
        """
        idxs = [self.id2row[sid] for sid in selected_song_ids if sid in self.id2row]
        if len(idxs) == 0:
            raise ValueError("None of the provided song_ids were found in features.")

        # User profile = mean of selected song feature vectors
        u = self.X[idxs].mean(axis=0)
        u_norm = np.linalg.norm(u)
        if u_norm == 0:
            # degenerate profile, return popular-ish by norm fallback
            scores = (self.Xn @ self.Xn.mean(axis=0))
        else:
            u = u / u_norm
            scores = self.Xn @ u  # cosine similarity to every song

        # Exclude the selected songs themselves
        scores[idxs] = -np.inf

        # Top-k
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            top_idx = part[np.argsort(-scores[part])]

        result = pd.DataFrame({
            "song_id": [self.song_ids[i] for i in top_idx],
            "score": scores[top_idx]
        })

        if self.meta is not None:
            result = result.merge(self.meta, on="song_id", how="left")

        return result
