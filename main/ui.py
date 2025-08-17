# ui.py
import streamlit as st
import pandas as pd

from existing_user_recommendation import HybridRecommender
from new_user_recommendation import NewUserContentRecommender

# =========================
# Paths (edit if needed)
# =========================
COMMON_SONG_CSV = "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/data/common_song.csv"
USER_INTERACTIONS_PARQUET = "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/user_interactions.parquet"

ALS_MODEL_PATH = "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/model/als_model"
VECTORIZED_FEATURES_PATH = "C:/Users/dhruv/OneDrive/Desktop/Music_Recommendation_project/outputs/vectorized_features.parquet"

# =========================
# Load metadata (song names, artists)
# =========================
metadata_df = pd.read_csv(COMMON_SONG_CSV)

# Ensure required columns exist
for c in ["song_id", "artist_name", "track_name"]:
    if c not in metadata_df.columns:
        raise ValueError(f"Column '{c}' missing in {COMMON_SONG_CSV}")

metadata_df["display_name"] = metadata_df["artist_name"].fillna("") + " - " + metadata_df["track_name"].fillna("")
song_map = dict(zip(metadata_df["display_name"], metadata_df["song_id"]))
id_to_display = dict(zip(metadata_df["song_id"], metadata_df["display_name"]))

# =========================
# Build user_id dropdown list (from interactions)
# =========================
try:
    # requires pyarrow for pandas.read_parquet
    ui_users_df = pd.read_parquet(USER_INTERACTIONS_PARQUET)
    if "user_id" not in ui_users_df.columns:
        raise ValueError("`user_id` column not found in user_interactions.parquet")
    user_ids = sorted(ui_users_df["user_id"].dropna().astype(str).unique().tolist())
except Exception as e:
    # fallback: empty list; user will see a warning
    user_ids = []
    st.warning(f"Could not load user_id list from parquet. Reason: {e}")

# =========================
# Initialize recommenders
# (They should encapsulate Spark/NumPy details themselves)
# =========================
hybrid_recommender = HybridRecommender(
    als_model_path=ALS_MODEL_PATH,
    vectorized_features_path=VECTORIZED_FEATURES_PATH,
    interactions_parquet_path=USER_INTERACTIONS_PARQUET,
)

new_user_recommender = NewUserContentRecommender(
    vectorized_features_path=VECTORIZED_FEATURES_PATH
)

# =========================
# UI
# =========================
st.title("🎵 Music Recommendation System")

user_type = st.radio("Are you a new user or an existing user?", ("Existing User", "New User"))

# ---------- Existing User ----------
if user_type == "Existing User":
    if not user_ids:
        st.error("No user_ids available for dropdown. Please check your user_interactions.parquet path.")
    else:
        user_id = st.selectbox("Select your User ID", options=user_ids, index=0)

        num_recs = st.slider("How many recommendations?", min_value=5, max_value=50, value=10, step=5)

        if st.button("Get Recommendations"):
            try:
                recs_df = hybrid_recommender.recommend(user_id=user_id, num_recs=num_recs)
                # Expect columns include at least: song_id (and optionally rank/score)
                if recs_df is None or len(recs_df) == 0:
                    st.error("No recommendations found for this user.")
                else:
                    # If Spark DF sneaks through, convert safely
                    if not isinstance(recs_df, pd.DataFrame):
                        try:
                            recs_df = recs_df.toPandas()
                        except Exception:
                            recs_df = pd.DataFrame(recs_df)

                    # Map IDs to names
                    if "song_id" in recs_df.columns:
                        recs_df["song_name"] = recs_df["song_id"].map(id_to_display).fillna(recs_df["song_id"])
                    # Add a clean rank if missing
                    if "rank" not in recs_df.columns:
                        recs_df.insert(0, "rank", range(1, len(recs_df) + 1))

                    st.subheader("Recommended Songs for You:")
                    st.dataframe(recs_df[["rank", "song_name"]])

            except Exception as e:
                st.error(f"Error while generating recommendations: {e}")

# ---------- New User ----------
else:
    st.subheader("Pick your top 10 favorite songs")

    selected_songs = st.multiselect(
        "Search and select songs:",
        options=metadata_df["display_name"].tolist(),
        max_selections=10
    )

    num_new_user_recs = st.slider("How many recommendations for new user?", min_value=5, max_value=50, value=10, step=5)

    if st.button("Get Recommendations"):
        if not selected_songs:
            st.warning("Please select at least 1 song.")
        else:
            try:
                # Convert display names to song_ids
                song_ids = [song_map[s] for s in selected_songs if s in song_map]

                results_df = new_user_recommender.recommend(song_ids, k=num_new_user_recs)
                if results_df is None or len(results_df) == 0:
                    st.error("No recommendations could be generated.")
                else:
                    # If Spark DF sneaks through, convert safely
                    if not isinstance(results_df, pd.DataFrame):
                        try:
                            results_df = results_df.toPandas()
                        except Exception:
                            results_df = pd.DataFrame(results_df)

                    # Map IDs back to names
                    if "song_id" in results_df.columns:
                        results_df["song_name"] = results_df["song_id"].map(id_to_display).fillna(results_df["song_id"])

                    # Add a rank column if missing
                    if "rank" not in results_df.columns:
                        results_df.insert(0, "rank", range(1, len(results_df) + 1))

                    st.subheader("Recommended Songs for You:")
                    st.dataframe(results_df[["rank", "song_name"]])

            except Exception as e:
                st.error(f"Error while generating recommendations: {e}")
