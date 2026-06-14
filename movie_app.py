import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

API_KEY = "cf257c0ddce9d94765c85efbed72947e"


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
    
)


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #050816 0%,
        #0b1635 50%,
        #1a1240 100%
    );
    color: white;
}

/* Header glow */
.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
}

/* Movie cards */
[data-testid="stVerticalBlock"] > div {
    border-radius: 20px;
}

/* Glass effect containers */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 20px;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    background: linear-gradient(
        90deg,
        #6d28d9,
        #2563eb
    );
    color: white;
    border: none;
}

/* Search box */
.stTextInput input {
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.03);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

@st.cache_resource
def load_files():
    df = pd.read_pickle("df.pkl")
    
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    
    # Reset DataFrame index to ensure alignment
    df = df.reset_index(drop=True)
    
    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_files()
df["popularity"] = pd.to_numeric(
    df["popularity"],
    errors="coerce"
)

# Create a reverse mapping from title to index
title_to_idx = {title: idx for idx, title in enumerate(df['title'])}

# --------------------------------------------------
# POSTER
# --------------------------------------------------

@st.cache_data
def fetch_poster(movie_name):
    try:
        movie_name = quote(movie_name)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        data = requests.get(url, timeout=10).json()
        
        if len(data["results"]) == 0:
            return None
        
        poster_path = data["results"][0]["poster_path"]
        
        if poster_path is None:
            return None
        
        return "https://image.tmdb.org/t/p/w500" + poster_path
    
    except:
        return None
    
# --------------------------------------------------
# RECOMMEND
# --------------------------------------------------

def recommend(movie_title):
    try:
        # Find the index of the selected movie
        if movie_title in title_to_idx:
            idx = title_to_idx[movie_title]
        else:
            st.error(f"Movie '{movie_title}' not found in the dataset")
            return []
        
        # Check if idx is within bounds
        if idx >= tfidf_matrix.shape[0]:
            st.error(f"Index {idx} is out of bounds for matrix with shape {tfidf_matrix.shape}")
            return []
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(
            tfidf_matrix[idx:idx+1],  # Use slicing to keep 2D shape
            tfidf_matrix
        ).flatten()
        
        # Get top 6 similar movies (excluding the first one which is the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:11]
        
        # Filter valid indices
        valid_indices = [i for i in similar_indices if i < len(df)]
        
        if len(valid_indices) == 0:
            st.warning("No valid recommendations found")
            return []
        
        recommendations = []
        for i in valid_indices:
            try:
                movie_data = df.iloc[i]
                recommendations.append({
                    "title": movie_data["title"],
                    "overview": movie_data.get("overview", "No overview available"),
                    "rating": movie_data.get("vote_average", "N/A"),
                    "popularity": movie_data.get("popularity", "N/A")
                })
            except Exception as e:
                st.warning(f"Could not process recommendation at index {i}: {e}")
                continue
        
        return recommendations
    
    except Exception as e:
        st.error(f"Error in recommendation: {str(e)}")
        return []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎬 Menu")
st.sidebar.success(f"Movies in Dataset: {len(df):,}")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 class='main-title'>
🎬 Movie Recommender
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">

<h3 style="text-align:center;">
🍿 Discover Your Next Favorite Movie
</h3>

<p style="text-align:center;color:#d1d5db;">
Search thousands of movies and get intelligent recommendations powered by NLP and Machine Learning.
</p>

</div>
""", unsafe_allow_html=True)
# st.markdown(
#     "<p style='text-align:center;color:gray;'>Find your next favorite movie in seconds</p>",
#     unsafe_allow_html=True
# )
st.divider()


# --------------------------------------------------
# SEARCH BOX
# --------------------------------------------------

movie_list = sorted(df["title"].dropna().unique())

search_text = st.text_input(
    "Search Movie",
    placeholder="Type Avatar, Batman, Harry Potter..."
)

selected_movie = None

if search_text:
    suggestions = [
        movie for movie in movie_list 
        if search_text.lower() in movie.lower()
    ][:20]
    
    if suggestions:
        selected_movie = st.selectbox("Suggestions", suggestions)
    else:
        st.warning("No movies found matching your search")


# --------------------------------------------------
# SELECTED MOVIE DETAILS
# --------------------------------------------------

if selected_movie:
    try:
        movie_row = df[df["title"] == selected_movie].iloc[0]
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            poster = fetch_poster(selected_movie)
            if poster:
                st.image(poster, use_container_width=True)
        
        with col2:
            st.subheader(selected_movie)
            
            if "vote_average" in movie_row and pd.notna(movie_row["vote_average"]):
                st.write(f"⭐ Rating: {movie_row['vote_average']:.1f}/10")
            
            if "overview" in movie_row and pd.notna(movie_row["overview"]):
                st.write(movie_row["overview"])
            
            # Check if movie exists in title_to_idx
            if selected_movie in title_to_idx:
                st.success("✓ Available for recommendations")
            else:
                st.error("⚠️ Movie is not available for recommendations")
    
    except Exception as e:
        st.error(f"Error displaying movie: {e}")


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if selected_movie:
    if st.button("🎯 Recommend Similar Movies", type="primary"):
        with st.spinner("Finding similar movies..."):
            recommendations = recommend(selected_movie)
        
        if recommendations:
            st.divider()
            st.subheader("🎬 Recommended Movies")

        for row_start in range(0, len(recommendations), 5):
            cols = st.columns(5)
            for col, movie in zip(
                cols,
                recommendations[row_start:row_start + 5]
            ):
                with col:
                    poster = fetch_poster(movie["title"])
                    if poster:
                        st.image(
                            poster,
                            use_container_width=True
                        )
                    else:
                        st.image(
                            "https://placehold.co/300x450?text=No+Poster",
                            use_container_width=True
                        )
                    st.markdown(f"**{movie['title']}**")
                    st.write(f"⭐ {movie['rating']}")

                    with st.expander("📋 View Details"):
                        st.write(movie["overview"])
                        st.write(f"**Rating:** {movie['rating']}")
                        st.write(f"**Popularity:** {movie['popularity']}")
                        st.write("these are the top recommendations ")
        # else:
        #     st.warning("No recommendations found. Try a different movie.")


# -------------------------------------
# HOME PAGE MOVIES
# -------------------------------------
st.divider()
st.subheader("🔥 Popular Movies")


popular_movies = (
    df[df["title"].notna()]
    .sort_values(
        by="popularity",
        ascending=False
    )
    .head(5)
)

cols = st.columns(5)

for col, (_, movie) in zip(cols, popular_movies.iterrows()):
    with col:
        poster = fetch_poster(movie["title"])
        if poster:
            st.image(
                poster,
                use_container_width=True
            )
        if pd.notna(movie["title"]):
            st.caption(movie["title"])


st.write("")
st.subheader("⭐ Top Rated Movies")

top_movies = df.sort_values(
    by="vote_average",
    ascending=False
).head(5)

cols = st.columns(5)

for col, (_, movie) in zip(cols, top_movies.iterrows()):
    with col:
        poster = fetch_poster(movie["title"])
        if poster:
            st.image(
                poster,
                use_container_width=True
            )
        if pd.notna(movie["title"]):
            st.caption(movie["title"])

st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Made with ❤️ </p>",
    unsafe_allow_html=True
)