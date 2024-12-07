### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import os
import pandas as pd
import streamlit as st
import joblib
import json
from recommendation import recommend
from recommendation import content_recommender
from scipy.sparse import load_npz
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file



#######################################################################################################################################
### Create a title
st.markdown("<h1 style='text-align: center;'>Book Recomendation System</h1>", unsafe_allow_html=True)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data_snippet(file, num_rows=100):
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, file)
    df = pd.read_csv(path, index_col=0, nrows= num_rows)
    return df

@st.cache_data
def load_data(file):
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, file)
    df = pd.read_csv(path, index_col=0)
    return df

### B. Load first 100 rows
df = load_data("Books_data.csv")

df_snippet = load_data_snippet("PostBooksEDA.csv", 10)

df_books_snippet = load_data_snippet("Books_data.csv", 10)
df_books_snippet = df_books_snippet.drop(['img_s','img_m','img_l'], axis=1)

df_content = pd.read_csv('../data/PostBooksEDA.csv', index_col=0)

### C. Display the dataframe in the app
st.markdown("<h2 style='text-align: center;'>Post Preproccesing - Original Dataframe</h2>", unsafe_allow_html=True)
st.dataframe(df_snippet)

st.divider()
#######################################################################################################################################
## Display 
st.markdown("<h2 style='text-align: center;'>Book Specific Data for Recommendations</h2>", unsafe_allow_html=True)
st.dataframe(df_books_snippet)


#######################################################################################################################################
### Models

# First Model Content Based
# st.markdown("<h2 style='text-align: center;'>Models</h2>", unsafe_allow_html=True)
# st.markdown("<h3>Content Based - Title model </h3>", unsafe_allow_html=True)

# Group data to calculate review count and average score
#@st.cache_data
#def prepare_data(df):
#    return df.groupby('book_title').agg(
#        review_count=('rating', 'count'),
#        avg_review_score=('rating', 'mean')
#    ).reset_index()

# Vectorize book titles
#@st.cache_data
#def compute_similarity(titles):
#    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
#   TF_IDF_matrix = vectorizer.fit_transform(titles)
#   return cosine_similarity(TF_IDF_matrix, dense_output=False)

# Function to find similar books
#def content_recommender(selected_title, unique_titles, similarities, vote_threshold=10):
#    title_index = unique_titles[unique_titles['book_title'] == selected_title].index[0]
#    similar_indices = similarities[title_index].toarray().argsort()[0][::-1]
#    similar_books = unique_titles.iloc[similar_indices].reset_index(drop=True)
#    similar_books = similar_books[similar_books['review_count'] >= vote_threshold]
#    return similar_books

#st.markdown("<h2 style='text-align: center;'>Models</h2>", unsafe_allow_html=True)
#st.markdown("<h3>Content Based - Title model </h3>", unsafe_allow_html=True)


#unique_titles = prepare_data(df_content)
#similarities = compute_similarity(unique_titles['book_title'])

#st.markdown("<h4>Harry Potter and the Chamber of Secrets (Book 2) - Recommendations<h4>", unsafe_allow_html=True)
#similar_books = content_recommender("Harry Potter and the Chamber of Secrets (Book 2)", unique_titles, similarities, vote_threshold=10)
#st.dataframe(similar_books.head(15))

#st.markdown("<h4>The Eyes of the Dragon - Recommendations<h4>", unsafe_allow_html=True)
#similar_books = content_recommender("The Eyes of the Dragon", unique_titles, similarities, vote_threshold=10)
#st.dataframe(similar_books.head(15))
st.divider()
# Second Model
st.markdown("<h3>Item Based Collaborative Filtering</h3>", unsafe_allow_html=True)
# A. Load the model using joblib
# Load the model
model_path = os.path.join(script_dir, 'model.joblib')
model = load(model_path)

# Load the csr_matrix from the file
csr_path = os.path.join(script_dir, 'csr_data.npz')
csr_data = load_npz(csr_path)

# Load book isbn IDs
book_path = os.path.join(script_dir, 'book_isbn.json')
with open(book_path, 'r') as f:
    book_isbns = json.load(f)
    
# B. Set up input field

# Get all unique book titles from the dataframe
book_titles = df['book_title'].unique().tolist()

# Use a selectbox with search capabilities
selected_title = st.selectbox(
    'Start typing a book title:', 
    options=book_titles, 
)


# C. Use the model to give recommendations & write result
if selected_title:
    # Find the corresponding ISBN for the selected title
    selected_isbn = df.loc[df['book_title'] == selected_title, 'isbn'].values

    if len(selected_isbn) > 0:  # Ensure we have a valid ISBN
        selected_isbn = selected_isbn[0]  # Take the first match if duplicates exist

        # Use the recommender function
        recommendations = recommend(
            book_isbn=selected_isbn,  
            knn_model=model,
            csr_data=csr_data,
            book_isbns=book_isbns,  # List of ISBNs corresponding to the rows in csr_data
            dataframe=df,  # Original DataFrame with book metadata
            n_neighbors=11
        )

        # Select the searched/selected book (first recommendation)
        selected_book_row = df[df['isbn'] == selected_isbn].iloc[0]

        # Center the searched book in the middle of the page 
        col1, col2, col3 = st.columns([5, 3, 4])  # Create three column spaces
        with col2:  # Middle space for the searched book
            st.write(f"**{selected_book_row['book_title']}**")
            st.write(f"*{selected_book_row['book_author']}*")
            if 'img_m' in selected_book_row and isinstance(selected_book_row['img_l'], str):
                st.image(selected_book_row['img_m'], width=200)

        # Show the recommendations section below the central searched book area
        st.markdown("<h4 style='text-align: center;'>Recommended Books</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([4, 3, 3])
        
        # Create columns for recommendations (not touching the center display)
        cols = st.columns(len(recommendations) - 1)  # Subtract 1 because the first book is already shown above
        for i, (_, row) in enumerate(recommendations.iloc[1:].iterrows()):  # Exclude the selected book itself
            with cols[i]:  # Display each recommendation
                st.write(f"**{row['Title']}**")
                st.write(f"*{row['Author']}*")
                if 'img_l' in row and isinstance(row['img_l'], str):
                    st.image(row['img_l'], width=150)

        st.divider()
        st.markdown("<h4 style='text-align: center;'>Full Recommendations Relevance DataFrame</h4>", unsafe_allow_html=True)
        subset_df = recommendations.iloc[1:11].drop(['ISBN','img_s','img_m','img_l'], axis=1)  # Select rows from index 1 through 10 (inclusive)
        st.dataframe(subset_df)

    
    else:
        st.error("No ISBN found for the selected book title.")




#######################################################################################################################################
