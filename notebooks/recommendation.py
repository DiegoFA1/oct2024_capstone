import pandas as pd
import numpy as np

# Title, Dataframe with books, Similarities matrix
def content_recommender(title, books, similarities, vote_threshold=10) :

    # Get the movie by the title
    book_index = books[books['book_title'] == title].index

    # Create a dataframe with the movie titles
    sim_df = pd.DataFrame(
        {'book': books['book_title'],
         'similarity': np.array(similarities[book_index, :].todense()).squeeze(),
         'Number of reviews': books['review_count'],
         'Avg Rating': books['avg_review_score']
        })

    # Get the top 10 movies with > 10 votes
    top_books = sim_df[sim_df['Number of reviews'] > vote_threshold].sort_values(by='similarity', ascending=False).head(10)

    return top_books


def recommend(book_isbn, knn_model, csr_data, book_isbns, dataframe, n_neighbors=10):
    """
    Recommends books similar to the given book's ISBN and fetches additional metadata,
    including explicit ratings and total interactions.

    Parameters:
    - book_isbn (str): The ISBN of the book to base recommendations on.
    - knn_model: Trained KNN model.
    - csr_data: Preprocessed sparse matrix.
    - book_isbns (list): List mapping row indices to book ISBNs.
    - dataframe (pd.DataFrame): Original DataFrame containing metadata for books.
    - n_neighbors (int): Number of neighbors to consider.

    Returns:
    - pd.DataFrame: DataFrame containing recommended book metadata, similarity scores,
                    average explicit ratings, and total interactions.
    """
    # Find the index of the given book
    try:
        book_idx = book_isbns.index(book_isbn)
    except ValueError:
        return pd.DataFrame(columns=["ISBN", "Message"], data=[[book_isbn, "No matching book found"]])

    # Find similar books using KNN
    distances, indices = knn_model.kneighbors(csr_data[book_idx], n_neighbors=n_neighbors)

    # Map indices to book ISBNs and similarity scores
    recommendations = [(book_isbns[i], round(dist, 3)) for i, dist in zip(indices[0], distances[0])]

    # Create a DataFrame with ISBN and similarity
    rec_df = pd.DataFrame(recommendations, columns=["ISBN", "Similarity"])

    # Merge with the original DataFrame to fetch additional book metadata
    merged = pd.merge(rec_df, dataframe, left_on="ISBN", right_on="isbn", how="left")

    
    # Sort by similarity and relevance
    merged = merged.sort_values(by=["Similarity", "explicit_rating_count"], ascending=[True, False])

    # Reorder and rename columns for better readability
    merged = merged[[
        "ISBN", "Similarity", "book_title", "book_author", "year_of_publication", 
        "publisher", "language", "category", "avg_rating", 
        "explicit_rating_count", "total_interactions",'img_s','img_m','img_l'
    ]]
    merged = merged.rename(
        columns={
            "book_title": "Title",
            "book_author": "Author",
            "Similarity": "Relevance",
            "avg_rating": "Average Rating",
            "explicit_rating_count": "Number of Reviews",
            "total_interactions": "Total Interactions"
        }
    )

    # Format numerical columns for cleaner display
    merged["Relevance"] = merged["Relevance"].round(3)
    merged["Average Rating"] = merged["Average Rating"].round(2)

    return merged

