import pandas as pd

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

    # Compute explicit ratings (only rows with explicit ratings)
    explicit_ratings = merged[merged["rating"] > 0]  # Filter for explicit ratings
    explicit_aggregated = (
        explicit_ratings.groupby(["ISBN", "Similarity", "book_title", "book_author", "year_of_publication", "publisher", "language", "category"])
        .agg(
            avg_rating=("rating", "mean"),  # Average explicit rating
            explicit_rating_count=("rating", "count")  # Count of explicit ratings
        )
        .reset_index()
    )

    # Compute total interactions (all interactions, including implicit)
    total_interactions_aggregated = (
        merged.groupby(["ISBN", "Similarity", "book_title", "book_author", "year_of_publication", "publisher", "language", "category"])
        .agg(
            total_interactions=("isbn", "count")  # Count of all interactions
        )
        .reset_index()
    )

    # Merge explicit ratings with total interactions
    final_aggregated = pd.merge(explicit_aggregated, total_interactions_aggregated, on=["ISBN", "Similarity", "book_title", "book_author", "year_of_publication", "publisher", "language", "category"])

    # Sort by similarity and relevance
    final_aggregated = final_aggregated.sort_values(by=["Similarity", "explicit_rating_count"], ascending=[True, False])

    # Reorder and rename columns for better readability
    final_aggregated = final_aggregated[[
        "ISBN", "Similarity", "book_title", "book_author", "year_of_publication", 
        "publisher", "language", "category", "avg_rating", 
        "explicit_rating_count", "total_interactions"
    ]]
    final_aggregated = final_aggregated.rename(
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
    final_aggregated["Relevance"] = final_aggregated["Relevance"].round(3)
    final_aggregated["Average Rating"] = final_aggregated["Average Rating"].round(2)

    return final_aggregated

