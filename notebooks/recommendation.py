import pandas as pd

def recommend(book_name, knn_model, csr_data, book_titles, n_neighbors=10):
    """
    Recommends books similar to the given book title.

    Parameters:
    - book_name (str): The title of the book to base recommendations on.
    - knn_model: Trained KNN model.
    - csr_data: Preprocessed sparse matrix.
    - book_titles (list): List mapping row indices to book titles.
    - n_neighbors (int): Number of neighbors to consider.

    Returns:
    - List of recommended book titles and similarity scores.
    """
    # Find the index of the given book
    try:
        book_idx = book_titles.index(book_name)
    except ValueError:
        return f"No matching book found for '{book_name}'"

    # Find similar books using KNN
    distances, indices = knn_model.kneighbors(csr_data[book_idx], n_neighbors=n_neighbors)

    # Map indices to book titles
    recommendations = [(book_titles[i], round(dist, 3)) for i, dist in zip(indices[0], distances[0])]
    return recommendations