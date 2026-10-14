# Project 4: Movie recommender

Movie recommender: Nicolas's fourth AI project recommends movies using collaborative filtering with no neural networks at all: only classical machine learning. Pick a movie and it returns the 10 most similar ones. Code: https://github.com/NicoDevGod/movie-recommender. Live demo: https://movie-recommender-ej1u.onrender.com.

Movie recommender, idea: Two movies are similar if the same users rated them similarly. Each movie is a vector of how every user rated it, and similarity is the cosine similarity between those vectors. It does not use genres or actors; the recommendation comes purely from the collective rating pattern (item-based collaborative filtering). The point of the project is that not every AI problem needs deep learning.

Movie recommender, data: MovieLens ml-latest-small, with about 100,000 ratings, about 9,700 movies and 610 users. Movies with fewer than 10 ratings are dropped (MIN_RATINGS=10) because a high similarity on so little data is coincidence; about 2,270 movies remain. Libraries: pandas to load and filter, SciPy for the sparse matrix, scikit-learn for cosine_similarity, Gradio for a searchable dropdown.

Movie recommender, sparse matrix: The ratings become a sparse movies-by-users matrix (SciPy csr_matrix). Only about 81,000 cells hold a real rating out of about 1.38 million possible cells, so storing only filled cells is the difference between fitting in memory and not. A query takes the chosen movie's row, runs cosine_similarity against all rows in one vectorized call, skips the movie itself and returns the top 10.

Movie recommender, design decision: He avoided precomputing the full movie-by-movie similarity matrix at startup (about 2,270 x 2,270). It grows with the square of the catalogue, the same kind of memory trap as the first two projects. Instead similarity is computed one row per query. It was the only project that deployed with no production bugs.

Movie recommender, example: Asking for Toy Story (1995) returns Toy Story 2, Jurassic Park and Star Wars among the top results, identical to the local result.
