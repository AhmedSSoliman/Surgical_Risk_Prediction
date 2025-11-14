def remove_duplicates_df(df):
    """
    Remove duplicate columns and index labels from a DataFrame.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df.index.duplicated()]
    return df


"""
MUVERA-based Clinical Notes Preprocessing

This module provides an alternative to process_notes.py, using MUVERA to compress multi-vector note embeddings (e.g., ColBERT) into fixed-size vectors for efficient retrieval and downstream modeling.

ColBERT Integration Example:
---------------------------
To use ColBERT as the embedder, pass a function or ColBERT model instance that takes a note string and returns a multi-vector embedding (shape: [n_vectors, embed_dim]).

Example:
    from colbert.inference import ColBERT
    colbert = ColBERT.from_pretrained('colbert-ir/colbertv2.0')
    def colbert_embed(note):
        return colbert.encode(note)  # returns np.ndarray [n_vectors, embed_dim]
    result = process_notes_muvera(note_texts, colbert_embed)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection

class MUVERA:
    """
    MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings
    Compresses multi-vector embeddings into single fixed-size vectors.
    """
    def __init__(self, n_partitions=8, reduced_dim=128, n_repeats=3, final_dim=128, random_state=42):
        self.n_partitions = n_partitions
        self.reduced_dim = reduced_dim
        self.n_repeats = n_repeats
        self.final_dim = final_dim
        self.random_state = random_state
        self.partitioners = []
        self.projectors = []
        self.final_projector = SparseRandomProjection(n_components=final_dim, random_state=random_state)

    def fit(self, multi_vectors):
        """
        Fit partitioners and projectors on multi-vector embeddings.
        Args:
            multi_vectors: np.ndarray of shape (n_vectors, embed_dim)
        """
        kmeans = KMeans(n_clusters=self.n_partitions, random_state=self.random_state)
        assignments = kmeans.fit_predict(multi_vectors)
        self.partitioners.append(kmeans)
        for i in range(self.n_partitions):
            partition = multi_vectors[assignments == i]
            projector = SparseRandomProjection(n_components=self.reduced_dim, random_state=self.random_state)
            if partition.shape[0] > 0:
                projector.fit(partition)
            self.projectors.append(projector)

    def transform(self, multi_vectors):
        assignments = self.partitioners[0].predict(multi_vectors)
        reduced_vectors = []
        for i in range(self.n_partitions):
            partition = multi_vectors[assignments == i]
            if partition.shape[0] == 0:
                reduced = np.zeros(self.reduced_dim)
            else:
                reduced = self.projectors[i].transform(partition).mean(axis=0)
            reduced_vectors.append(reduced)
        stacked = np.hstack(reduced_vectors)
        for _ in range(self.n_repeats - 1):
            stacked = np.hstack([stacked, stacked])
        fixed_vector = self.final_projector.transform(stacked.reshape(1, -1))[0]
        return fixed_vector

def process_notes_muvera(note_texts, embedder, muvera_model=None):
    """
    Process clinical notes using multi-vector embeddings and MUVERA compression.
    Args:
        note_texts: List of raw note strings
        embedder: Callable that returns multi-vector embeddings for a note (e.g., ColBERT)
        muvera_model: MUVERA instance (fit on training set)
    Returns:
        Dict with compressed embedding and metadata
    """
    # Step 1: Generate multi-vector embeddings for each note
    multi_vectors = []
    for note in note_texts:
        mv = embedder(note)  # shape: (n_vectors, embed_dim)
        multi_vectors.append(mv)
    # Step 2: Concatenate all multi-vectors for patient
    all_vectors = np.vstack(multi_vectors)
    # Step 3: Compress with MUVERA
    if muvera_model is None:
        muvera_model = MUVERA()
        muvera_model.fit(all_vectors)
    compressed = muvera_model.transform(all_vectors)
    return {
        'compressed_embedding': compressed,
        'num_notes': len(note_texts),
        'total_vectors': all_vectors.shape[0],
        'embedding_dim': compressed.shape[0]
    }
