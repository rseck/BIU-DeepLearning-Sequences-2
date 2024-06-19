# Roee esquira, ID 309840791
# Yedidia Kfir, ID 209365188

from pathlib import Path
from typing import List, Tuple

import numpy as np


def most_similar(word, k, embeddings) -> Tuple[List[int], List[float]]:
    similarity = np.dot(word, embeddings.T) / (
        np.linalg.norm(word) * np.linalg.norm(embeddings, axis=-1)
    )
    indicies = np.argsort(similarity)[-k:]
    distance = similarity[indicies]
    return indicies.tolist(), distance.tolist()


def main(vec_file_name: str, words_file_name: str, words: list, k: int):
    vocab = Path(words_file_name).read_text().split()
    vecs = np.loadtxt(vec_file_name)

    for word in words:
        if word not in vocab:
            print(f"{word} not in vocabulary")
            continue
        word_vec = vecs[vocab.index(word)]
        similar_idx, distances = most_similar(word_vec, k, vecs)
        print(f"Words most similar to {word}:")
        for i, idx in enumerate(similar_idx):
            print(f"{i + 1}. {vocab[idx]} - {distances[i]:.2f}")


if __name__ == "__main__":
    main(
        r"wordVectors.txt",
        r"vocab.txt",
        ["dog", "england", "john", "explode", "office"],
        5,
    )
