import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def image_to_graph(img_array: np.ndarray) -> np.ndarray:

    M, N, C = img_array.shape
    num_pixels = M * N

    # Αναδιάταξη εικόνας: κάθε pixel γίνεται 1 γραμμή διαστάσεων C
    pixels = img_array.reshape(num_pixels, C)  # [M*N, C]

    # Δημιουργία affinity πίνακα
    affinity_mat = np.zeros((num_pixels, num_pixels), dtype=float)

    # Υπολογισμός ευκλείδειας απόστασης και αντιστροφής εκθετικής
    for i in range(num_pixels):
        for j in range(i, num_pixels):
            diff = pixels[i] - pixels[j]
            dist = np.sqrt(np.sum(diff ** 2))
            affinity = 1.0 / np.exp(dist)

            affinity_mat[i, j] = affinity
            affinity_mat[j, i] = affinity 

    return affinity_mat