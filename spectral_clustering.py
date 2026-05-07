import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:

    N = affinity_mat.shape[0]

    # Βήμα 1: Υπολογισμός διαγώνιου πίνακα D
    D = np.diag(np.sum(affinity_mat, axis=1))  # [N, N]

    # Βήμα 2: Υπολογισμός μη κανονικοποιημένου Λαπλασιανού πίνακα
    L = D - affinity_mat  # [N, N]

    # Βήμα 3: Υπολογισμός των k μικρότερων ιδιοτιμών/ιδιοδιανυσμάτων του L)
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')  # Smallest Magnitude

    # Βήμα 4: Δημιουργία πίνακα U με τα ιδιοδιανύσματα ως στήλες
    U = eigenvectors  # [N, k]

    # Βήμα 5: Κανονικοποίηση γραμμών
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    # Βήμα 6: Εφαρμογή k-means clustering στις γραμμές του U
    kmeans = KMeans(n_clusters=k, random_state=1,  n_init='auto')
    kmeans.fit(U_norm)
    labels = kmeans.labels_

    return labels.astype(float)


