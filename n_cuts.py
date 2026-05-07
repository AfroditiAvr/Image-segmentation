import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:

    D = np.diag(affinity_mat.sum(axis=1))
    L = D - affinity_mat
    vals, vecs = eigsh(L, k=k, M=D, which='SM')
    U = vecs

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    cluster_idx = kmeans.fit_predict(U)
    return cluster_idx.astype(float)


def calculate_n_cut_value(
    affinity_mat: np.ndarray,
    cluster_idx: np.ndarray
) -> float:

    if len(np.unique(cluster_idx)) != 2:
        raise ValueError("Η calculate_n_cut_value απαιτεί δύο clusters (αναδρομική περίπτωση).")

    A = np.where(cluster_idx == 0)[0]
    B = np.where(cluster_idx == 1)[0]

    W = affinity_mat

    assoc_AA = np.sum(W[np.ix_(A, A)])
    assoc_BB = np.sum(W[np.ix_(B, B)])
    assoc_AV = np.sum(W[A, :])
    assoc_BV = np.sum(W[B, :])

    nassoc = (assoc_AA / assoc_AV) + (assoc_BB / assoc_BV)
    ncut_value = 2.0 - nassoc

    return ncut_value


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    n = affinity_mat.shape[0]
    if n < 3 or np.allclose(affinity_mat, affinity_mat[0, 0]):
        return np.zeros(n, dtype=int)

    cluster_idx = n_cuts(affinity_mat, k=2)
    if len(np.unique(cluster_idx)) < 2:
        return np.zeros(n, dtype=int)

    ncut_value = calculate_n_cut_value(affinity_mat, cluster_idx)
    count_0, count_1 = np.sum(cluster_idx == 0), np.sum(cluster_idx == 1)
    print(f"n_cut_value: {ncut_value:.6f}, count_0: {count_0}, count_1: {count_1}")

    if ncut_value > T2 or count_0 < T1 or count_1 < T1:
        return cluster_idx

    labels = np.zeros_like(cluster_idx)
    next_label = 0

    for i in [0, 1]:
        idx = np.where(cluster_idx == i)[0]
        sub_affinity = affinity_mat[np.ix_(idx, idx)]
        sub_labels = n_cuts_recursive(sub_affinity, T1, T2)

        if np.all(sub_labels == 0):
            labels[idx] = next_label
            next_label += 1
        else:
            labels[idx] = sub_labels + next_label
            next_label = labels.max() + 1

    return labels