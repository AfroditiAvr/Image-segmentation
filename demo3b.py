import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts, calculate_n_cut_value

T1 = -1  # ελάχιστο μέγεθος cluster
T2 = 2   # μέγιστη αποδεκτή τιμή ncut

print(f"Parameters: T1 = {T1}, T2 = {T2}")

mat_data = loadmat('dip_hw_3.mat')
img_d2a = mat_data['d2a']  # [M, N, 3]
img_d2b = mat_data['d2b']  # [M, N, 3]

images = {'Image 1': img_d2a, 'Image 2': img_d2b}

for name, image in images.items():
    M, N, _ = image.shape
    pixels = M * N
    print(f"\n--- {name} ---")
    print(f"Image shape: {image.shape}")

    affinity_mat = image_to_graph(image)

    print("\n[INFO] Executing n_cuts for k=2...")
    cluster_idx_ncuts = n_cuts(affinity_mat, k=2)
    unique_labels_ncuts = np.unique(cluster_idx_ncuts)
    segmentation_ncuts = cluster_idx_ncuts.reshape(M, N)

    if len(unique_labels_ncuts) == 2:
        count_0 = np.sum(cluster_idx_ncuts == 0)
        count_1 = np.sum(cluster_idx_ncuts == 1)
        ncut_val = calculate_n_cut_value(affinity_mat, cluster_idx_ncuts)
        print(f"[nCuts] Unique labels: {unique_labels_ncuts}")
        print(f"[nCuts] count_0: {count_0}, count_1: {count_1}")
        print(f"[nCuts] Ncut value: {ncut_val:.4f}")
        ncut_title = f'nCuts (k=2)\nNcut={ncut_val:.4f}'
    else:
        print(f"[nCuts] Unique labels: {unique_labels_ncuts} (ncut=N/A - no partition)")
        ncut_val = None
        ncut_title = f'nCuts (k=2)\nNcut=N/A'

    print("\n[INFO] Executing spectral_clustering for k=2...")
    cluster_idx_spec = spectral_clustering(affinity_mat, k=2)
    unique_labels_spec = np.unique(cluster_idx_spec)
    segmentation_spec = cluster_idx_spec.reshape(M, N)
    print(f"[Spectral] Unique labels: {unique_labels_spec}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'{name} - NCuts vs Spectral Clustering (k=2)', fontsize=14)

    axes[0].imshow(segmentation_ncuts, cmap='tab10')
    axes[0].set_title(ncut_title)
    axes[0].axis('off')

    axes[1].imshow(segmentation_spec, cmap='tab10')
    axes[1].set_title('Spectral Clustering')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
