import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts

mat_data = loadmat('dip_hw_3.mat')
img_d2a = mat_data['d2a']  # [M, N, 3]
img_d2b = mat_data['d2b']

images = {'Εικόνα d2a': img_d2a, 'Εικόνα d2b': img_d2b}
k_values = [2, 3, 4]

for title, image in images.items():
    M, N, _ = image.shape
    pixels = M * N
    affinity_mat = image_to_graph(image)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'{title} - Normalized Cuts Ομαδοποίηση', fontsize=14)

    for i, k in enumerate(k_values):
        cluster_idx_ncuts = n_cuts(affinity_mat, k)
        segmentation_ncuts = cluster_idx_ncuts.reshape(M, N)

        cluster_idx_spec = spectral_clustering(affinity_mat, k)
        segmentation_spec = cluster_idx_spec.reshape(M, N)

        axes[0, i].imshow(segmentation_ncuts, cmap='tab10')
        axes[0, i].set_title(f'Ncuts, k={k}')
        axes[0, i].axis('off')

        axes[1, i].imshow(segmentation_spec, cmap='tab10')
        axes[1, i].set_title(f'Spectral Clustering, k={k}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()