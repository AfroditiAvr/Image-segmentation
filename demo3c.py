import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
from n_cuts import n_cuts, calculate_n_cut_value, n_cuts_recursive

T1 = 5      # Ελάχιστο αποδεκτό μέγεθος cluster
T2 = 0.20   # Μέγιστη αποδεκτή τιμή ncut

data = loadmat("dip_hw_3.mat")
img1 = data["d2a"]
img2 = data["d2b"]

images = {"Image 1 (d2a)": img1, "Image 2 (d2b)": img2}

for title, img in images.items():
    print("\n---", title, "---")
    M, N, _ = img.shape
    affinity = image_to_graph(img)

    cluster_recursive = n_cuts_recursive(affinity, T1=T1, T2=T2)
    segmentation_recursive = cluster_recursive.reshape((M, N))
    print(f"[Recursive] Unique labels: {np.unique(cluster_recursive)}")

    cluster_ncuts_k2 = n_cuts(affinity, k=2)
    segmentation_k2 = cluster_ncuts_k2.reshape((M, N))
    cluster_ncuts_k3 = n_cuts(affinity, k=3)
    segmentation_k3 = cluster_ncuts_k3.reshape((M, N))

    cluster_spec_k2 = spectral_clustering(affinity, k=2)
    segmentation_spec_k2 = cluster_spec_k2.reshape((M, N))
    cluster_spec_k3 = spectral_clustering(affinity, k=3)
    segmentation_spec_k3 = cluster_spec_k3.reshape((M, N))

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{title} - Recursive & Flat Ncuts vs Spectral", fontsize=15)

    axs[0, 0].imshow(segmentation_recursive, cmap="tab10")
    axs[0, 0].set_title(f"Recursive Ncuts\nLabels: {len(np.unique(cluster_recursive))}")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(segmentation_k2, cmap="tab10")
    axs[0, 1].set_title("Ncuts k=2")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(segmentation_k3, cmap="tab10")
    axs[0, 2].set_title("Ncuts k=3")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(img)
    axs[1, 0].set_title("Original Image")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(segmentation_spec_k2, cmap="tab10")
    axs[1, 1].set_title("Spectral Clustering k=2")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(segmentation_spec_k3, cmap="tab10")
    axs[1, 2].set_title("Spectral Clustering k=3")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
