import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering 
import time

mat_data = loadmat('dip_hw_3.mat')
img_d2a = mat_data['d2a']  # [M, N, 3]
img_d2b = mat_data['d2b']

images = {'Εικόνα d2a': img_d2a, 'Εικόνα d2b': img_d2b}

k_values = [2, 3, 4]

for img_name, img in images.items():
    M, N, C = img.shape
    print(f"\n=== {img_name} ===")

    print("Υπολογισμός του affinity matrix (graph)...")
    start_time = time.time()
    affinity_mat = image_to_graph(img)
    print(f"Ολοκληρώθηκε σε {time.time() - start_time:.2f} sec")

    fig, axs = plt.subplots(1, len(k_values) + 1, figsize=(5 * (len(k_values) + 1), 5))

    axs[0].imshow(img)
    axs[0].set_title(f"{img_name}\n(Original)")
    axs[0].axis('off')

    for idx, k in enumerate(k_values):
        print(f"\n--- Clustering {img_name} με k = {k} ---")
        labels = spectral_clustering(affinity_mat, k)
        print("Πρώτα 20 labels:", labels[:20])
        print("Μοναδικά labels:", np.unique(labels))

        labels_img = labels.reshape(M, N)
        axs[idx + 1].imshow(labels_img, cmap='tab10')
        axs[idx + 1].set_title(f"{img_name}\nk = {k}")
        axs[idx + 1].axis('off')

    plt.suptitle(f"Αποτελέσματα Spectral Clustering για {img_name}")
    plt.tight_layout()
    plt.show()

