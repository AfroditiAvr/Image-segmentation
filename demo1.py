import numpy as np
from scipy.io import loadmat
from spectral_clustering import spectral_clustering 

def main():
    # Φόρτωση του affinity πίνακα από το .mat αρχείο
    data = loadmat("dip_hw_3.mat")
    d1a = data["d1a"]  # [N, N] affinity matrix

    print("Μέγεθος πίνακα d1a:", d1a.shape)

    #spectral clustering για k = 2, 3, 4
    for k in [2, 3, 4]:
        print(f"\n--- Clustering με k = {k} ---")
        labels = spectral_clustering(d1a, k=k)
        print("Ετικέτες clusters (πρώτα 20):", labels[:20]) 
        print("Σύνολο μοναδικών ετικετών:", np.unique(labels))

if __name__ == "__main__":
    main()
