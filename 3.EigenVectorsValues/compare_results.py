import numpy as np
from eigenvalue_manual import create_matrix, manual_eig

def compare():
    # Use the same matrix as in manual script
    A = create_matrix()
    # Manual method
     man_vals, man_vecs = manual_eig(A)
    # NumPy method
    np_vals, np_vecs = np.linalg.eig(A)
    print("--- Manuel Hesaplama ---")
    print("Özdeğerler:", man_vals)
    print("Özvektörler (sütunlar):\n", man_vecs)
    print("\n--- NumPy Hesaplama ---")
    print("Özdeğerler:", np_vals)
    print("Özvektörler (sütunlar):\n", np_vecs)
    # Compare
    print("\n--- Karşılaştırma ---")
    print("Özdeğerler yakın mı?", np.allclose(np.sort(man_vals), np.sort(np_vals)))
   # Eigenvectors may differ in sign or scale, compare normalized for each column
    n = A.shape[0]
    for i in range(n):
        v1 = man_vecs[:, i] / np.linalg.norm(man_vecs[:, i])
        v2 = np_vecs[:, i] / np.linalg.norm(np_vecs[:, i])
        # Compare up to sign
        close = np.allclose(v1, v2) or np.allclose(v1, -v2)
        print(f"Özvektör {i+1} yakın mı (işaret hariç)?", close)

if __name__ == "__main__":
    compare()
