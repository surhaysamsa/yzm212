import numpy as np
from eigenvalue_manual import create_matrix, characteristic_polynomial, solve_quadratic, find_all_eigenvectors

def compare():
    # Use the same matrix as in manual script
    A = create_matrix()
    # Manual method
    coeffs = characteristic_polynomial(A)
    man_vals = solve_quadratic(coeffs)
    man_vecs = find_all_eigenvectors(A, man_vals)
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
    # Eigenvectors may differ in sign or scale, compare normalized
    for i in range(2):
        v1 = man_vecs[:, i] / np.linalg.norm(man_vecs[:, i])
        v2 = np_vecs[:, i] / np.linalg.norm(np_vecs[:, i])
        print(f"Özvektör {i+1} yakın mı (işaret hariç)?", np.allclose(np.abs(v1), np.abs(v2)))

if __name__ == "__main__":
    compare()
