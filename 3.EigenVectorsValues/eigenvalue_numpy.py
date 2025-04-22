import numpy as np

def main():
    A = np.array([[4, -2], [1, 1]])
    print("Matrix A:\n", A)
    values, vectors = np.linalg.eig(A)
    print("NumPy eigenvalues:", values)
    print("NumPy eigenvectors (columns):\n", vectors)

if __name__ == "__main__":
    main()
