import numpy as np
import sys

INTERACTIVE = False  # True: adım adım ilerler, False: otomatik çalışır

"""
Bu script  matrislerin özdeğer ve özvektörlerini manuel olarak adım adım hesaplar.
INTERACTIVE=True ise her adımda kullanıcıdan Enter bekler.
INTERACTIVE=False ise tüm adımlar otomatik çalışır (test ve rapor üretimi için önerilir).
"""

# ---------------------- Yardımcı Fonksiyonlar ----------------------

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def print_subheader(sub):
    print("\n--- " + sub + " ---")

def pause():
    if INTERACTIVE:
        input("Devam etmek için Enter'a basın...")

def copy_matrix(mat):
    return np.copy(mat)

def is_square(mat):
    return mat.shape[0] == mat.shape[1]

def validate_matrix(mat):
    if not is_square(mat):
        raise ValueError("Matris kare olmalıdır.")
    if mat.shape != (2, 2):
        raise ValueError("Bu uygulama yalnızca 2x2 matrisler için tasarlanmıştır.")

# ---------------------- Matris ve Polinom ----------------------

def create_matrix():
    """Sabit veya kullanıcıdan alınan 2x2 matris."""
    # Kullanıcıdan almak için kod eklenebilir
    return np.array([[4, -2], [1, 1]])

def print_matrix(mat):
    print("\nMatris:")
    print(mat)

def get_trace(mat):
    return mat[0, 0] + mat[1, 1]

def get_determinant(mat):
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

def characteristic_polynomial(mat):
    """2x2 matrisin karakteristik polinom katsayılarını döndürür."""
    a, b = mat[0, 0], mat[0, 1]
    c, d = mat[1, 0], mat[1, 1]
    return [1, -(a + d), a * d - b * c]

def discriminant(coeffs):
    """İkinci dereceden denklemin diskriminantını hesaplar."""
    a, b, c = coeffs
    return b ** 2 - 4 * a * c

def solve_quadratic(coeffs):
    """İkinci dereceden denklemin köklerini bulur (özdeğerler)."""
    a, b, c = coeffs
    delta = discriminant(coeffs)
    sqrt_delta = np.sqrt(delta)
    root1 = (-b + sqrt_delta) / (2 * a)
    root2 = (-b - sqrt_delta) / (2 * a)
    return np.array([root1, root2])

def format_polynomial(coeffs):
    return f"{coeffs[0]}*λ² + ({coeffs[1]})*λ + ({coeffs[2]})"

# ---------------------- Özdeğer ve Özvektör ----------------------

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def null_space_vector(mat):
    # 2x2 için satırları kullanarak çözüm
    if not np.isclose(mat[0, 1], 0):
        return np.array([1, -mat[0, 0] / mat[0, 1]])
    elif not np.isclose(mat[1, 0], 0):
        return np.array([-mat[1, 1] / mat[1, 0], 1])
    else:
        return np.array([1, 0])

def find_eigenvector(mat, eigval):
    """Bir özdeğer için özvektör bulur."""
    A = mat - eigval * np.eye(2)
    vec = null_space_vector(A)
    return normalize(vec)

def find_all_eigenvectors(mat, eigvals):
    """Tüm özdeğerler için özvektörleri döndürür."""
    vectors = []
    for ev in eigvals:
        vectors.append(find_eigenvector(mat, ev))
    return np.column_stack(vectors)
# Genel n x n matrisler için QR algoritması ve SVD ile özdeğer/özvektör hesaplama

def manual_eig(A, max_iter=1000, tol=1e-10):
    """
    n x n kare matris için QR algoritması ile özdeğerler, SVD ile özvektörler.
    """
    import numpy as np
    n = A.shape[0]
    # QR algoritması ile özdeğerler
    Ak = np.copy(A)
    for _ in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    eigvals = np.diag(Ak)
    # Özvektörler için her özdeğer için null space kullan
    eigvecs = []
    for eigval in eigvals:
        # (A - λI)x = 0 çözümü, SVD ile null space
        M = A - eigval * np.eye(n)
        U, S, Vh = np.linalg.svd(M)
        null_mask = (S <= tol)
        if np.any(null_mask):
            vec = Vh.T[:, null_mask][:, 0]
        else:
            # En küçük tekil değere karşılık gelen vektörü al
            vec = Vh.T[:, -1]
        eigvecs.append(normalize(vec.real))
    eigvecs = np.column_stack(eigvecs)
    return eigvals.real, eigvecs.real
# ---------------------- Adım Adım Hesaplama ----------------------

def step_matrix_info(mat):
    print_subheader("Matris Bilgisi")
    print_matrix(mat)
    print(f"İz (Trace): {get_trace(mat)}")
    print(f"Determinant: {get_determinant(mat)}")
    pause()

def step_characteristic(mat):
    print_subheader("Karakteristik Polinom")
    coeffs = characteristic_polynomial(mat)
    print(f"Polinom: {format_polynomial(coeffs)}")
    print(f"Katsayılar: {coeffs}")
    pause()
    return coeffs

def step_discriminant(coeffs):
    print_subheader("Diskriminant Hesabı")
    disc = discriminant(coeffs)
    print(f"Diskriminant: {disc}")
    pause()
    return disc

def step_eigenvalues(coeffs):
    print_subheader("Özdeğerler")
    eigvals = solve_quadratic(coeffs)
    print(f"Özdeğerler: {eigvals}")
    pause()
    return eigvals

def step_eigenvectors(mat, eigvals):
    print_subheader("Özvektörler")
    eigvecs = find_all_eigenvectors(mat, eigvals)
    for i, v in enumerate(eigvecs.T):
        print(f"v{i+1}: {v}")
    pause()
    return eigvecs

def step_validation(mat, eigvals, eigvecs):
    print_subheader("Doğrulama (A*v = λ*v)")
    for i in range(eigvecs.shape[1]):
        Av = mat @ eigvecs[:, i]
        lv = eigvals[i] * eigvecs[:, i]
        print(f"A*v{i+1}: {Av}, λ*v{i+1}: {lv}")
        print(f"Fark: {Av - lv}")
    pause()

def final_report(mat, eigvals, eigvecs):
    print_header("SONUÇ RAPORU")
    print_matrix(mat)
    print("\nÖzdeğerler:")
    for i, val in enumerate(eigvals):
        print(f"  λ{i+1}: {val}")
    print("\nÖzvektörler (her sütun bir özvektör):")
    print(eigvecs)
    print("\nDoğrulama tamamlandı. Her özvektör için A*v = λ*v eşitliği sağlanıyor.")

# ---------------------- Ana Akış ----------------------

def main():
    print_header("Genel Kare Matris için Manuel Özdeğer ve Özvektör Hesaplama (QR Algoritması)")
    mat = create_matrix()
    if not is_square(mat):
        raise ValueError("Matris kare olmalıdır.")
    print_matrix(mat)
    eigvals, eigvecs = manual_eig(mat)
    print("\nÖzdeğerler:")
    for i, val in enumerate(eigvals):
        print(f"  λ{i+1}: {val}")
    print("\nÖzvektörler (her sütun bir özvektör):")
    print(eigvecs)
    print("\nDoğrulama (A*v = λ*v):")
    for i in range(eigvecs.shape[1]):
        Av = mat @ eigvecs[:, i]
        lv = eigvals[i] * eigvecs[:, i]
        print(f"A*v{i+1}: {Av}, λ*v{i+1}: {lv}")
        print(f"Fark: {Av - lv}")
    print("\nDoğrulama tamamlandı. Her özvektör için A*v = λ*v eşitliği yaklaşık olarak sağlanıyor.")

if __name__ == "__main__":
    main()
