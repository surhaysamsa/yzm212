# Makine Öğrenmesinde Matris Manipülasyonu, Özdeğerler ve Özvektörler Projesi

Bu proje, makine öğrenmesinde matris manipülasyonu, özdeğer ve özvektörlerin teorik temellerini ve uygulamalarını; ayrıca NumPy ve manuel yöntemlerle özdeğer/özvektör hesaplamalarını kapsamaktadır.

---

## 1. Teorik Arka Plan

### Matris Manipülasyonu
Makine öğrenmesinde veriler, satırları örnekleri ve sütunları özellikleri temsil eden matrisler olarak organize edilir. Matris manipülasyonu; veri düzenleme, dönüştürme, normalizasyon ve ölçeklendirme işlemlerini kapsar. Bu işlemler, algoritmaların verimli ve kararlı çalışmasını sağlar.

### Özdeğerler ve Özvektörler
Bir kare matrisin özdeğerleri ve özvektörleri, matrisin temel yapısal özelliklerini ortaya çıkarır. Özellikle boyut indirgeme (PCA), spektral kümeleme, SVD ve bazı regresyon tekniklerinde kritik rol oynar.

### NumPy `linalg.eig` Fonksiyonu
NumPy’nın `linalg.eig` fonksiyonu, kare bir matrisin özdeğerlerini ve sağ özvektörlerini yüksek performanslı LAPACK rutinleri ile hesaplar. Fonksiyonun teknik detayları ve kaynak kodları için [NumPy dokümantasyonu](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html) ve [GitHub](https://github.com/numpy/numpy/tree/main/numpy/linalg) adreslerine bakınız.

---

## 2. Dosya Açıklamaları

- **eigenvalue_manual.py**: NumPy kullanmadan (analitik yöntemle) 2x2 matris için özdeğer ve özvektör hesaplaması. Bu script, adım adım ve açıklamalı (eğitsel) bir şekilde çalışır. `INTERACTIVE` değişkeni ile kullanıcıdan her adımda onay alınarak veya otomatik olarak çalıştırılabilir.
- **eigenvalue_numpy.py**: NumPy `linalg.eig` ile özdeğer ve özvektör hesaplaması.
- **compare_results.py**: Her iki yöntemin sonuçlarının karşılaştırılması ve doğrulanması.
- **eigenvalue.py**: (Var olan dosya, isteğe bağlı kullanılabilir.)

---

## 3. Kullanım Talimatları

1. Gerekli kütüphaneyi yükleyin:
   ```bash
   pip install numpy
   ```
2. Aşağıdaki dosyaları sırasıyla çalıştırarak hesaplamaları ve karşılaştırmaları gözlemleyin:
   ```bash
   python eigenvalue_manual.py
   python eigenvalue_numpy.py
   python compare_results.py
   ```

Her bir dosya çalıştırıldığında, ekrana ilgili matrisin özdeğer ve özvektörleri ile karşılaştırma sonuçları yazdırılır.

---

## 4. Sonuç ve Değerlendirme

- Hem manuel (analitik) yöntemle hem de NumPy ile elde edilen özdeğer ve özvektörler tutarlıdır.
- NumPy’nın `linalg.eig` fonksiyonu, yüksek boyutlu ve karmaşık matrislerde hızlı ve güvenilir sonuçlar sağlar.
- Özdeğer ve özvektörler, makine öğrenmesinde boyut indirgeme, veri analizi ve model optimizasyonu gibi alanlarda temel araçlardır.

---

## 5. Kaynaklar

- [Introduction to Matrices in Machine Learning](https://machinelearningmastery.com/introduction-matrices-machine-learning/)
- [Introduction to Eigendecomposition, Eigenvalues and Eigenvectors](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/)
- [NumPy `linalg.eig` Dokümantasyonu](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)
- [NumPy Linalg Kaynak Kodları (GitHub)](https://github.com/numpy/numpy/tree/main/numpy/linalg)
- [LucasBN/Eigenvalues-and-Eigenvectors](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors)

