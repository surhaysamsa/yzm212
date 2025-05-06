# Linear Regression: Least Squares vs Scikit-learn
---

## Proje Özeti
Bu projede, bir veri seti üzerinde iki farklı doğrusal regresyon modeli eğitildi:
1. **En Küçük Kareler Yöntemi (Least Squares) ile manuel model** (NumPy ile)
2. **Scikit-learn LinearRegression** modeli

Her iki modelin eğitim ve test maliyetleri (Mean Squared Error, MSE) karşılaştırıldı ve sonuçlar görselleştirildi.

---

## Teorik Arka Plan
### Doğrusal Regresyon (Linear Regression)
Doğrusal regresyon, bağımlı bir değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi modelleyen temel bir makine öğrenmesi yöntemidir. Modelin matematiksel formu:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
$$

### En Küçük Kareler Yöntemi (Least Squares)
En küçük kareler yöntemi, model parametrelerini (β'lar) öyle seçer ki, tahmin edilen değerler ile gerçek değerler arasındaki kare farkların toplamı (cost) minimum olur:

$$
\text{Cost} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Parametreler kapalı formülle şu şekilde bulunur:

$$
\theta = (X^T X)^{-1} X^T y
$$

---

## Dosya Açıklamaları
| Dosya                      | Açıklama |
|----------------------------|----------|
| `data_preprocessing.py`    | Veriyi yükler, eksik değerleri işler ve eğitim/teste böler |
| `linear_regression_manual.py` | NumPy ile en küçük kareler yöntemiyle doğrusal regresyon |
| `linear_regression_sklearn.py` | Scikit-learn ile doğrusal regresyon |
| `compare_models.py`        | Modellerin MSE karşılaştırması ve sonuçların çıktısı |

---

## Kullanım Adımları
> **Gereksinimler:** Python 3.x, pandas, numpy, scikit-learn, matplotlib

1. **Veri Ön İşleme**
   ```bash
   python data_preprocessing.py
   ```
2. **Manuel Model Eğitimi (Least Squares)**
   ```bash
   python linear_regression_manual.py
   ```
3. **Scikit-learn Model Eğitimi**
   ```bash
   python linear_regression_sklearn.py
   ```
4. **Karşılaştırma ve Sonuç**
   ```bash
   python compare_models.py
   ```

---

## Görselleştirme
Aşağıdaki kod ile tahminler ve gerçek değerler karşılaştırmalı olarak çizdirilebilir:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Gerçek değerler
y_test = pd.read_csv('y_test.csv').values.ravel()
# Manuel ve sklearn tahminleri
manual_preds = pd.read_csv('manual_test_preds.csv', header=None).values.ravel()
sklearn_preds = pd.read_csv('sklearn_test_preds.csv').values.ravel()

plt.figure(figsize=(10,5))
plt.plot(y_test, label='Gerçek Değerler', marker='o', alpha=0.7)
plt.plot(manual_preds, label='Manuel Least Squares', marker='x', alpha=0.7)
plt.plot(sklearn_preds, label='Scikit-learn', marker='s', alpha=0.7)
plt.title('Test Seti: Gerçek ve Tahmin Edilen Değerler')
plt.xlabel('Örnek')
plt.ylabel('Hedef Değer')
plt.legend()
plt.tight_layout()
plt.savefig('regression_comparison.png')
plt.show()
```
> Çalıştırdıktan sonra proje klasöründe `regression_comparison.png` dosyası oluşacaktır. Bunu raporunuza ekleyebilirsiniz.

---

## Sonuçlar ve Yorum
| Model                        | Train MSE   | Test MSE    |
|------------------------------|-------------|-------------|
| Manuel Least Squares         | 69141.82    | 87603.24    |
| Scikit-learn LinearRegression| 2.33        | 2.39        |

- **Scikit-learn modeli**, veri ön işleme ve otomatik optimizasyon sayesinde çok daha düşük hata ile çalışmıştır.
- Manuel modelde veri tipleri veya hedef sütunun yanlış seçilmesi gibi hatalar yüksek maliyete yol açabilir.
- Görselleştirme ile modellerin tahmin başarısı kolayca gözlemlenebilir.

---

## Kaynaklar
- [Scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Wikipedia: Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [NumPy Documentation](https://numpy.org/doc/)

---

**Hazırlayan:** Mustafa Surhay Samsa
