# Şarap Kalite Sınıflandırma Projesi

Bu proje, şarapların çeşitli özelliklerini kullanarak kalite sınıflandırması yapmaktadır. Karar ağaçları, Gradient Boosting ve AdaBoost gibi makine öğrenmesi modelleri kullanılmaktadır.

## Veri Seti Hakkında

`wines_SPA.csv` veri seti İspanyol şarapları hakkında detaylı bilgiler içermektedir. Veri seti Kaggle'dan alınmıştır ve aşağıdaki özellikleri içermektedir:

- winery: Şaraphane adı
- wine: Şarap adı
- year: Üretim yılı
- rating: Şarap puanı (4.1-5.0 arası)
- num_reviews: Değerlendirme sayısı
- country: Üretildiği ülke
- region: Üretildiği bölge
- price: Fiyat
- type: Şarap tipi
- body: Şarabın gövdesi
- acidity: Asitlik seviyesi

Veri seti şarapları kalitelerine göre dört kategoriye ayırmaktadır:

- İyi (4.1-4.3)
- Çok İyi (4.3-4.5)
- Mükemmel (4.5-4.7)
- Üstün (4.7-5.0)

## Kurulum Gereksinimleri

1. Veri seti için `data` klasörü oluşturulmalıdır:

   ```bash
   mkdir data
   ```

2. Veri setleri `data` klasörüne yerleştirilmelidir:

   - `wines_SPA.csv`

3. Model görselleştirmeleri için `output_png` klasörü oluşturulmalıdır:
   ```bash
   mkdir output_png
   ```

## Çıktılar

Program çalıştırıldığında aşağıdaki görseller `output_png` klasöründe oluşturulacaktır:

- `karar_agaci.png`: Karar ağacı görselleştirmesi
- `learning_curve_Gradient_Boosting.png`: Gradient Boosting öğrenme eğrisi
- `learning_curve_AdaBoost.png`: AdaBoost öğrenme eğrisi
- `depth_analysis.png`: Derinlik analizi grafiği
- `knn_confusion_matrix.png`: KNN karmaşıklık matrisi
- `svm_confusion_matrix.png`: SVM karmaşıklık matrisi

## Metrik Analizleri

Program her model için aşağıdaki metrikleri hesaplamaktadır:

- Doğruluk (Accuracy)
- Kesinlik (Precision)
- Duyarlılık (Recall)
- F1-skoru (F1-score)
- Karmaşıklık matrisi (Confusion Matrix)

Her kategori (İyi, Çok İyi, Mükemmel, Üstün) için ayrı ayrı metrik sonuçları görüntülenmektedir.

## Kullanım

Programı çalıştırmak için:

```bash
python3 main.py
```
