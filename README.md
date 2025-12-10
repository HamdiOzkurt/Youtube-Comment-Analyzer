#YouTube Comment Analyzer

YouTube videolarından toplu yorum çekme ve makine öğrenmesi ile duygu analizi & sınıflandırma yapan Python projesi.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Selenium](https://img.shields.io/badge/Selenium-4.0+-green.svg)
![BERT](https://img.shields.io/badge/BERT-Turkish-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

##Özellikler

- 🔍 **Otomatik Video Arama**: Selenium ile YouTube'da arama yaparak video URL'lerini toplar
- 💬 **Toplu Yorum Çekme**: yt-dlp ile hızlı ve paralel yorum çekme
- 🧹 **Veri Ön İşleme**: Yorumları temizleme, normalize etme
- 🤖 **Makine Öğrenmesi Sınıflandırma**: 
  - Şarkıya Dair Yorum
  - Sanatçıya Dair Yorum
  - Genel Yorum
- 😊 **Duygu Analizi**: BERT tabanlı Türkçe duygu analizi (Positive/Negative)
- 📊 **Çoklu Model Karşılaştırma**: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost ve daha fazlası

##Kurulum

### Gereksinimler
- Python 3.10+
- Chrome tarayıcı (Selenium için)
- CUDA destekli GPU (opsiyonel, duygu analizi için hızlandırma)

###Adımlar

```bash
# Repo'yu klonla
git clone https://github.com/HamdiOzkurt/youtube-comment-analyzer.git
cd youtube-comment-analyzer

# Sanal ortam oluştur
python -m venv venv

# Aktifleştir (Windows)
venv\Scripts\activate

# Aktifleştir (Linux/Mac)
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# ML modelleri için ek bağımlılıklar
pip install scikit-learn xgboost lightgbm catboost transformers torch
```

##Kullanım

### 1. Yorum Çekme (Interactive Mode)

```bash
python main.py
```

Ardından:
- Arama kelimesini girin (örn: "Müslüm Gürses")
- Dil seçin (tr, en, vb.)
- Video sayısını belirleyin
- Video başına yorum limitini ayarlayın

### 2. Veri Ön İşleme

`data_preprocessing.ipynb` notebook'unu açın:
- Yorumları temizleme
- Emoji ve özel karakterleri kaldırma
- Stop words temizleme
- TF-IDF vektörizasyonu

### 3. Makine Öğrenmesi & Duygu Analizi

`machine_learning.ipynb` notebook'unu açın:
- Model eğitimi ve karşılaştırma
- GridSearchCV ile hyperparameter tuning
- BERT ile duygu analizi
- Sonuçları Excel'e kaydetme

##Proje Yapısı

```
youtube-comment-analyzer/
├── main.py                    # Ana program (interaktif mod)
├── search_worker.py           # Selenium ile video arama
├── comment_worker.py          # yt-dlp ile yorum çekme
├── data_manager.py            # Veri kaydetme/yükleme
├── config.py                  # Konfigürasyon ayarları
├── Comment_clasfication.py    # Yorum sınıflandırma modülü
├── data_preprocessing.ipynb   # Veri ön işleme notebook'u
├── machine_learning.ipynb     # ML modelleri notebook'u
├── requirements.txt           # Python bağımlılıkları
└── output/                    # Çıktı dosyaları (CSV, JSON, Excel)
```

##Kullanılan Modeller

### Sınıflandırma Modelleri
| Model | Açıklama |
|-------|----------|
| Logistic Regression | En iyi performansı gösteren model |
| Random Forest | Ensemble öğrenme |
| XGBoost | Gradient boosting |
| LightGBM | Hızlı gradient boosting |
| CatBoost | Kategorik veri desteği |
| SVM | Destek vektör makineleri |
| KNN | K-en yakın komşu |

### Duygu Analizi
- **Model**: `savasy/bert-base-turkish-sentiment-cased`
- **Çıktı**: Positive / Negative + Confidence Score

##Örnek Çıktı

```
Toplam Video: 50
Toplam Yorum: 130,000+
Kaydedilen Dosyalar:
   • CSV: muslum_gurses_tr_20251203.csv
   • JSON: muslum_gurses_tr_20251203.json
   • EXCEL: muslum_gurses_tr_20251203.xlsx
```

##Konfigürasyon

`config.py` dosyasından ayarları özelleştirebilirsiniz:

```python
# Paralel işlem sayısı
PARALLEL_WORKERS = 5

# Video başına maksimum yorum
MAX_COMMENTS_PER_VIDEO = 100

# Çıktı dizini
OUTPUT_DIR = "output"
```

##Notlar

- Büyük veri setleri için GPU kullanımı önerilir
- YouTube API limitlerine dikkat edin
- Yorumlar Türkçe için optimize edilmiştir

##Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit'leyin (`git commit -m 'Add amazing feature'`)
4. Push'layın (`git push origin feature/amazing-feature`)
5. Pull Request açın

##Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

##Geliştirici

**Hamdi Özkurt**
- GitHub: [@HamdiOzkurt](https://github.com/HamdiOzkurt)
- Email: hamdi.ozkurt@ogr.sakarya.edu.tr

---

Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
