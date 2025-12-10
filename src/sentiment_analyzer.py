"""
Sentiment Analyzer - BERT Tabanlı Türkçe Duygu Analizi
savasy/bert-base-turkish-sentiment-cased modeli ile duygu analizi
"""

import torch
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SentimentResult:
    """Duygu analizi sonucu"""
    text: str
    label: str  # 'positive' veya 'negative'
    score: float  # 0.0 - 1.0 arası güven skoru
    raw_scores: Dict[str, float]  # Tüm etiketlerin skorları


class SentimentAnalyzer:
    """BERT tabanlı Türkçe duygu analizi sınıfı"""
    
    MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
    
    def __init__(self, 
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        Args:
            device: 'cuda', 'cpu' veya None (otomatik)
            batch_size: Batch işleme boyutu
            max_length: Maksimum token uzunluğu
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        
        # Device seçimi
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 SentimentAnalyzer başlatılıyor... (Device: {self.device})")
    
    def _load_model(self):
        """Modeli lazy loading ile yükle"""
        if self._initialized:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            print(f"📥 Model yükleniyor: {self.MODEL_NAME}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            
            # GPU'ya taşı
            if self.device == "cuda":
                self.model = self.model.to("cuda")
                print("🚀 GPU (CUDA) kullanılıyor")
            else:
                print("💻 CPU kullanılıyor")
            
            # Pipeline oluştur
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=self.max_length
            )
            
            self._initialized = True
            print("✅ Model başarıyla yüklendi!")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Tek bir metni analiz et
        
        Args:
            text: Analiz edilecek metin
            
        Returns:
            SentimentResult objesi
        """
        self._load_model()
        
        if not text or not isinstance(text, str):
            return SentimentResult(
                text="",
                label="neutral",
                score=0.0,
                raw_scores={}
            )
        
        # Metni kısalt (çok uzunsa)
        text = text[:self.max_length * 4]  # Yaklaşık karakter limiti
        
        try:
            result = self.pipeline(text)[0]
            
            # Label'ı normalize et
            label = result['label'].lower()
            if label in ['positive', 'pos', 'pozitif', 'label_1', '1']:
                normalized_label = 'positive'
            elif label in ['negative', 'neg', 'negatif', 'label_0', '0']:
                normalized_label = 'negative'
            else:
                normalized_label = 'neutral'
            
            return SentimentResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                label=normalized_label,
                score=result['score'],
                raw_scores={result['label']: result['score']}
            )
            
        except Exception as e:
            print(f"⚠️ Analiz hatası: {e}")
            return SentimentResult(
                text=text[:50],
                label="error",
                score=0.0,
                raw_scores={}
            )
    
    def analyze_batch(self, texts: List[str], show_progress: bool = True) -> List[SentimentResult]:
        """
        Birden fazla metni batch olarak analiz et
        
        Args:
            texts: Metin listesi
            show_progress: İlerleme göster
            
        Returns:
            SentimentResult listesi
        """
        self._load_model()
        
        if not texts:
            return []
        
        results = []
        total = len(texts)
        
        # Batch'ler halinde işle
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            if show_progress:
                progress = min(i + self.batch_size, total)
                print(f"📊 İşleniyor: {progress}/{total} ({100*progress/total:.1f}%)")
            
            for text in batch:
                results.append(self.analyze(text))
        
        return results
    
    def get_sentiment_distribution(self, results: List[SentimentResult]) -> Dict[str, int]:
        """Duygu dağılımını hesapla"""
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0, 'error': 0}
        
        for result in results:
            if result.label in distribution:
                distribution[result.label] += 1
        
        return distribution
    
    def get_average_confidence(self, results: List[SentimentResult]) -> float:
        """Ortalama güven skorunu hesapla"""
        if not results:
            return 0.0
        
        valid_scores = [r.score for r in results if r.label not in ['error', 'neutral']]
        
        if not valid_scores:
            return 0.0
        
        return sum(valid_scores) / len(valid_scores)
    
    def filter_by_sentiment(self, 
                           results: List[SentimentResult], 
                           sentiment: str,
                           min_confidence: float = 0.0) -> List[SentimentResult]:
        """Belirli bir duyguya göre filtrele"""
        return [
            r for r in results 
            if r.label == sentiment and r.score >= min_confidence
        ]
    
    def get_summary_stats(self, results: List[SentimentResult]) -> Dict:
        """Özet istatistikler"""
        if not results:
            return {}
        
        distribution = self.get_sentiment_distribution(results)
        total = len(results)
        
        return {
            'total_analyzed': total,
            'positive_count': distribution['positive'],
            'negative_count': distribution['negative'],
            'neutral_count': distribution['neutral'],
            'positive_ratio': distribution['positive'] / total if total > 0 else 0,
            'negative_ratio': distribution['negative'] / total if total > 0 else 0,
            'average_confidence': self.get_average_confidence(results),
            'sentiment_score': (distribution['positive'] - distribution['negative']) / total if total > 0 else 0
        }


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("🧠 BERT SENTIMENT ANALYZER TEST")
    print("=" * 60)
    
    # Analyzer başlat
    analyzer = SentimentAnalyzer()
    
    # Test metinleri
    test_texts = [
        "Bu video harika olmuş, çok beğendim!",
        "Berbat bir içerik, hiç beğenmedim.",
        "Şarkı süper ama klip biraz sıkıcı",
        "Efsane! Kesinlikle izlenmeli!",
        "Vakit kaybı, izlemeyin.",
        "İdare eder, fena değil.",
        "Müthiş performans, tebrikler!",
        "Çok kötü, hayal kırıklığı."
    ]
    
    print("\n📝 TEK TEK ANALİZ:\n")
    
    for text in test_texts[:3]:
        result = analyzer.analyze(text)
        emoji = "✅" if result.label == "positive" else "❌"
        print(f"{emoji} [{result.label.upper()}] ({result.score:.2f}) - {text}")
    
    print("\n" + "=" * 60)
    print("📊 BATCH ANALİZ:")
    print("=" * 60 + "\n")
    
    results = analyzer.analyze_batch(test_texts, show_progress=True)
    
    print("\n📈 ÖZET İSTATİSTİKLER:\n")
    stats = analyzer.get_summary_stats(results)
    
    print(f"Toplam Analiz: {stats['total_analyzed']}")
    print(f"Pozitif: {stats['positive_count']} ({stats['positive_ratio']*100:.1f}%)")
    print(f"Negatif: {stats['negative_count']} ({stats['negative_ratio']*100:.1f}%)")
    print(f"Nötr: {stats['neutral_count']}")
    print(f"Ortalama Güven: {stats['average_confidence']:.2f}")
    print(f"Sentiment Skoru: {stats['sentiment_score']:.2f}")
