"""
Topic Modeler - BERTopic ile Dinamik Konu Modelleme
Yorumlardan otomatik konu başlıkları çıkarır
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TopicResult:
    """Konu modelleme sonucu"""
    topic_id: int
    topic_name: str
    keywords: List[str]
    document_count: int
    representative_docs: List[str]


class TopicModeler:
    """BERTopic tabanlı konu modelleme sınıfı"""
    
    def __init__(self, 
                 language: str = "turkish",
                 min_topic_size: int = 5,
                 nr_topics: Optional[int] = None):
        """
        Args:
            language: Dil (turkish, english vb.)
            min_topic_size: Minimum konu boyutu
            nr_topics: İstenen konu sayısı (None = otomatik)
        """
        self.language = language
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.model = None
        self.topics = None
        self.topic_info = None
        self._initialized = False
    
    def _init_model(self):
        """Modeli lazy loading ile başlat"""
        if self._initialized:
            return
        
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            print("📥 BERTopic modeli yükleniyor...")
            
            # Türkçe için embedding modeli
            if self.language == "turkish":
                embedding_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
            else:
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            self.model = BERTopic(
                embedding_model=embedding_model,
                language=self.language,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics,
                verbose=False
            )
            
            self._initialized = True
            print("✅ BERTopic modeli hazır!")
            
        except ImportError as e:
            raise ImportError(f"Gerekli paket yüklü değil: {e}\npip install bertopic sentence-transformers")
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {e}")
    
    def fit_transform(self, documents: List[str]) -> Tuple[List[int], List[float]]:
        """
        Dokümanları modelleyip konu ata
        
        Args:
            documents: Yorum/metin listesi
            
        Returns:
            (topic_ids, probabilities) tuple'ı
        """
        self._init_model()
        
        if not documents or len(documents) < self.min_topic_size:
            print(f"⚠️ Minimum {self.min_topic_size} doküman gerekli!")
            return [], []
        
        print(f"🔄 {len(documents)} doküman analiz ediliyor...")
        
        # Boş dokümanları filtrele
        valid_docs = [doc for doc in documents if doc and len(doc.strip()) > 10]
        
        if len(valid_docs) < self.min_topic_size:
            print("⚠️ Yeterli geçerli doküman yok!")
            return [], []
        
        self.topics, probs = self.model.fit_transform(valid_docs)
        self.topic_info = self.model.get_topic_info()
        
        print(f"✅ {len(self.topic_info) - 1} konu tespit edildi!")  # -1 için outlier topic
        
        return self.topics, probs
    
    def get_topics(self) -> List[TopicResult]:
        """Tespit edilen konuları döndür"""
        if self.topic_info is None:
            return []
        
        results = []
        
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            
            # Outlier topic'i atla
            if topic_id == -1:
                continue
            
            # Konu kelimelerini al
            topic_words = self.model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:10]] if topic_words else []
            
            # Temsilci dokümanları al
            try:
                rep_docs = self.model.get_representative_docs(topic_id)
            except:
                rep_docs = []
            
            results.append(TopicResult(
                topic_id=topic_id,
                topic_name=row.get('Name', f'Topic_{topic_id}'),
                keywords=keywords,
                document_count=row.get('Count', 0),
                representative_docs=rep_docs[:3] if rep_docs else []
            ))
        
        return results
    
    def get_topic_for_document(self, document: str) -> int:
        """Yeni bir doküman için konu tahmin et"""
        if not self._initialized or self.model is None:
            return -1
        
        try:
            topics, _ = self.model.transform([document])
            return topics[0]
        except:
            return -1
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """Konu dağılımını döndür"""
        if self.topic_info is None:
            return {}
        
        distribution = {}
        for _, row in self.topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Outlier'ı atla
                name = row.get('Name', f'Topic_{topic_id}')
                distribution[name] = row.get('Count', 0)
        
        return distribution
    
    def visualize_topics(self, output_path: Optional[str] = None):
        """Konuları görselleştir (HTML olarak kaydet)"""
        if self.model is None:
            print("⚠️ Önce fit_transform() çağrılmalı!")
            return None
        
        try:
            fig = self.model.visualize_topics()
            
            if output_path:
                fig.write_html(output_path)
                print(f"📊 Görselleştirme kaydedildi: {output_path}")
            
            return fig
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")
            return None
    
    def visualize_barchart(self, top_n_topics: int = 10, output_path: Optional[str] = None):
        """Konu anahtar kelimelerini bar chart olarak görselleştir"""
        if self.model is None:
            return None
        
        try:
            fig = self.model.visualize_barchart(top_n_topics=top_n_topics)
            
            if output_path:
                fig.write_html(output_path)
                print(f"📊 Bar chart kaydedildi: {output_path}")
            
            return fig
        except Exception as e:
            print(f"❌ Görselleştirme hatası: {e}")
            return None
    
    def get_summary(self) -> Dict:
        """Model özeti"""
        if self.topic_info is None:
            return {}
        
        topics = self.get_topics()
        
        return {
            'total_topics': len(topics),
            'topics': [
                {
                    'id': t.topic_id,
                    'name': t.topic_name,
                    'keywords': t.keywords[:5],
                    'doc_count': t.document_count
                }
                for t in topics
            ]
        }


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("🎯 TOPIC MODELER TEST")
    print("=" * 60)
    
    # Test yorumları (minimum 10+ doküman gerekli)
    test_comments = [
        "Bu şarkının sözleri çok anlamlı, yazara teşekkürler",
        "Şarkı sözleri muhteşem, her dinlediğimde farklı şeyler hissediyorum",
        "Müziğin melodisi çok güzel, kulağa hoş geliyor",
        "Ritim ve melodi harika uyum sağlamış",
        "Sanatçının sesi çok etkileyici",
        "Vokal performansı mükemmel",
        "Klipteki görüntüler çok kaliteli",
        "Video prodüksiyonu profesyonelce yapılmış",
        "Bu tarz müzikleri çok seviyorum",
        "Rock müzik en iyisi, devamını bekliyorum",
        "Pop müziğe güzel bir yorum",
        "Enstrüman çalışı çok iyi",
        "Gitar solosu efsane olmuş",
        "Davul ritmi çok enerjik",
        "Nostaljik hisler uyandırıyor"
    ]
    
    try:
        modeler = TopicModeler(language="turkish", min_topic_size=3)
        topics, probs = modeler.fit_transform(test_comments)
        
        print("\n📋 BULUNAN KONULAR:\n")
        
        for topic in modeler.get_topics():
            print(f"🏷️ Topic {topic.topic_id}: {topic.topic_name}")
            print(f"   Anahtar Kelimeler: {', '.join(topic.keywords[:5])}")
            print(f"   Doküman Sayısı: {topic.document_count}")
            print()
        
        print("📊 DAĞILIM:")
        print(modeler.get_topic_distribution())
        
    except ImportError as e:
        print(f"\n⚠️ TEST YAPILAMADI: {e}")
        print("Çözüm: pip install bertopic sentence-transformers")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
