"""
Ollama LLM Wrapper - Local AI Model Support
Gemini yerine local Ollama kullanımı için
"""

import requests
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class OllamaSummaryResult:
    """Ollama özet sonucu"""
    summary: str
    raw_response: str


class OllamaLLM:
    """Local Ollama ile yorum özetleme"""
    
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
        """
        Args:
            model_name: Kullanılacak Ollama modeli
            base_url: Ollama API URL'i
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def _call_ollama(self, prompt: str, max_tokens: int = 1000) -> str:
        """Ollama API'ye istek gönder"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Ollama'ya bağlanılamadı! Lütfen Ollama'nın çalıştığından emin olun.\n"
                f"Başlatmak için: ollama serve\n"
                f"Model indirmek için: ollama pull {self.model_name}"
            )
        except Exception as e:
            raise Exception(f"Ollama API hatası: {e}")
    
    def summarize_comments(
        self, 
        comments: List[str], 
        video_title: str = "",
        sentiment_distribution: Optional[dict] = None
    ) -> OllamaSummaryResult:
        """
        Yorumları özetle
        
        Args:
            comments: Yorum metinleri listesi
            video_title: Video başlığı (opsiyonel)
            sentiment_distribution: Sentiment analizi sonuçları (opsiyonel)
                Örnek: {"positive": 26, "negative": 74, "neutral": 0}
            
        Returns:
            OllamaSummaryResult objesi
        """
        if not comments:
            return OllamaSummaryResult(
                summary="Analiz edilecek yorum bulunamadı.",
                raw_response=""
            )
        
        # İlk 100 yorumu al (Ollama için)
        comments_sample = comments[:100]
        comments_text = "\n".join([f"- {c[:300]}" for c in comments_sample])
        
        # Sentiment bilgisini prompt'a ekle
        sentiment_context = ""
        if sentiment_distribution:
            pos = sentiment_distribution.get('positive', 0)
            neg = sentiment_distribution.get('negative', 0)
            neu = sentiment_distribution.get('neutral', 0)
            
            # Dominant sentiment'i belirle
            if neg > pos and neg > neu:
                dominant = f"ÇOĞUNLUKLA NEGATİF (Negatif: %{neg}, Pozitif: %{pos}, Nötr: %{neu})"
            elif pos > neg and pos > neu:
                dominant = f"ÇOĞUNLUKLA POZİTİF (Pozitif: %{pos}, Negatif: %{neg}, Nötr: %{neu})"
            else:
                dominant = f"KARIŞIK (Pozitif: %{pos}, Negatif: %{neg}, Nötr: %{neu})"
            
            sentiment_context = f"\n⚠️ SENTIMENT ANALİZ SONUCU: {dominant}\n"
        
        prompt = f"""Sen bir YouTube video analiz uzmanısın. Aşağıdaki yorumları analiz et ve Türkçe özet çıkar.

{"Video: " + video_title if video_title else ""}
{sentiment_context}
YORUMLAR:
{comments_text}

GÖREV: Bu yorumları analiz ederek aşağıdaki bilgileri ver:

1. GENEL ÖZET (2-3 cümle): İzleyicilerin genel tepkisi nedir? (Sentiment analizi sonucuna dikkat et!)

2. ANA NOKTALAR (3-5 madde): En çok vurgulanan konular

3. DUYGU ANALİZİ: Genel atmosfer - Yukarıdaki sentiment dağılımını doğrula

4. ÖNERİLER (2-3 madde): İçerik üreticiye öneriler

Kısa ve öz yanıt ver. Sentiment analizi sonucuyla uyumlu bir özet yaz!"""

        try:
            response_text = self._call_ollama(prompt, max_tokens=800)
            
            return OllamaSummaryResult(
                summary=response_text,
                raw_response=response_text
            )
            
        except Exception as e:
            print(f"❌ Ollama özet hatası: {e}")
            return OllamaSummaryResult(
                summary=f"Hata: {str(e)}",
                raw_response=""
            )
    
    def check_connection(self) -> bool:
        """Ollama bağlantısını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Mevcut modelleri listele"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []

    def summarize_video_description(self, description: str) -> str:
        """Video açıklamasını özetle ve içeriği çıkar"""
        if not description:
            return "Video açıklaması bulunamadı."
        
        prompt = f"""Aşağıdaki YouTube video açıklamasını analiz et ve videonun konusunu akıcı bir dille özetle.
        
        AÇIKLAMA:
        {description[:2500]}
        
        GÖREV:
        Bu videonun ne hakkında olduğunu 3-4 cümleyi geçmeyecek şekilde, tek bir paragraf halinde özetle.
        
        KURALLAR:
        - Kesinlikle madde işareti (bullet point) kullanma.
        - Listeleme yapma.
        - Akıcı bir Türkçe kullan.
        - Sadece özeti yaz, "İşte özet:" gibi başlangıçlar yapma.
        """
        
        try:
            return self._call_ollama(prompt, max_tokens=300)
        except Exception as e:
            return f"Özetlenemedi: {e}"


# Test
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 OLLAMA LLM TEST")
    print("=" * 60)
    
    ollama = OllamaLLM(model_name="gemma3:4b")
    
    # Bağlantı kontrolü
    if ollama.check_connection():
        print("✅ Ollama bağlantısı başarılı!")
        
        models = ollama.list_models()
        print(f"\n📋 Mevcut modeller: {', '.join(models)}")
        
        # Test yorumları
        test_comments = [
            "Bu video harika olmuş, çok beğendim!",
            "Açıklamalar net ve anlaşılır",
            "Devam videoları bekliyoruz",
            "Ses kalitesi biraz düşük olmuş",
            "10 numara içerik!"
        ]
        
        print("\n📝 Test özetlemesi...")
        result = ollama.summarize_comments(test_comments, "Test Video")
        print(f"\n{result.summary}")
    else:
        print("❌ Ollama'ya bağlanılamadı!")
        print("\nÇözüm:")
        print("  1. Ollama'yı başlatın: ollama serve")
        print("  2. Model indirin: ollama pull gemma3:4b")
