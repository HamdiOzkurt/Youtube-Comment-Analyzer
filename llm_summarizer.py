"""
LLM Summarizer - Ollama ile Yorum Özetleme
(Google Gemini desteği kaldırıldı, yerel AI kullanılıyor)
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

# Ollama import
try:
    from ollama_llm import OllamaLLM, OllamaSummaryResult
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class SummaryResult:
    """Özet sonucu"""
    summary: str
    key_points: List[str]
    questions_from_audience: List[str]
    sentiment_overview: str
    content_suggestions: List[str]
    raw_response: str


class LLMSummarizer:
    """Ollama ile yorum özetleme (Gemini desteği kaldırıldı)"""
    
    def __init__(self, model_name: str = "gemma3:4b"):
        """
        Args:
            model_name: Kullanılacak Ollama modeli
        """
        self.model_name = model_name
        self._ollama = None
        
        if not OLLAMA_AVAILABLE:
            print("⚠️ Ollama modülü bulunamadı! ollama_llm.py dosyasını kontrol edin.")
    
    def _get_ollama(self) -> Optional[OllamaLLM]:
        """Lazy loading ile Ollama instance al"""
        if self._ollama is None and OLLAMA_AVAILABLE:
            self._ollama = OllamaLLM(model_name=self.model_name)
        return self._ollama
    
    def summarize_comments(self, 
                          comments: List[str], 
                          video_title: str = "") -> SummaryResult:
        """
        Yorumları özetle
        
        Args:
            comments: Yorum metinleri listesi
            video_title: Video başlığı (opsiyonel)
            
        Returns:
            SummaryResult objesi
        """
        if not comments:
            return SummaryResult(
                summary="Analiz edilecek yorum bulunamadı.",
                key_points=[],
                questions_from_audience=[],
                sentiment_overview="Belirsiz",
                content_suggestions=[],
                raw_response=""
            )
        
        ollama = self._get_ollama()
        if not ollama:
            return SummaryResult(
                summary="Ollama bağlantısı kurulamadı. 'ollama serve' komutunu çalıştırın.",
                key_points=[],
                questions_from_audience=[],
                sentiment_overview="Hata",
                content_suggestions=[],
                raw_response=""
            )
        
        try:
            # OllamaLLM kullanarak özet al
            result = ollama.summarize_comments(comments, video_title)
            
            # Parse et
            raw_text = result.summary
            
            return SummaryResult(
                summary=raw_text,
                key_points=self._extract_bullet_points(raw_text, "ANA NOKTALAR"),
                questions_from_audience=self._extract_bullet_points(raw_text, "SORULAR"),
                sentiment_overview=self._extract_section(raw_text, "DUYGU"),
                content_suggestions=self._extract_bullet_points(raw_text, "ÖNERİLER"),
                raw_response=raw_text
            )
            
        except Exception as e:
            print(f"❌ Özet oluşturma hatası: {e}")
            return SummaryResult(
                summary=f"Hata: {str(e)}",
                key_points=[],
                questions_from_audience=[],
                sentiment_overview="Hata",
                content_suggestions=[],
                raw_response=""
            )
    
    def ask_about_comments(self, comments: List[str], question: str) -> str:
        """
        Yorumlar hakkında soru sor
        
        Args:
            comments: Yorum listesi
            question: Sorulacak soru
            
        Returns:
            Yanıt metni
        """
        if not comments:
            return "Analiz edilecek yorum yok."
        
        ollama = self._get_ollama()
        if not ollama:
            return "Ollama bağlantısı kurulamadı."
        
        try:
            # Soru-cevap için özel prompt
            prompt = f"""Aşağıda bir YouTube videosunun yorumları var:

{chr(10).join([f'- {c[:300]}' for c in comments[:50]])}

SORU: {question}

Bu soruyu yorumlara dayanarak Türkçe olarak yanıtla. Yanıtın kısa ve öz olsun (2-3 cümle)."""
            
            result = ollama._call_ollama(prompt, max_tokens=500)
            return result
            
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def compare_videos(self, 
                      video1_comments: List[str], 
                      video2_comments: List[str],
                      video1_title: str = "Video 1",
                      video2_title: str = "Video 2") -> str:
        """
        İki videonun yorumlarını karşılaştır (Battle Mode)
        """
        ollama = self._get_ollama()
        if not ollama:
            return "Ollama bağlantısı kurulamadı."
        
        v1_text = "\n".join([f"- {c[:200]}" for c in video1_comments[:30]])
        v2_text = "\n".join([f"- {c[:200]}" for c in video2_comments[:30]])
        
        prompt = f"""İki YouTube videosunun yorumlarını karşılaştır:

**{video1_title} YORUMLARI:**
{v1_text}

**{video2_title} YORUMLARI:**
{v2_text}

Bu iki videonun izleyici tepkilerini karşılaştırarak Türkçe bir analiz yap:

1. Hangisi daha pozitif karşılanmış?
2. Her birinin güçlü yanları neler?
3. Her birinin eleştiri aldığı noktalar neler?
4. Kazanan hangisi ve neden?

Kısa ve öz yanıtla (maksimum 200 kelime)."""
        
        try:
            result = ollama._call_ollama(prompt, max_tokens=800)
            return result
        except Exception as e:
            return f"Karşılaştırma hatası: {str(e)}"
    
    def _extract_bullet_points(self, text: str, section_name: str) -> List[str]:
        """Metinden bullet point'leri çıkar"""
        import re
        
        points = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            
            if in_section:
                # Yeni bölüm başlığı gelirse dur
                if '**' in line and ':' in line:
                    break
                
                # Bullet point'i al
                line = line.strip()
                if line.startswith(('-', '•', '*', '–')):
                    point = re.sub(r'^[-•*–]\s*', '', line)
                    if point and len(point) > 5:
                        points.append(point)
                elif line and len(line) > 10 and not line.startswith('#'):
                    points.append(line)
        
        return points[:5]  # Maksimum 5 madde
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Metinden bir bölümü çıkar"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if section_name.upper() in line.upper():
                # Sonraki satırı al
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('**'):
                        return next_line
                # Aynı satırda ise
                parts = line.split(':')
                if len(parts) > 1:
                    return ':'.join(parts[1:]).strip()
        
        return ""


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 LLM SUMMARIZER TEST (OLLAMA)")
    print("=" * 60)
    
    # Test yorumları
    test_comments = [
        "Bu şarkı efsane olmuş, dinlemeden geçmeyin!",
        "Sanatçının sesi çok güzel ama klip biraz sönük kalmış",
        "Devamını sabırsızlıkla bekliyorum, lütfen daha çok video çekin",
        "Bu tarz müzikleri çok özlemişiz, teşekkürler",
        "Sözler çok anlamlı, yazara teşekkürler",
    ]
    
    summarizer = LLMSummarizer()
    
    print("\n📝 Yorumlar özetleniyor (Ollama)...\n")
    result = summarizer.summarize_comments(test_comments, "Test Video")
    print(result.summary)
