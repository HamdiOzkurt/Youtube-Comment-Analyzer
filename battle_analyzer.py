"""
Battle Analyzer - Kategori Bazlı Video Karşılaştırma
Ollama kullanarak yorumları kullanıcı tanımlı kategorilere sınıflandırır
"""

import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import requests


@dataclass
class CategoryResult:
    """Kategori sınıflandırma sonucu"""
    category_name: str
    matched_comments: List[str]
    match_count: int
    match_percentage: float


@dataclass
class BattleResult:
    """İki video karşılaştırma sonucu"""
    video1_title: str
    video2_title: str
    video1_total_comments: int
    video2_total_comments: int
    categories: Dict[str, Dict]  # {category: {v1_count, v1_pct, v2_count, v2_pct, v1_samples, v2_samples}}
    winner: str
    summary: str
    # Yeni: Her yorum için kategori atamaları
    v1_classifications: List[Dict]  # [{comment, cat1: 0/1, cat2: 0/1, ...}]
    v2_classifications: List[Dict]  # [{comment, cat1: 0/1, cat2: 0/1, ...}]


class BattleAnalyzer:
    """Kategori bazlı video karşılaştırma analizi"""
    
    BATCH_SIZE = 5  # Aynı anda kaç yorum gönderilecek
    
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://localhost:11434", use_gpu: bool = True):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.use_gpu = use_gpu
    
    def _call_ollama(self, prompt: str, max_tokens: int = 100) -> str:
        """Ollama API çağrısı (GPU destekli)"""
        try:
            options = {
                "num_predict": max_tokens,
                "temperature": 0.1
            }
            
            # GPU kullanımı için num_gpu ekle
            if self.use_gpu:
                options["num_gpu"] = 999  # Tüm GPU katmanlarını kullan
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            print(f"Ollama API hatası: {e}")
            return ""
    
    def classify_batch(self, comments: List[str], category_name: str, category_description: str) -> List[bool]:
        """
        Birden fazla yorumu aynı anda sınıflandır (3 yorum batch)
        Returns: Her yorum için True/False listesi
        """
        if not comments:
            return []
        
        # Yorumları numaralandır
        numbered_comments = "\n".join([f"{i+1}. \"{c[:150]}\"" for i, c in enumerate(comments)])
        
        prompt = f"""Aşağıdaki yorumların her birini "{category_name}" kategorisine uyup uymadığına göre sınıflandır.

KATEGORİ: {category_name}
AÇIKLAMA: {category_description}

YORUMLAR:
{numbered_comments}

Her yorum için sadece numarasını ve E (evet) veya H (hayır) yaz.
Örnek format:
1:E
2:H
3:E"""

        response = self._call_ollama(prompt, max_tokens=50)
        
        # Parse response
        results = [False] * len(comments)
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                try:
                    parts = line.split(":")
                    idx = int(parts[0].strip()) - 1  # 0-indexed
                    answer = parts[1].strip().upper()
                    if 0 <= idx < len(comments):
                        results[idx] = "E" in answer or "Y" in answer or "1" in answer
                except:
                    pass
        
        return results
    
    def classify_single_comment(self, comment: str, category_name: str, category_description: str) -> bool:
        """Tek bir yorumu kategoriye göre sınıflandır (fallback)"""
        prompt = f"""Aşağıdaki yorumu "{category_name}" kategorisine ait olup olmadığını belirle.

YORUM: "{comment[:300]}"

KATEGORİ: {category_name}
AÇIKLAMA: {category_description}

Bu yorum bu kategoriye uyuyor mu? Sadece "EVET" veya "HAYIR" yaz."""

        response = self._call_ollama(prompt, max_tokens=10)
        return "EVET" in response.upper() or "YES" in response.upper() or "1" in response

    
    def classify_comments_batch(
        self, 
        comments: List[str], 
        category_name: str,
        category_description: str,
        max_samples: int = 50,
        progress_callback: Optional[Callable] = None
    ) -> CategoryResult:
        """Yorum listesini bir kategoriye göre toplu sınıflandır"""
        matched = []
        sample_comments = comments[:max_samples]  # İlk N yorumu analiz et
        
        for i, comment in enumerate(sample_comments):
            if progress_callback:
                progress_callback(f"Sınıflandırma: {i+1}/{len(sample_comments)} - {category_name}")
            
            if self.classify_single_comment(comment, category_name, category_description):
                matched.append(comment)
        
        match_pct = (len(matched) / len(sample_comments) * 100) if sample_comments else 0
        
        return CategoryResult(
            category_name=category_name,
            matched_comments=matched,
            match_count=len(matched),
            match_percentage=match_pct
        )
    
    def compare_videos(
        self,
        video1_comments: List[str],
        video2_comments: List[str],
        video1_title: str,
        video2_title: str,
        categories: Dict[str, str],  # {name: description}
        max_comments_per_video: int = 30,
        progress_callback: Optional[Callable] = None
    ) -> BattleResult:
        """İki videoyu kategorilere göre karşılaştır"""
        
        category_results = {}
        v1_total_score = 0
        v2_total_score = 0
        
        # Her yorum için sınıflandırma matrisi
        v1_sample = video1_comments[:max_comments_per_video]
        v2_sample = video2_comments[:max_comments_per_video]
        
        v1_classifications = [{"yorum": c[:100]} for c in v1_sample]
        v2_classifications = [{"yorum": c[:100]} for c in v2_sample]
        
        for cat_name, cat_desc in categories.items():
            if progress_callback:
                progress_callback(f"📊 Kategori: {cat_name}")
            
            # Video 1 sınıflandırma - BATCH işleme (3 yorum aynı anda)
            v1_matched = []
            total_v1 = len(v1_sample)
            batch_size = self.BATCH_SIZE
            
            for batch_start in range(0, total_v1, batch_size):
                batch_end = min(batch_start + batch_size, total_v1)
                batch_comments = v1_sample[batch_start:batch_end]
                
                if progress_callback:
                    progress_callback(f"🔵 V1 [{cat_name[:10]}]: {batch_end}/{total_v1}")
                
                # Batch sınıflandırma
                batch_results = self.classify_batch(batch_comments, cat_name, cat_desc)
                
                for j, is_match in enumerate(batch_results):
                    idx = batch_start + j
                    v1_classifications[idx][cat_name] = 1 if is_match else 0
                    if is_match:
                        v1_matched.append(v1_sample[idx])
            
            if progress_callback:
                progress_callback(f"✅ V1 [{cat_name[:10]}] tamamlandı: {len(v1_matched)}/{total_v1} eşleşme")
            
            # Video 2 sınıflandırma - BATCH işleme (3 yorum aynı anda)
            v2_matched = []
            total_v2 = len(v2_sample)
            
            for batch_start in range(0, total_v2, batch_size):
                batch_end = min(batch_start + batch_size, total_v2)
                batch_comments = v2_sample[batch_start:batch_end]
                
                if progress_callback:
                    progress_callback(f"🟣 V2 [{cat_name[:10]}]: {batch_end}/{total_v2}")
                
                # Batch sınıflandırma
                batch_results = self.classify_batch(batch_comments, cat_name, cat_desc)
                
                for j, is_match in enumerate(batch_results):
                    idx = batch_start + j
                    v2_classifications[idx][cat_name] = 1 if is_match else 0
                    if is_match:
                        v2_matched.append(v2_sample[idx])
            
            if progress_callback:
                progress_callback(f"✅ V2 [{cat_name[:10]}] tamamlandı: {len(v2_matched)}/{total_v2} eşleşme")
            
            v1_pct = (len(v1_matched) / len(v1_sample) * 100) if v1_sample else 0
            v2_pct = (len(v2_matched) / len(v2_sample) * 100) if v2_sample else 0
            
            category_results[cat_name] = {
                'v1_count': len(v1_matched),
                'v1_percent': v1_pct,
                'v2_count': len(v2_matched),
                'v2_percent': v2_pct,
                'v1_samples': v1_matched[:3],
                'v2_samples': v2_matched[:3]
            }
            
            # Basit skor hesaplama
            v1_total_score += v1_pct
            v2_total_score += v2_pct
        
        # Kazanan belirleme
        if v1_total_score > v2_total_score:
            winner = video1_title
        elif v2_total_score > v1_total_score:
            winner = video2_title
        else:
            winner = "Berabere"
        
        # Özet oluştur
        summary = self._generate_summary(
            video1_title, video2_title, 
            category_results, winner
        )
        
        return BattleResult(
            video1_title=video1_title,
            video2_title=video2_title,
            video1_total_comments=len(video1_comments),
            video2_total_comments=len(video2_comments),
            categories=category_results,
            winner=winner,
            summary=summary,
            v1_classifications=v1_classifications,
            v2_classifications=v2_classifications
        )
    
    def _generate_summary(
        self, 
        v1_title: str, 
        v2_title: str, 
        results: Dict, 
        winner: str
    ) -> str:
        """Karşılaştırma özeti oluştur"""
        summary_parts = []
        
        for cat_name, data in results.items():
            v1_pct = data['v1_percent']
            v2_pct = data['v2_percent']
            
            if v1_pct > v2_pct:
                summary_parts.append(f"• **{cat_name}**: {v1_title[:20]} önde (%{v1_pct:.1f} vs %{v2_pct:.1f})")
            elif v2_pct > v1_pct:
                summary_parts.append(f"• **{cat_name}**: {v2_title[:20]} önde (%{v2_pct:.1f} vs %{v1_pct:.1f})")
            else:
                summary_parts.append(f"• **{cat_name}**: Eşit (%{v1_pct:.1f})")
        
        return "\n".join(summary_parts)
    
    def check_connection(self) -> bool:
        """Ollama bağlantısını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Test kodu
if __name__ == '__main__':
    print("=" * 60)
    print("⚔️ BATTLE ANALYZER TEST")
    print("=" * 60)
    
    analyzer = BattleAnalyzer()
    
    if analyzer.check_connection():
        print("✅ Ollama bağlantısı başarılı!")
        
        # Test kategorileri
        categories = {
            "Olumlu Geri Bildirim": "Videoyu beğenen, övgü içeren yorumlar",
            "Eleştiri": "Olumsuz, eleştirel veya şikayet içeren yorumlar"
        }
        
        # Test yorumları
        v1_comments = [
            "Harika bir video olmuş!",
            "Çok beğendim, devamını bekliyoruz",
            "Ses kalitesi kötü maalesef"
        ]
        
        v2_comments = [
            "Süper içerik",
            "Biraz uzun olmuş",
            "Mükemmel açıklamalar"
        ]
        
        print("\n🔍 Test karşılaştırması yapılıyor...")
        result = analyzer.compare_videos(
            v1_comments, v2_comments,
            "Test Video 1", "Test Video 2",
            categories
        )
        
        print(f"\n🏆 Kazanan: {result.winner}")
        print(f"\n📊 Özet:\n{result.summary}")
    else:
        print("❌ Ollama bağlantısı kurulamadı!")
        print("Çözüm: ollama serve komutunu çalıştırın")
