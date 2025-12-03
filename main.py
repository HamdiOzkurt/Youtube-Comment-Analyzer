"""
YouTube Toplu Yorum Çekici - Ana Program
Selenium (video arama) + yt-dlp (yorum çekme) entegrasyonu
"""

import sys
from pathlib import Path

# Kendi modüllerimiz
# NOT: Bu dosyalar aynı klasörde olmalı
try:
    from search_worker import SearchWorker  # Selenium worker (verdiğin kod)
    from comment_worker import CommentWorker  # yt-dlp worker
    from data_manager import DataManager  # Veri yönetimi
except ImportError as e:
    print(f"❌ Modül import hatası: {e}")
    print("Lütfen tüm dosyaların aynı klasörde olduğundan emin olun:")
    print("  - search_worker.py")
    print("  - comment_worker.py")
    print("  - data_manager.py")
    sys.exit(1)

from PyQt6.QtCore import QThread, pyqtSignal


class BulkCommentScraper:
    """Ana orkestrasyon sınıfı"""
    
    def __init__(self, output_dir="output"):
        self.data_manager = DataManager(output_dir)
        self.search_results = []
        self.comment_results = []
        
    def scrape_and_extract(self, 
                           search_query, 
                           video_limit=10,
                           max_comments_per_video=None,
                           parallel_workers=3,
                           filter_keywords=None,
                           lang=None):
        """
        Tam iş akışı: Arama → URL toplama → Yorum çekme → Kaydetme
        
        Args:
            search_query: YouTube'da aranacak kelime
            video_limit: Kaç video bulunacak
            max_comments_per_video: Video başına max yorum
            parallel_workers: Paralel işlem sayısı
            filter_keywords: Yorumları filtrelemek için kelimeler (list)
            lang: Arama dili (örn: 'en', 'tr')
        """
        print("\n" + "="*80)
        print(f"🚀 TOPLU YORUM ÇEKME BAŞLATILDI")
        print(f"🔍 Arama Kelimesi: '{search_query}'")
        print(f"🌍 Dil: {lang if lang else 'Varsayılan'}")
        print(f"📹 Video Limiti: {video_limit}")
        print(f"💬 Video Başına Yorum: {max_comments_per_video or 'HEPSI'}")
        print(f"⚙️  Paralel İşlem: {parallel_workers}")
        print("="*80 + "\n")
        
        # ===== 1. ADIM: VIDEO URL'LERİNİ TOPLA (SELENIUM) =====
        print("📡 1. ADIM: Video URL'leri toplanıyor (Selenium)...\n")
        
        # SearchWorker'ı QThread olmadan kullanmak için basit çalıştırma
        # Not: GUI olmadan çalışıyoruz, direkt run() metodunu çağırabiliriz
        search_worker = SearchWorker(query=search_query, limit=video_limit, lang=lang)
        
        # Sinyalleri bağla
        search_worker.search_finished.connect(self._on_search_finished)
        search_worker.search_error.connect(self._on_search_error)
        
        # run() metodunu çağır (blocking)
        search_worker.run()
        
        if not self.search_results:
            print("❌ Hiç video URL'i bulunamadı!")
            return None
        
        print(f"\n✅ {len(self.search_results)} video URL'i toplandı!\n")
        
        # ===== 2. ADIM: YORUMLARI ÇEK (YT-DLP) =====
        print("💬 2. ADIM: Yorumlar çekiliyor (yt-dlp)...\n")
        
        comment_worker = CommentWorker(
            max_workers=parallel_workers,
            max_comments_per_video=max_comments_per_video
        )
        
        self.comment_results = comment_worker.fetch_bulk_comments(self.search_results)
        
        if not self.comment_results:
            print("❌ Hiç yorum çekilemedi!")
            return None
        
        # ===== 3. ADIM: FİLTRELEME (Opsiyonel) =====
        if filter_keywords:
            print(f"\n🔍 3. ADIM: Yorumlar filtreleniyor...")
            print(f"   Anahtar Kelimeler: {', '.join(filter_keywords)}\n")
            
            original_count = sum(len(v['yorumlar']) for v in self.comment_results)
            self.comment_results = self.data_manager.filter_comments_by_keyword(
                self.comment_results, 
                filter_keywords,
                case_sensitive=False
            )
            filtered_count = sum(len(v['yorumlar']) for v in self.comment_results)
            
            print(f"   ✅ {original_count} → {filtered_count} yorum kaldı\n")
        
        # ===== 4. ADIM: İSTATİSTİKLER =====
        print("📊 4. ADIM: İstatistikler hesaplanıyor...\n")
        stats = comment_worker.get_statistics()
        
        # ===== 5. ADIM: KAYDETME =====
        print("💾 5. ADIM: Dosyalara kaydediliyor...\n")
        
        prefix = search_query.replace(' ', '_').lower()
        if lang:
            prefix += f"_{lang}"
            
        saved_files = self.data_manager.save_all_formats(
            self.comment_results, 
            stats, 
            prefix=prefix
        )
        
        # ===== ÖZET =====
        print("\n" + "="*80)
        print("🎉 İŞLEM TAMAMLANDI!")
        print("="*80)
        print(f"📹 Toplam Video: {len(self.comment_results)}")
        print(f"💬 Toplam Yorum: {sum(len(v['yorumlar']) for v in self.comment_results):,}")
        print(f"📁 Kaydedilen Dosyalar:")
        for fmt, path in saved_files.items():
            print(f"   • {fmt.upper()}: {path.name}")
        print("="*80 + "\n")
        
        return {
            'videos': self.comment_results,
            'stats': stats,
            'files': saved_files
        }
    
    def _on_search_finished(self, urls):
        """Selenium arama tamamlandığında çağrılır"""
        self.search_results = urls
    
    def _on_search_error(self, error_msg):
        """Arama hatası olduğunda çağrılır"""
        print(f"❌ ARAMA HATASI: {error_msg}")
        self.search_results = []


def interactive_mode():
    """Kullanıcıdan girdi alarak çalışan mod"""
    print("\n" + "="*60)
    print("🎥 YOUTUBE YORUM ÇEKİCİ - İNTERAKTİF MOD")
    print("="*60 + "\n")
    
    try:
        query = input("🔍 Arama yapılacak kelime/konu: ").strip()
        if not query:
            print("❌ Arama kelimesi boş olamaz!")
            return

        lang = input("🌍 Arama dili? (örn: 'en', 'tr', Boş=Varsayılan): ").strip()
        if not lang:
            lang = None

        limit_str = input("📹 Kaç video taransın? (Varsayılan: 10): ").strip()
        limit = int(limit_str) if limit_str.isdigit() else 10
        
        comments_str = input("💬 Video başına max yorum? (Varsayılan: 100, Hepsi için 'all'): ").strip()
        if comments_str.lower() == 'all':
            max_comments = None
        else:
            max_comments = int(comments_str) if comments_str.isdigit() else 100
            
        workers_str = input("⚙️  Paralel işlem sayısı? (Varsayılan: 5): ").strip()
        workers = int(workers_str) if workers_str.isdigit() else 5
        
        scraper = BulkCommentScraper()
        scraper.scrape_and_extract(
            search_query=query,
            video_limit=limit,
            max_comments_per_video=max_comments,
            parallel_workers=workers,
            lang=lang
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem iptal edildi.")
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")


# ============= ANA PROGRAM =============
if __name__ == '__main__':
    # PyQt6 uygulaması (Selenium için gerekli)
    from PyQt6.QtCore import QCoreApplication
    
    app = QCoreApplication(sys.argv)
    
    # İnteraktif modu başlat
    interactive_mode()
    
    sys.exit(0)