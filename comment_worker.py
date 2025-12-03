"""
YouTube Video Yorum Çekici Worker
yt-dlp ile paralel olarak birden fazla videodan yorum çeker
"""

import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time


class CommentWorker:
    def __init__(self, max_workers=5, max_comments_per_video=None):
        """
        Args:
            max_workers: Aynı anda kaç video işlenecek (paralel)
            max_comments_per_video: Her videodan max kaç yorum (None = hepsi)
        """
        self.max_workers = max_workers
        self.max_comments_per_video = max_comments_per_video
        self.results = []
        self.errors = []
        
    def fetch_comments_from_url(self, video_url):
        """Tek bir videodan yorum çeker"""
        ydl_opts = {
            'skip_download': True,
            'getcomments': True,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {
                'youtube': {
                    'max_comments': ['all'] if self.max_comments_per_video is None 
                                   else [str(self.max_comments_per_video)]
                }
            },
        }
        
        try:
            print(f"🔄 İşleniyor: {video_url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                video_data = {
                    'url': video_url,
                    'video_id': info.get('id', ''),
                    'baslik': info.get('title', 'Bilinmiyor'),
                    'kanal': info.get('uploader', 'Bilinmiyor'),
                    'kanal_id': info.get('channel_id', ''),
                    'goruntulenme': info.get('view_count', 0),
                    'begeni': info.get('like_count', 0),
                    'sure': info.get('duration', 0),
                    'yuklenme_tarihi': info.get('upload_date', ''),
                    'yorumlar': []
                }
                
                comments = info.get('comments', [])
                
                if not comments:
                    print(f"⚠️  {video_data['baslik'][:50]} - Yorum yok!")
                    return video_data
                
                # Yorumları işle
                for i, comment in enumerate(comments[:self.max_comments_per_video] 
                                           if self.max_comments_per_video else comments):
                    video_data['yorumlar'].append({
                        'sira': i + 1,
                        'yazar': comment.get('author', 'Anonim'),
                        'yazar_id': comment.get('author_id', ''),
                        'metin': comment.get('text', ''),
                        'begeni': comment.get('like_count', 0),
                        'timestamp': comment.get('timestamp', 0),
                        'cevap_sayisi': comment.get('reply_count', 0),
                    })
                
                print(f"✅ {video_data['baslik'][:50]}... ({video_url}) - {len(video_data['yorumlar'])} yorum çekildi")
                return video_data
                
        except Exception as e:
            error_msg = f"❌ Hata ({video_url}): {str(e)}"
            print(error_msg)
            self.errors.append({
                'url': video_url,
                'hata': str(e),
                'zaman': datetime.now().isoformat()
            })
            return None
    
    def fetch_bulk_comments(self, video_urls):
        """
        Birden fazla videodan paralel olarak yorum çeker
        
        Args:
            video_urls: Video URL listesi
            
        Returns:
            list: Başarılı sonuçlar listesi
        """
        self.results = []
        self.errors = []
        
        print(f"\n🚀 {len(video_urls)} video için yorum çekme başlatıldı...")
        print(f"⚙️  Paralel işlem sayısı: {self.max_workers}")
        print(f"💬 Video başına max yorum: {self.max_comments_per_video or 'HEPSİ'}\n")
        
        start_time = time.time()
        
        # Paralel işleme
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.fetch_comments_from_url, url): url 
                for url in video_urls
            }
            
            completed = 0
            total = len(video_urls)
            
            for future in as_completed(future_to_url):
                completed += 1
                result = future.result()
                
                if result:
                    self.results.append(result)
                
                print(f"📊 İlerleme: {completed}/{total} video tamamlandı")
        
        elapsed = time.time() - start_time
        
        # Özet
        print(f"\n{'='*60}")
        print(f"✅ TAMAMLANDI!")
        print(f"⏱️  Süre: {elapsed:.2f} saniye")
        print(f"📹 Başarılı: {len(self.results)}/{total} video")
        print(f"❌ Hatalı: {len(self.errors)} video")
        
        total_comments = sum(len(v['yorumlar']) for v in self.results)
        print(f"💬 Toplam yorum: {total_comments:,}")
        print(f"{'='*60}\n")
        
        return self.results
    
    def get_statistics(self):
        """Toplanan yorumlar hakkında istatistik döner"""
        if not self.results:
            return None
        
        total_comments = sum(len(v['yorumlar']) for v in self.results)
        total_likes = sum(
            sum(c['begeni'] for c in v['yorumlar']) 
            for v in self.results
        )
        
        # En çok yorumlu video
        most_commented = max(self.results, key=lambda x: len(x['yorumlar']))
        
        # En çok beğenilen yorum
        all_comments = []
        for video in self.results:
            for comment in video['yorumlar']:
                comment['video_baslik'] = video['baslik']
                all_comments.append(comment)
        
        top_comments = sorted(all_comments, key=lambda x: x['begeni'], reverse=True)[:5]
        
        return {
            'toplam_video': len(self.results),
            'toplam_yorum': total_comments,
            'toplam_begeni': total_likes,
            'ortalama_yorum': total_comments / len(self.results) if self.results else 0,
            'en_cok_yorumlu_video': {
                'baslik': most_commented['baslik'],
                'yorum_sayisi': len(most_commented['yorumlar'])
            },
            'en_populer_yorumlar': top_comments
        }


# ============= TEST KODU =============
if __name__ == '__main__':
    # Test URL'leri (kendi linklerinizi girin)
    test_urls = [
        "https://youtu.be/2YJ7zO3UWlA",
        # Daha fazla URL ekleyebilirsiniz
    ]
    
    # Worker başlat
    worker = CommentWorker(
        max_workers=3,  # 3 video aynı anda işlensin
        max_comments_per_video=50  # Her videodan 50 yorum
    )
    
    # Yorumları çek
    results = worker.fetch_bulk_comments(test_urls)
    
    # İstatistikleri göster
    stats = worker.get_statistics()
    if stats:
        print("\n📊 DETAYLI İSTATİSTİKLER:")
        print(f"📹 Toplam Video: {stats['toplam_video']}")
        print(f"💬 Toplam Yorum: {stats['toplam_yorum']:,}")
        print(f"👍 Toplam Beğeni: {stats['toplam_begeni']:,}")
        print(f"📈 Video Başına Ort. Yorum: {stats['ortalama_yorum']:.1f}")
        print(f"\n🏆 En Çok Yorumlu Video:")
        print(f"   {stats['en_cok_yorumlu_video']['baslik'][:60]}")
        print(f"   {stats['en_cok_yorumlu_video']['yorum_sayisi']} yorum")
    
    # Hataları göster
    if worker.errors:
        print("\n❌ HATALAR:")
        for err in worker.errors:
            print(f"   {err['url']}: {err['hata']}")