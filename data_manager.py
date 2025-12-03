import os
import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path

class DataManager:
    """Veri yönetimi ve kaydetme işlemleri"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def filter_comments_by_keyword(self, results, keywords, case_sensitive=False):
        """
        Yorumları anahtar kelimelere göre filtreler
        
        Args:
            results: Yorum sonuçları listesi
            keywords: Aranacak kelimeler listesi
            case_sensitive: Büyük/küçük harf duyarlılığı
            
        Returns:
            Filtrelenmiş sonuç listesi
        """
        filtered_results = []
        
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]
            
        for video in results:
            filtered_video = video.copy()
            filtered_comments = []
            
            for comment in video['yorumlar']:
                text = comment.get('metin', '')
                if not case_sensitive:
                    text = text.lower()
                    
                # Herhangi bir anahtar kelime geçiyor mu?
                if any(keyword in text for keyword in keywords):
                    filtered_comments.append(comment)
            
            if filtered_comments:
                filtered_video['yorumlar'] = filtered_comments
                filtered_results.append(filtered_video)
                
        return filtered_results

    def save_all_formats(self, results, stats, prefix="output"):
        """
        Verileri tüm formatlarda kaydeder (JSON, CSV, Excel, TXT)
        
        Args:
            results: Yorum verileri
            stats: İstatistik verileri
            prefix: Dosya adı öneki
            
        Returns:
            dict: Kaydedilen dosya yolları
        """
        # Klasörün var olduğundan emin ol
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
        saved_files = {}
        
        # 1. JSON Kaydet
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'meta': {
                    'tarih': datetime.now().isoformat(),
                    'istatistikler': stats
                },
                'veriler': results
            }, f, ensure_ascii=False, indent=4)
        saved_files['json'] = json_path
        
        # 2. CSV/Excel için düzleştirme
        flat_data = []
        for video in results:
            for comment in video['yorumlar']:
                flat_data.append({
                    'Video Başlığı': video['baslik'],
                    'Video URL': video['url'],
                    'Kanal': video['kanal'],
                    'Yazar': comment['yazar'],
                    'Yorum': comment['metin'],
                    'Beğeni': comment['begeni'],
                    'Tarih': comment['timestamp']
                })
        
        if flat_data:
            df = pd.DataFrame(flat_data)
            
            # CSV Kaydet
            csv_path = self.output_dir / f"{base_name}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            saved_files['csv'] = csv_path
            
            # Excel Kaydet
            excel_path = self.output_dir / f"{base_name}.xlsx"
            df.to_excel(excel_path, index=False)
            saved_files['xlsx'] = excel_path
            
        # 3. TXT Rapor Kaydet
        txt_path = self.output_dir / f"{base_name}_rapor.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"RAPOR - {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
            f.write("="*50 + "\n\n")
            
            if stats:
                f.write("İSTATİSTİKLER\n")
                f.write(f"Toplam Video: {stats.get('toplam_video', 0)}\n")
                f.write(f"Toplam Yorum: {stats.get('toplam_yorum', 0)}\n")
                f.write(f"Toplam Beğeni: {stats.get('toplam_begeni', 0)}\n\n")
            
            f.write("EN POPÜLER YORUMLAR\n")
            f.write("-" * 30 + "\n")
            if stats and 'en_populer_yorumlar' in stats:
                for i, comment in enumerate(stats['en_populer_yorumlar'], 1):
                    f.write(f"{i}. [{comment.get('begeni', 0)} Beğeni] {comment.get('yazar', 'Anonim')}\n")
                    f.write(f"   Video: {comment.get('video_baslik', '')}\n")
                    f.write(f"   Yorum: {comment.get('metin', '')[:100]}...\n\n")
                    
        saved_files['txt'] = txt_path
        
        return saved_files
