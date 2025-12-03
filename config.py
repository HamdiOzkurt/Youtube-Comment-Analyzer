"""
Yapılandırma Dosyası
Tüm ayarları buradan yönet
"""

# ============= SELENIUM AYARLARI =============
SELENIUM_CONFIG = {
    # ChromeDriver ayarları
    'headless': True,  # Tarayıcı görünür mü?
    'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    
    # Bekleme süreleri (saniye)
    'page_load_timeout': 20,
    'element_wait_timeout': 10,
    'scroll_pause_time': 1.5,
    
    # Kaydırma ayarları
    'scroll_distance': 3000,  # Her kaydırmada kaç piksel
    'max_idle_scroll_attempts': 7,  # Boş kaydırma sayısı limiti
}


# ============= YT-DLP AYARLARI =============
YTDLP_CONFIG = {
    # İndirme ayarları
    'skip_download': True,  # Video indirme (False yapma!)
    'getcomments': True,
    'quiet': True,
    'no_warnings': True,
    
    # Hata yönetimi
    'ignoreerrors': True,  # Hatalı videoları atla
    'no_check_certificate': True,
}


# ============= İŞLEME AYARLARI =============
PROCESSING_CONFIG = {
    # Paralel işlem
    'default_parallel_workers': 5,  # Aynı anda kaç video işlensin
    'max_parallel_workers': 10,  # Maksimum paralel işlem
    
    # Limit ayarları
    'default_video_limit': 10,
    'default_comment_limit_per_video': 100,  # None = hepsi
    
    # Timeout ayarları
    'video_fetch_timeout': 300,  # Video başına max süre (saniye)
}


# ============= VERİ KAYDETME AYARLARI =============
DATA_CONFIG = {
    # Dosya yolları
    'output_directory': 'output',
    'log_directory': 'logs',
    
    # Dosya formatları
    'save_json': True,
    'save_txt': True,
    'save_csv': True,
    'save_statistics': True,
    
    # Encoding
    'encoding': 'utf-8',
    'csv_encoding': 'utf-8-sig',  # Excel için BOM ile
}


# ============= FİLTRELEME AYARLARI =============
FILTER_CONFIG = {
    # Anahtar kelime ayarları
    'case_sensitive': False,  # Büyük/küçük harf duyarlı mı?
    'match_whole_word': False,  # Tam kelime eşleşmesi mi?
    
    # Spam filtreleme (gelecekte eklenebilir)
    'min_comment_length': 5,  # Minimum karakter
    'max_comment_length': 5000,  # Maximum karakter
    'filter_spam': False,  # Spam kelimelerini filtrele
    'spam_keywords': ['spam', 'click here', 'visit my channel'],
}


# ============= LOGLAMA AYARLARI =============
LOGGING_CONFIG = {
    'enable_console_logging': True,
    'enable_file_logging': True,
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
}


# ============= KULLANICI AYARLARI =============
USER_CONFIG = {
    # Kullanıcı tercihleri
    'show_progress': True,
    'show_statistics': True,
    'auto_save': True,
    
    # Bildirimler
    'notify_on_completion': True,
    'notify_on_error': True,
}


# ============= GELİŞMİŞ AYARLAR =============
ADVANCED_CONFIG = {
    # Rate limiting (hız sınırlama)
    'enable_rate_limiting': False,
    'requests_per_minute': 30,
    
    # Retry mekanizması
    'max_retries': 3,
    'retry_delay': 5,  # saniye
    
    # Cache
    'enable_cache': False,
    'cache_duration': 3600,  # saniye
}


# ============= YARDIMCI FONKSİYONLAR =============

def get_config(section=None):
    """Belirli bir bölümün veya tüm ayarların dict'ini döner"""
    if section:
        return globals().get(f"{section.upper()}_CONFIG", {})
    
    # Tüm config'leri birleştir
    all_configs = {}
    for key, value in globals().items():
        if key.endswith('_CONFIG'):
            all_configs[key.replace('_CONFIG', '').lower()] = value
    
    return all_configs


def print_config():
    """Tüm ayarları yazdır"""
    configs = get_config()
    
    print("\n" + "="*60)
    print("⚙️  MEVCUT YAPILANDIRMA")
    print("="*60 + "\n")
    
    for section, settings in configs.items():
        print(f"📁 {section.upper()}:")
        for key, value in settings.items():
            print(f"   • {key}: {value}")
        print()


def update_config(section, key, value):
    """Ayarları çalışma zamanında güncelle"""
    config_name = f"{section.upper()}_CONFIG"
    if config_name in globals():
        globals()[config_name][key] = value
        return True
    return False


# ============= TEST =============
if __name__ == '__main__':
    print_config()
    
    # Örnek güncelleme
    update_config('processing', 'default_parallel_workers', 8)
    print("\n✅ default_parallel_workers 8'e güncellendi\n")
    
    print(f"Yeni değer: {PROCESSING_CONFIG['default_parallel_workers']}")