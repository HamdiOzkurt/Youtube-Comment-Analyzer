"""
NLP Processor - Metin Temizleme ve Normalizasyon Modülü
YouTube yorumları için Türkçe NLP işleme pipeline'ı
data_preprocessing.ipynb örneğine göre tasarlandı
"""

import re
import pandas as pd
from typing import List, Optional, Union, Set
from timeit import default_timer as timer
from datetime import timedelta

# NLTK stopwords için lazy loading
_stop_kelimeler = None

def get_turkish_stopwords() -> Set[str]:
    """NLTK'dan Türkçe stop words yükle (lazy loading)"""
    global _stop_kelimeler
    
    if _stop_kelimeler is None:
        try:
            from nltk.corpus import stopwords
            _stop_kelimeler = set(stopwords.words('turkish'))
        except LookupError:
            import nltk
            print("📥 NLTK stopwords indiriliyor...")
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _stop_kelimeler = set(stopwords.words('turkish'))
        except ImportError:
            print("⚠️ NLTK yüklü değil, varsayılan stop words kullanılıyor")
            # Fallback stop words
            _stop_kelimeler = {
                've', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'o', 'ne', 'var',
                'ben', 'sen', 'biz', 'siz', 'onlar', 'şu', 'her', 'daha', 'çok',
                'en', 'gibi', 'kadar', 'sonra', 'önce', 'ama', 'fakat', 'ancak',
                'ki', 'mi', 'mı', 'mu', 'mü', 'ya', 'yani', 'hem', 'veya', 'ise',
                'bile', 'sadece', 'artık', 'hep', 'hiç', 'olan', 'olarak'
            }
    
    return _stop_kelimeler


def preprocessing(
    series: pd.Series,
    remove_hashtag: bool = False,
    remove_mentions: bool = False,
    remove_links: bool = False,
    remove_numbers: bool = False,
    remove_short_text: bool = False,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_stopwords: bool = False,
    remove_rare_words: bool = False,
    remove_non_latin: bool = False,
    rare_limit: int = 5,
    custom_stopwords: Optional[Set[str]] = None,
    min_text_length: int = 3,
    verbose: bool = True
) -> pd.Series:
    """
    Pandas Series üzerinde metin ön işleme uygular.
    
    Parameters
    ----------
    series : pandas.Series
        İşlenecek metin serisi.
    remove_hashtag : bool, default=False
        True ise hashtag'leri (#) kaldırır.
    remove_mentions : bool, default=False
        True ise mention'ları (@) kaldırır.
    remove_links : bool, default=False
        True ise URL'leri kaldırır.
    remove_numbers : bool, default=False
        True ise sayıları kaldırır.
    remove_short_text : bool, default=False
        True ise kısa kelimeleri (min_text_length'den kısa) kaldırır.
    lowercase : bool, default=False
        True ise tüm metni küçük harfe çevirir.
    remove_punctuation : bool, default=False
        True ise noktalama işaretlerini kaldırır.
    remove_stopwords : bool, default=False
        True ise Türkçe stop words'leri kaldırır.
    remove_rare_words : bool, default=False
        True ise nadir kelimeleri (rare_limit veya daha az geçen) kaldırır.
    remove_non_latin : bool, default=False
        True ise Arapça, Kiril, emoji vb. karakterleri kaldırır.
    rare_limit : int, default=5
        Nadir kelime eşiği.
    custom_stopwords : set veya None, default=None
        Varsayılan yerine kullanılacak özel stop words.
    min_text_length : int, default=3
        remove_short_text=True iken minimum kelime uzunluğu.
    verbose : bool, default=True
        True ise işlem süreleri yazdırılır.
        
    Returns
    -------
    pandas.Series
        İşlenmiş metin serisi.
    """
    # Orijinali değiştirmemek için kopyala
    series = series.copy()
    
    def log(msg):
        if verbose:
            print(msg)
    
    if lowercase:
        log("🔄 Küçük harfe çevriliyor...")
        start = timer()
        series = series.str.lower()
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_hashtag:
        log("🔄 Hashtag'ler kaldırılıyor...")
        start = timer()
        series = series.str.replace(r'#\w+', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_mentions:
        log("🔄 Mention'lar kaldırılıyor...")
        start = timer()
        series = series.str.replace(r'@\w+', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_links:
        log("🔄 URL'ler kaldırılıyor...")
        start = timer()
        series = series.str.replace(r'http\S+|www\.\S+', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_numbers:
        log("🔄 Sayılar kaldırılıyor...")
        start = timer()
        series = series.str.replace(r'\d+', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_punctuation:
        log("🔄 Noktalama işaretleri kaldırılıyor...")
        start = timer()
        series = series.str.replace(r'[^\w\s]', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_short_text:
        log(f"🔄 Kısa kelimeler kaldırılıyor (uzunluk < {min_text_length})...")
        start = timer()
        pattern = r'\b\w{1,' + str(min_text_length - 1) + r'}\b'
        series = series.str.replace(pattern, '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_non_latin:
        log("🔄 Latin olmayan karakterler kaldırılıyor (Arapça, Kiril, Emoji)...")
        start = timer()
        # a-z, 0-9 ve Türkçe harfler (çğıöşü) HARİÇ her şeyi sil
        series = series.str.replace(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]', '', regex=True)
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_stopwords:
        log("🔄 Stop words kaldırılıyor...")
        start = timer()
        stopwords_to_use = custom_stopwords if custom_stopwords else get_turkish_stopwords()
        series = series.apply(lambda x: ' '.join([
            word for word in str(x).split() 
            if word.lower() not in stopwords_to_use
        ]))
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    if remove_rare_words:
        log("🔄 Nadir kelimeler kaldırılıyor...")
        start = timer()
        all_words = ' '.join(series.astype(str)).split()
        word_counts = pd.Series(all_words).value_counts()
        
        log(f"   📊 Toplam benzersiz kelime: {len(word_counts)}")
        rare_words = word_counts[word_counts <= rare_limit]
        log(f"   📊 Nadir kelimeler ({rare_limit} veya daha az): {len(rare_words)} "
            f"({len(rare_words)/len(word_counts)*100:.2f}%)")
        
        rare_words_set = set(rare_words.index)
        series = series.apply(lambda x: ' '.join([
            word for word in str(x).split() 
            if word not in rare_words_set
        ]))
        log(f"✅ Tamamlandı: {timedelta(seconds=timer() - start)}")
    
    # Fazla boşlukları temizle
    series = series.str.strip().str.replace(r'\s+', ' ', regex=True)
    
    return series


def tr_en_char_translate(series: pd.Series) -> pd.Series:
    """
    Türkçe karakterleri İngilizce karşılıklarına çevirir.
    
    Parameters
    ----------
    series : pandas.Series
        Çevrilecek metin serisi.
        
    Returns
    -------
    pandas.Series
        Çevrilmiş metin serisi.
    """
    series = series.str.replace('ı', 'i')
    series = series.str.replace('ü', 'u')
    series = series.str.replace('ö', 'o')
    series = series.str.replace('ğ', 'g')
    series = series.str.replace('ş', 's')
    series = series.str.replace('ç', 'c')
    series = series.str.replace('İ', 'I')
    series = series.str.replace('Ü', 'U')
    series = series.str.replace('Ö', 'O')
    series = series.str.replace('Ğ', 'G')
    series = series.str.replace('Ş', 'S')
    series = series.str.replace('Ç', 'C')
    return series


class NLPProcessor:
    """Türkçe metin temizleme ve normalizasyon sınıfı - Basit API"""
    
    def __init__(self,
                 remove_hashtag: bool = True,
                 remove_mentions: bool = True,
                 remove_links: bool = True,
                 remove_numbers: bool = False,
                 remove_non_latin: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_stopwords: bool = False,
                 remove_short_text: bool = False,
                 min_text_length: int = 3,
                 custom_stopwords: Optional[Set[str]] = None):
        """
        Args:
            remove_hashtag: Hashtag'leri kaldır
            remove_mentions: Mention'ları kaldır
            remove_links: URL'leri kaldır
            remove_numbers: Sayıları kaldır
            remove_non_latin: Emoji, Arapça vb. kaldır
            lowercase: Küçük harfe çevir
            remove_punctuation: Noktalama işaretlerini kaldır
            remove_stopwords: Türkçe stop words'leri kaldır
            remove_short_text: Kısa kelimeleri kaldır
            min_text_length: Minimum kelime uzunluğu
            custom_stopwords: Özel stop words listesi
        """
        self.remove_hashtag = remove_hashtag
        self.remove_mentions = remove_mentions
        self.remove_links = remove_links
        self.remove_numbers = remove_numbers
        self.remove_non_latin = remove_non_latin
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_short_text = remove_short_text
        self.min_text_length = min_text_length
        self.custom_stopwords = custom_stopwords
        
        # Stop words'ü önceden yükle
        if remove_stopwords:
            self.stopwords = custom_stopwords if custom_stopwords else get_turkish_stopwords()
        else:
            self.stopwords = set()
    
    def clean_text(self, text: str) -> str:
        """Tek bir metni temizle"""
        if not text or not isinstance(text, str):
            return ""
        
        # Küçük harfe çevir
        if self.lowercase:
            text = text.lower()
        
        # Hashtag'leri kaldır
        if self.remove_hashtag:
            text = re.sub(r'#\w+', '', text)
        
        # Mention'ları kaldır
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # URL'leri kaldır
        if self.remove_links:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Sayıları kaldır
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Latin olmayan karakterleri kaldır (emoji dahil)
        if self.remove_non_latin:
            text = re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s]', '', text)
        
        # Noktalama işaretlerini kaldır
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Kısa kelimeleri kaldır
        if self.remove_short_text:
            pattern = r'\b\w{1,' + str(self.min_text_length - 1) + r'}\b'
            text = re.sub(pattern, '', text)
        
        # Stop words kaldır
        if self.remove_stopwords and self.stopwords:
            words = text.split()
            text = ' '.join([w for w in words if w.lower() not in self.stopwords])
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process(self, text: str) -> str:
        """clean_text alias'ı"""
        return self.clean_text(text)
    
    def process_batch(self, texts: List[str], verbose: bool = False) -> List[str]:
        """Birden fazla metni işle"""
        if verbose:
            print(f"🔄 {len(texts)} metin işleniyor...")
        return [self.clean_text(text) for text in texts]
    
    def process_series(self, series: pd.Series, verbose: bool = True) -> pd.Series:
        """Pandas Series üzerinde işlem yap"""
        return preprocessing(
            series,
            remove_hashtag=self.remove_hashtag,
            remove_mentions=self.remove_mentions,
            remove_links=self.remove_links,
            remove_numbers=self.remove_numbers,
            remove_short_text=self.remove_short_text,
            lowercase=self.lowercase,
            remove_punctuation=self.remove_punctuation,
            remove_stopwords=self.remove_stopwords,
            remove_non_latin=self.remove_non_latin,
            custom_stopwords=self.custom_stopwords,
            min_text_length=self.min_text_length,
            verbose=verbose
        )
    
    def extract_questions(self, text: str) -> List[str]:
        """Metinden soru cümlelerini çıkar"""
        if not text:
            return []
        
        questions = []
        # Soru işareti ile biten cümleleri bul
        sentences = re.split(r'[.!]', text)
        for sentence in sentences:
            if '?' in sentence:
                parts = sentence.split('?')
                for part in parts[:-1]:
                    q = part.strip() + '?'
                    if len(q) > 5:
                        questions.append(q)
        return questions
    
    def extract_requests(self, text: str) -> List[str]:
        """Metinden talep/istek cümlelerini çıkar"""
        if not text:
            return []
        
        request_patterns = [
            r'lütfen\s+.+',
            r'.+\s+yapabilir\s*misiniz',
            r'.+\s+yapar\s*mısınız',
            r'.+\s+istiyorum',
            r'.+\s+bekliyorum',
            r'devamını\s+.+',
        ]
        
        requests = []
        text_lower = text.lower()
        
        for pattern in request_patterns:
            matches = re.findall(pattern, text_lower)
            requests.extend(matches)
        
        return list(set(requests))
    
    def get_word_frequencies(self, texts: List[str], top_n: int = 50) -> dict:
        """Kelime frekanslarını hesapla"""
        word_count = {}
        
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            
            for word in words:
                if len(word) >= self.min_text_length:
                    word_count[word] = word_count.get(word, 0) + 1
        
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:top_n])


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("🧹 NLP PROCESSOR TEST")
    print("=" * 60)
    
    # Test metinleri
    test_texts = [
        "Bu video çok güzel olmuş! 🎉🔥 https://youtube.com/watch?v=123",
        "@kanal çok beğendim, devamını bekliyorum lütfen yapın",
        "Şarkı süper ama sanatçının sesi biraz yorgun mu?",
        "Bu tarz videolar yapabilir misiniz? Çok istiyorum! 😍",
        "مرحبا 你好 Merhaba dünya! #test @user"
    ]
    
    # Pandas Series oluştur
    series = pd.Series(test_texts)
    
    print("\n📋 PANDAS SERIES PREPROCESSING:\n")
    
    # Preprocessing uygula
    processed = preprocessing(
        series,
        lowercase=True,
        remove_mentions=True,
        remove_links=True,
        remove_non_latin=True,
        verbose=True
    )
    
    print("\n📊 SONUÇLAR:")
    for i, (orig, proc) in enumerate(zip(test_texts, processed)):
        print(f"\n{i+1}. Orijinal: {orig}")
        print(f"   Temiz:    {proc}")
    
    print("\n" + "=" * 60)
    print("🔧 NLPPROCESSOR SINIFI TEST:")
    print("=" * 60)
    
    processor = NLPProcessor(
        remove_non_latin=True,
        remove_links=True,
        remove_mentions=True,
        lowercase=True,
        remove_stopwords=True
    )
    
    for text in test_texts[:3]:
        print(f"\n📝 Orijinal: {text}")
        print(f"✨ Temiz: {processor.clean_text(text)}")
        
        questions = processor.extract_questions(text)
        if questions:
            print(f"❓ Sorular: {questions}")
