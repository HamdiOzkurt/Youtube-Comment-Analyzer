"""
WordCloud Generator - Kelime Bulutu Oluşturucu
Yorumlardan görsel kelime bulutu üretir
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict, Optional, List
import numpy as np


# Türkçe stop words
TURKISH_STOP_WORDS = {
    've', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'o', 'ne', 'var',
    'ben', 'sen', 'biz', 'siz', 'onlar', 'şu', 'her', 'daha', 'çok',
    'en', 'gibi', 'kadar', 'sonra', 'önce', 'ama', 'fakat', 'ancak',
    'ki', 'mi', 'mı', 'mu', 'mü', 'ya', 'yani', 'hem', 'veya', 'ise',
    'bile', 'sadece', 'artık', 'hep', 'hiç', 'olan', 'olarak', 'evet',
    'hayır', 'tamam', 'peki', 'işte', 'böyle', 'şöyle', 'öyle', 'olan'
}


def generate_wordcloud(
    text: str = None,
    word_frequencies: Dict[str, int] = None,
    width: int = 800,
    height: int = 400,
    background_color: str = 'rgba(0,0,0,0)',
    colormap: str = 'viridis',  # Light Mode için daha zengin renkler
    max_words: int = 100,
    min_font_size: int = 10,
    max_font_size: int = 100
) -> Optional[str]:
    """
    Kelime bulutu oluştur ve base64 olarak döndür
    
    Args:
        text: Metin (word_frequencies verilmediyse)
        word_frequencies: Kelime frekansları dict'i
        width: Genişlik
        height: Yükseklik
        background_color: Arkaplan rengi
        colormap: Matplotlib colormap
        max_words: Maksimum kelime sayısı
        
    Returns:
        Base64 encoded PNG image
    """
    try:
        if word_frequencies:
            # Frekanslardan oluştur
            filtered_freq = {
                k: v for k, v in word_frequencies.items() 
                if k.lower() not in TURKISH_STOP_WORDS and len(k) > 2
            }
            
            if not filtered_freq:
                return None
            
            wc = WordCloud(
                width=width,
                height=height,
                background_color=None,
                mode='RGBA',
                colormap=colormap,
                max_words=max_words,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                prefer_horizontal=0.9,
                relative_scaling=0.5
            ).generate_from_frequencies(filtered_freq)
            
        elif text:
            # Metinden oluştur
            wc = WordCloud(
                width=width,
                height=height,
                background_color=None,
                mode='RGBA',
                colormap=colormap,
                max_words=max_words,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                stopwords=TURKISH_STOP_WORDS,
                prefer_horizontal=0.9,
                relative_scaling=0.5
            ).generate(text)
        else:
            return None
        
        # PNG olarak kaydet
        img_buffer = BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Base64'e çevir
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"WordCloud hatası: {e}")
        return None


def generate_sentiment_wordcloud(
    positive_words: Dict[str, int],
    negative_words: Dict[str, int],
    width: int = 800,
    height: int = 400
) -> Dict[str, Optional[str]]:
    """
    Pozitif ve negatif kelimeler için ayrı word cloud'lar oluştur
    
    Returns:
        {'positive': base64_image, 'negative': base64_image}
    """
    result = {'positive': None, 'negative': None}
    
    if positive_words:
        result['positive'] = generate_wordcloud(
            word_frequencies=positive_words,
            width=width,
            height=height,
            colormap='Greens'
        )
    
    if negative_words:
        result['negative'] = generate_wordcloud(
            word_frequencies=negative_words,
            width=width,
            height=height,
            colormap='Reds'
        )
    
    return result


def get_word_frequencies_from_texts(texts: List[str], top_n: int = 100) -> Dict[str, int]:
    """Metin listesinden kelime frekansları çıkar"""
    import re
    
    word_count = {}
    
    for text in texts:
        if not text:
            continue
        
        # Kelimeleri çıkar
        words = re.findall(r'\b[a-zçğıöşüA-ZÇĞIİÖŞÜ]{3,}\b', text.lower())
        
        for word in words:
            if word not in TURKISH_STOP_WORDS:
                word_count[word] = word_count.get(word, 0) + 1
    
    # En sık kullanılanları döndür
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_words[:top_n])


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("☁️ WORDCLOUD GENERATOR TEST")
    print("=" * 60)
    
    test_texts = [
        "Bu şarkı çok güzel olmuş harika efsane",
        "Müzik kalitesi süper vokal mükemmel",
        "Şarkı sözleri anlamlı duygusal etkileyici",
        "Sanatçı yetenekli harika performans",
        "Video klip kaliteli profesyonel çekim"
    ]
    
    # Frekansları çıkar
    frequencies = get_word_frequencies_from_texts(test_texts)
    print(f"\n📊 Kelime Frekansları: {frequencies}")
    
    # WordCloud oluştur
    wc_base64 = generate_wordcloud(word_frequencies=frequencies, width=400, height=200)
    
    if wc_base64:
        print(f"\n✅ WordCloud oluşturuldu! (Base64 length: {len(wc_base64)})")
    else:
        print("\n❌ WordCloud oluşturulamadı!")
