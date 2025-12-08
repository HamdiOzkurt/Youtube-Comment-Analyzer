"""
Progress Bar Component - Durum Çubuğu Bileşeni
Tüm sayfalarda kullanılabilir ilerleme göstergesi
"""

import streamlit as st
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ProgressTracker:
    """İlerleme takibi için yardımcı sınıf"""
    total_steps: int
    current_step: int = 0
    current_phase: str = ""
    sub_progress: float = 0.0
    
    def get_overall_progress(self) -> float:
        """Toplam ilerleme yüzdesini hesapla (0.0 - 1.0)"""
        if self.total_steps == 0:
            return 0.0
        base = self.current_step / self.total_steps
        sub_contribution = self.sub_progress / self.total_steps
        return min(base + sub_contribution, 1.0)


class ProgressBar:
    """
    Streamlit progress bar wrapper
    Tüm sayfalarda tutarlı kullanım için
    """
    
    def __init__(self, container=None):
        """
        Args:
            container: st.empty() veya benzeri bir Streamlit container
        """
        self.container = container or st.empty()
        self.progress_bar = None
        self.status_text = None
        self._setup_ui()
    
    def _setup_ui(self):
        """UI elementlerini oluştur"""
        with self.container.container():
            self.status_text = st.empty()
            self.progress_bar = st.progress(0)
    
    def update(self, progress: float, status: str = ""):
        """
        İlerlemeyi güncelle
        
        Args:
            progress: 0.0 - 1.0 arası ilerleme değeri
            status: Gösterilecek durum mesajı
        """
        # Clamp progress between 0 and 1
        progress = max(0.0, min(1.0, progress))
        
        if self.status_text:
            percentage = int(progress * 100)
            self.status_text.markdown(f"**{status}** ({percentage}%)")
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
    
    def complete(self, message: str = "Tamamlandı!"):
        """İşlem tamamlandığında çağır"""
        self.update(1.0, f"✅ {message}")
    
    def error(self, message: str = "Hata oluştu!"):
        """Hata durumunda çağır"""
        if self.status_text:
            self.status_text.markdown(f"❌ **{message}**")
    
    def clear(self):
        """Progress bar'ı temizle"""
        self.container.empty()


def create_battle_progress_callback(progress_bar: ProgressBar, total_categories: int, comments_per_category: int):
    """
    Battle Mode için progress callback oluştur
    
    Args:
        progress_bar: ProgressBar instance
        total_categories: Toplam kategori sayısı
        comments_per_category: Her kategoride işlenecek yorum sayısı (video başına)
    
    Returns:
        Callback fonksiyonu
    """
    # Batch size = 3, her video için batch sayısı
    batch_size = 3
    batches_per_video = (comments_per_category + batch_size - 1) // batch_size
    
    # Toplam batch: kategori * 2 video * batch sayısı
    total_batches = total_categories * 2 * batches_per_video
    
    state = {
        'processed_batches': 0,  # Sadece artan sayaç
        'current_category': '',
        'current_video': 1,
        'last_progress': 0.0
    }
    
    def callback(message: str):
        # Kategori başlangıcı
        if "Kategori:" in message:
            state['current_category'] = message.split(":")[-1].strip()[:15]
            state['current_video'] = 1
        
        # V1 batch işleniyor
        elif "V1 [" in message and "/" in message:
            state['current_video'] = 1
            try:
                parts = message.split(":")[-1].strip()
                current, total = parts.split("/")
                batch_num = (int(current) + batch_size - 1) // batch_size
                # Sadece artıyorsa güncelle
                new_progress = state['processed_batches'] + batch_num
                if new_progress > state['last_progress'] * total_batches:
                    state['processed_batches'] = max(state['processed_batches'], 
                        (state['processed_batches'] // (2 * batches_per_video)) * (2 * batches_per_video) + batch_num)
            except:
                pass
        
        # V2 batch işleniyor
        elif "V2 [" in message and "/" in message:
            state['current_video'] = 2
            try:
                parts = message.split(":")[-1].strip()
                current, total = parts.split("/")
                batch_num = (int(current) + batch_size - 1) // batch_size
                # V2 için offset ekle
                base_batches = (state['processed_batches'] // (2 * batches_per_video)) * (2 * batches_per_video)
                state['processed_batches'] = max(state['processed_batches'], 
                    base_batches + batches_per_video + batch_num)
            except:
                pass
        
        # V1 tamamlandı
        elif "tamamlandı" in message and "V1" in message:
            base_batches = (state['processed_batches'] // (2 * batches_per_video)) * (2 * batches_per_video)
            state['processed_batches'] = base_batches + batches_per_video
            state['current_video'] = 2
        
        # V2 tamamlandı - sonraki kategoriye geç
        elif "tamamlandı" in message and "V2" in message:
            base_batches = (state['processed_batches'] // (2 * batches_per_video)) * (2 * batches_per_video)
            state['processed_batches'] = base_batches + 2 * batches_per_video
            state['current_video'] = 1
        
        # Progress hesapla (sadece artabilir)
        progress = state['processed_batches'] / total_batches if total_batches > 0 else 0
        progress = max(state['last_progress'], min(1.0, progress))
        state['last_progress'] = progress
        
        # Durum mesajı
        cat_name = state['current_category'] or "..."
        video_label = "Video 1" if state['current_video'] == 1 else "Video 2"
        status = f"{cat_name} | {video_label}"
        
        progress_bar.update(progress, status)
    
    return callback


def create_analysis_progress_callback(progress_bar: ProgressBar, total_videos: int = 1, comments_per_video: int = 100):
    """
    Video analizi için progress callback oluştur
    
    Args:
        progress_bar: ProgressBar instance
        total_videos: Toplam video sayısı
        comments_per_video: Video başına yorum sayısı
    """
    state = {
        'phase': 'search',  # search, fetch, analyze
        'current_video': 0,
        'total_videos': total_videos
    }
    
    def callback(message: str):
        # Phase detection
        if "Video" in message and "çekiliyor" in message:
            state['phase'] = 'fetch'
        elif "analiz" in message.lower():
            state['phase'] = 'analyze'
        elif "URL" in message or "arama" in message.lower():
            state['phase'] = 'search'
        
        # Video sayısı algılama
        if "/" in message:
            try:
                parts = message.split("/")
                for i, part in enumerate(parts):
                    if part.strip().isdigit() and i > 0:
                        state['current_video'] = int(parts[i-1].split()[-1])
                        break
            except:
                pass
        
        # İlerleme hesapla
        if state['phase'] == 'search':
            progress = 0.1
            status = "🔍 Video aranıyor..."
        elif state['phase'] == 'fetch':
            base = 0.1
            fetch_progress = (state['current_video'] / max(state['total_videos'], 1)) * 0.6
            progress = base + fetch_progress
            status = f"📥 Yorumlar çekiliyor ({state['current_video']}/{state['total_videos']})"
        else:
            progress = 0.85
            status = "🧠 Analiz yapılıyor..."
        
        progress_bar.update(progress, status)
    
    return callback
