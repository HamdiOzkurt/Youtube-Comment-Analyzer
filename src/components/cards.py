"""
Card Components - İstatistik Kartları
Dashboard için modern stat kartları
"""

import streamlit as st
from typing import Optional, Union


def stat_card(
    title: str,
    value: Union[str, int, float],
    icon: str = "📊",
    delta: Optional[float] = None,
    delta_suffix: str = "%",
    color: str = "#667eea"
) -> None:
    """
    Modern istatistik kartı
    
    Args:
        title: Kart başlığı
        value: Ana değer
        icon: Emoji ikonu
        delta: Değişim yüzdesi (opsiyonel)
        delta_suffix: Delta soneki
        color: Ana renk (hex)
    """
    delta_html = ""
    if delta is not None:
        delta_color = "#00D26A" if delta >= 0 else "#FF6B6B"
        delta_icon = "↗" if delta >= 0 else "↘"
        delta_html = f'<div style="color: {delta_color}; font-size: 14px; margin-top: 5px;">{delta_icon} {abs(delta):.1f}{delta_suffix}</div>'
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
            border: 1px solid {color}44;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 40px {color}33';"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
            <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 28px; font-weight: 700; color: white; margin-bottom: 4px;">{value}</div>
            <div style="font-size: 14px; color: rgba(255,255,255,0.7); text-transform: uppercase; letter-spacing: 1px;">{title}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def info_card(
    title: str,
    content: str,
    icon: str = "ℹ️",
    type: str = "info"
) -> None:
    """
    Bilgi kartı (alert tarzı)
    
    Args:
        title: Başlık
        content: İçerik
        icon: Emoji
        type: 'info', 'success', 'warning', 'error'
    """
    colors = {
        'info': ('#3498db', '#3498db22'),
        'success': ('#00D26A', '#00D26A22'),
        'warning': ('#f39c12', '#f39c1222'),
        'error': ('#FF6B6B', '#FF6B6B22')
    }
    
    border_color, bg_color = colors.get(type, colors['info'])
    
    st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 4px solid {border_color};
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
        ">
            <div style="font-size: 16px; font-weight: 600; color: white; margin-bottom: 8px;">
                {icon} {title}
            </div>
            <div style="font-size: 14px; color: rgba(255,255,255,0.8); line-height: 1.6;">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)


def video_card(
    title: str,
    channel: str,
    views: int,
    comments: int,
    sentiment_score: float,
    thumbnail_url: Optional[str] = None
) -> None:
    """
    Video bilgi kartı
    """
    # Sentiment rengi
    if sentiment_score >= 0.3:
        sentiment_color = "#00D26A"
        sentiment_text = "Pozitif"
    elif sentiment_score >= -0.3:
        sentiment_color = "#FFA500"
        sentiment_text = "Nötr"
    else:
        sentiment_color = "#FF6B6B"
        sentiment_text = "Negatif"
    
    thumbnail_html = ""
    if thumbnail_url:
        thumbnail_html = f'<img src="{thumbnail_url}" style="width: 100%; border-radius: 12px; margin-bottom: 12px;">'
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
        ">
            {thumbnail_html}
            <h3 style="color: white; font-size: 18px; margin-bottom: 8px; line-height: 1.4;">{title[:60]}{'...' if len(title) > 60 else ''}</h3>
            <p style="color: rgba(255,255,255,0.6); font-size: 14px; margin-bottom: 16px;">{channel}</p>
            
            <div style="display: flex; justify-content: space-between; gap: 10px;">
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 18px; font-weight: 600; color: white;">{views:,}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.5);">Görüntülenme</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 18px; font-weight: 600; color: white;">{comments:,}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.5);">Yorum</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="font-size: 18px; font-weight: 600; color: {sentiment_color};">{sentiment_text}</div>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.5);">Duygu</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def comment_card(
    author: str,
    text: str,
    likes: int,
    sentiment: str,
    confidence: float
) -> None:
    """
    Yorum kartı
    """
    sentiment_colors = {
        'positive': '#00D26A',
        'negative': '#FF6B6B',
        'neutral': '#95A5A6'
    }
    color = sentiment_colors.get(sentiment, '#95A5A6')
    
    st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.05);
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: white;">👤 {author}</span>
                <span style="color: rgba(255,255,255,0.5); font-size: 12px;">👍 {likes}</span>
            </div>
            <p style="color: rgba(255,255,255,0.8); font-size: 14px; line-height: 1.6; margin: 0;">{text[:300]}{'...' if len(text) > 300 else ''}</p>
            <div style="margin-top: 8px; display: flex; gap: 10px;">
                <span style="
                    background: {color}22;
                    color: {color};
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                ">{sentiment.upper()}</span>
                <span style="color: rgba(255,255,255,0.5); font-size: 12px;">Güven: {confidence:.0%}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def progress_card(
    title: str,
    current: int,
    total: int,
    icon: str = "📊"
) -> None:
    """
    İlerleme kartı
    """
    percentage = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; color: white;">{icon} {title}</span>
                <span style="color: rgba(255,255,255,0.7);">{current:,} / {total:,}</span>
            </div>
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                height: 10px;
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                    height: 100%;
                    width: {percentage}%;
                    border-radius: 10px;
                    transition: width 0.5s ease;
                "></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
