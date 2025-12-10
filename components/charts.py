import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

# --- PROFESSIONAL PALETTE (LIGHT MODE - ENTERPRISE) ---
COLORS = {
    'primary': '#4A90E2',   # Royal Blue (Video 1 / accent)
    'secondary': '#6C5CE7', # Soft Purple (Video 2 / secondary)
    'video1': '#4A90E2',    # Royal Blue - Video 1
    'video2': '#6C5CE7',    # Soft Purple - Video 2
    'success': '#059669',   # Emerald 600 - Positive (for sentiment ONLY)
    'danger': '#DC2626',    # Red 600 - Negative (for sentiment ONLY)
    'neutral': '#6B7280',   # Gray 500 - Neutral
    'text': '#1F2937',      # Dark Gray (main text)
    'text_muted': '#9CA3AF',# Gray 400 (secondary text)
    'grid': 'rgba(0, 0, 0, 0.04)',  # Very light grid lines
    'bg': 'rgba(255,255,255,0)',    # Transparent white
    'heatmap_scale': ['#F0F9FF', '#BAE6FD', '#7DD3FC', '#38BDF8', '#0EA5E9', '#0284C7', '#0369A1'],  # Monochromatic Blue
}

FONT_FAMILY = "Inter, sans-serif"

# ============ SANITIZATION FUNCTIONS ============
def sanitize_value(value, default=""):
    """Convert None, undefined, or invalid values to safe defaults"""
    if value is None:
        return default
    if isinstance(value, str):
        if value.lower() in ['none', 'undefined', 'null', 'nan']:
            return default
    return value

def sanitize_text(text: str, default: str = "") -> str:
    """Ensure text is a valid string, never None or undefined"""
    if text is None:
        return default
    if not isinstance(text, str):
        return str(text) if text else default
    if text.lower() in ['none', 'undefined', 'null']:
        return default
    return text

def sanitize_title(title: str) -> str:
    """Sanitize title - return empty string if None or undefined"""
    return sanitize_text(title, "")

def sanitize_name(name: str, default: str = "Item") -> str:
    """Sanitize name - return default if None"""
    return sanitize_text(name, default)

def sanitize_number(value, default: float = 0.0) -> float:
    """Ensure numeric value is valid"""
    if value is None:
        return default
    try:
        result = float(value)
        if pd.isna(result) or result != result:  # Check for NaN
            return default
        return result
    except (ValueError, TypeError):
        return default

# ============ LAYOUT UPDATE (SAFE) ============
def _update_layout(fig: go.Figure, title: str = None, height: int = 350):
    """Common layout update for consistency - Light Mode optimized with sanitization"""
    # Sanitize title - if None or empty, don't show title at all
    safe_title = sanitize_title(title)
    
    title_config = None
    if safe_title and len(safe_title.strip()) > 0:
        title_config = dict(text=safe_title, font=dict(size=14, color=COLORS['text']))
    else:
        # Explicitly set title text to empty string to prevent "undefined" artifact
        # Setting it to None sometimes causes JS "undefined" in some Plotly versions
        title_config = dict(text="", font=dict(size=1))
    
    fig.update_layout(
        title=title_config,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family=FONT_FAMILY),
        margin=dict(t=40 if safe_title else 20, b=20, l=40, r=20),
        height=height,
        xaxis=dict(
            showgrid=False, 
            gridcolor=COLORS['grid'],
            linecolor='rgba(0,0,0,0.1)',
            tickfont=dict(color=COLORS['text_muted'])
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor=COLORS['grid'],
            linecolor='rgba(0,0,0,0.1)',
            tickfont=dict(color=COLORS['text_muted']),
            gridwidth=1
        ),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            font=dict(color=COLORS['text'])
        )
    )

def create_sentiment_pie_chart(positive: int, negative: int, neutral: int = 0) -> go.Figure:
    """Donut chart for sentiment distribution - SANITIZED"""
    # Sanitize all values
    positive = int(sanitize_number(positive, 0))
    negative = int(sanitize_number(negative, 0))
    neutral = int(sanitize_number(neutral, 0))
    
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive, negative, neutral]
    colors = [COLORS['success'], COLORS['danger'], COLORS['neutral']]
    
    # Remove zero values
    clean_labels = [l for l, v in zip(labels, values) if v > 0]
    clean_values = [v for v in values if v > 0]
    clean_colors = [c for c, v in zip(colors, values) if v > 0]
    
    # If all values are zero, show empty state
    if not clean_values:
        clean_labels = ['No Data']
        clean_values = [1]
        clean_colors = [COLORS['neutral']]
    
    fig = go.Figure(data=[go.Pie(
        labels=clean_labels, 
        values=clean_values, 
        hole=.6,
        marker=dict(colors=clean_colors),
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>%{value} comments<br>%{percent}<extra></extra>',
        textfont=dict(size=14, color=COLORS['text'])
    )])
    
    # Add center text
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=18, color=COLORS['text']),
        showarrow=False
    )
    
    _update_layout(fig, title=None, height=300)
    fig.update_layout(margin=dict(t=20, b=0, l=0, r=0))
    return fig

def create_engagement_gauge(score: float) -> go.Figure:
    """Gauge chart for overall sentiment - FULLY SANITIZED"""
    # Handle None or invalid score with sanitize_number
    score = sanitize_number(score, 0.0)
    
    # Ensure score is in valid range
    score = max(-1.0, min(1.0, float(score)))
    
    # Map -1..1 to 0..100 for gauge display
    val = (score + 1) * 50
    
    # Determine color and status text based on score - NO EMOJIS
    if score > 0.2:
        color = COLORS['success']
        status_text = "Positive"
        status_color = COLORS['success']
    elif score < -0.2:
        color = COLORS['danger']
        status_text = "Negative"
        status_color = COLORS['danger']
    else:
        color = COLORS['neutral']
        status_text = "Neutral"
        status_color = COLORS['neutral']
    
    # Ensure status_text is never None or undefined
    status_text = sanitize_text(status_text, "")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': status_text, 'font': {'size': 16, 'color': status_color}} if status_text else None,
        number = {'font': {'color': COLORS['text'], 'size': 42}, 'valueformat': '.2f'},
        gauge = {
            'axis': {
                'range': [0, 100], 
                'tickwidth': 1, 
                'tickcolor': COLORS['text_muted'],
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['-1', '-0.5', '0', '+0.5', '+1']
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(0,0,0,0.02)",
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 33], 'color': 'rgba(220, 38, 38, 0.15)'},   # Negatif (kırmızı)
                {'range': [33, 66], 'color': 'rgba(107, 114, 128, 0.10)'}, # Nötr (gri)
                {'range': [66, 100], 'color': 'rgba(5, 150, 105, 0.15)'}  # Pozitif (yeşil)
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 3},
                'thickness': 0.85,
                'value': val
            }
        }
    ))
    
    _update_layout(fig, height=300)
    fig.update_layout(margin=dict(t=80, b=20, l=30, r=30))
    return fig

def create_timeline_from_comments(comments: list, sentiment_results: list = None) -> go.Figure:
    """Line chart for sentiment trend"""
    if not sentiment_results:
        return go.Figure()

    # Create dummy time series based on index
    scores = []
    for res in sentiment_results:
        s = res.score
        if res.label == 'negative': s = -s
        elif res.label == 'neutral': s = 0
        scores.append(s)
        
    df = pd.DataFrame({'score': scores})
    # Smooth line
    df['ma'] = df['score'].rolling(window=max(5, len(df)//20), min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=df['ma'],
        mode='lines',
        name='Trend',
        line=dict(color=COLORS['primary'], width=4, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.12)'
    ))
    
    _update_layout(fig, title="Yorum Sırasına Göre Duygu Trendi", height=300)
    fig.update_xaxes(title="Yorum Sırası", showgrid=False)
    fig.update_yaxes(title="Duygu Skoru (-1 ile +1 arası)", range=[-1.1, 1.1])
    
    return fig

def generate_wordcloud(word_frequencies: Dict[str, int], width=800, height=400):
    """Generate WordCloud image (requires wordcloud library)"""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        wc = WordCloud(
            width=width, 
            height=height,
            background_color=None,
            mode="RGBA",
            colormap='cool',  # Blue/Cyan theme
            max_words=100
        ).generate_from_frequencies(word_frequencies)
        
        return wc.to_array()
    except ImportError:
        return None

# --- BATTLE MODE CHARTS ---

def create_category_comparison_chart(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Side-by-side bar chart for categories - SANITIZED"""
    cat_names = list(categories.keys())
    # Sanitize all category names
    cat_names = [sanitize_text(c, "Category") for c in cat_names]
    
    # Sanitize percentages
    v1_percents = [sanitize_number(categories[c].get('v1_percent', 0), 0) for c in categories.keys()]
    v2_percents = [sanitize_number(categories[c].get('v2_percent', 0), 0) for c in categories.keys()]
    
    # Get comment counts for labels
    v1_count = sum(sanitize_number(categories[c].get('v1_count', 0), 0) for c in categories.keys())
    v2_count = sum(sanitize_number(categories[c].get('v2_count', 0), 0) for c in categories.keys())
    
    # Sanitize video names
    safe_v1_name = sanitize_name(v1_name, "Video A")[:20]
    safe_v2_name = sanitize_name(v2_name, "Video B")[:20]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=f"{safe_v1_name} ({int(v1_count)})",
        x=cat_names,
        y=v1_percents,
        marker_color=COLORS['video1'],
        text=[f'{p:.0f}%' for p in v1_percents],
        textposition='auto',
        width=0.3,
        hovertemplate='<b>%{x}</b><br>Match: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name=f"{safe_v2_name} ({int(v2_count)})",
        x=cat_names,
        y=v2_percents,
        marker_color=COLORS['video2'],
        text=[f'{p:.0f}%' for p in v2_percents],
        textposition='auto',
        width=0.3,
        hovertemplate='<b>%{x}</b><br>Match: %{y:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Category Comparison (Match Rate)", height=350)
    fig.update_layout(
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1
    )
    fig.update_yaxes(title="Match Rate (%)")
    
    return fig

def create_category_radar_chart(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Radar chart comparison - SANITIZED"""
    cat_names = list(categories.keys())
    
    # Sanitize all category names
    cat_names = [sanitize_text(c, "Category") for c in cat_names]
    
    # Warning: Radar chart needs at least 3 categories
    if len(cat_names) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Radar chart requires at least 3 categories.<br>Current: {len(cat_names)} categories.<br><br>Add more categories or use<br>Side-by-Side view.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text_muted']),
            align="center"
        )
        _update_layout(fig, title="Radar View (Insufficient Categories)", height=350)
        return fig
    
    # Sanitize values
    v1_vals = [sanitize_number(categories[c].get('v1_percent', 0), 0) for c in categories.keys()]
    v2_vals = [sanitize_number(categories[c].get('v2_percent', 0), 0) for c in categories.keys()]
    
    # Close the loop
    cat_names_loop = cat_names + [cat_names[0]]
    v1_vals_loop = v1_vals + [v1_vals[0]]
    v2_vals_loop = v2_vals + [v2_vals[0]]
    
    # Sanitize video names
    safe_v1_name = sanitize_name(v1_name, "Video A")[:25]
    safe_v2_name = sanitize_name(v2_name, "Video B")[:25]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=v1_vals_loop,
        theta=cat_names_loop,
        fill='toself',
        name=safe_v1_name,
        line_color=COLORS['video1'],
        fillcolor='rgba(74, 144, 226, 0.2)',
        hovertemplate='%{theta}<br>Rate: %{r:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=v2_vals_loop,
        theta=cat_names_loop,
        fill='toself',
        name=safe_v2_name,
        line_color=COLORS['video2'],
        fillcolor='rgba(108, 92, 231, 0.2)',
        hovertemplate='%{theta}<br>Rate: %{r:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Category Radar (Match Rate %)", height=400)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100], 
                gridcolor=COLORS['grid'],
                ticksuffix='%'
            ),
            bgcolor='rgba(255, 255, 255, 0.02)'
        )
    )
    
    return fig

def create_winner_summary_chart(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Win count summary"""
    v1_wins = sum(1 for c in categories.values() if c['v1_percent'] > c['v2_percent'])
    v2_wins = sum(1 for c in categories.values() if c['v2_percent'] > c['v1_percent'])
    draws = len(categories) - v1_wins - v2_wins
    
    labels = [v1_name[:15], 'Draw', v2_name[:15]]
    values = [v1_wins, draws, v2_wins]
    colors = [COLORS['primary'], COLORS['neutral'], COLORS['secondary']]
    
    fig = go.Figure(go.Bar(
        x=labels, 
        y=values, 
        marker_color=colors,
        text=values,
        textposition='auto',
        width=0.5
    ))
    
    _update_layout(fig, title="Categories Won", height=250)
    fig.update_yaxes(visible=False)
    
    return fig

def create_category_heatmap(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Heatmap view - category match rates with monochromatic blue scale (enterprise style)"""
    cat_names = list(categories.keys())
    cat_names = [c if c else "Unknown" for c in cat_names]  # Handle undefined
    
    # Matrix: row=video, col=category
    z = [
        [categories[c]['v1_percent'] for c in categories.keys()],
        [categories[c]['v2_percent'] for c in categories.keys()]
    ]
    
    # Calculate comment counts for tooltip
    v1_total = sum(categories[c].get('v1_count', 0) for c in categories.keys())
    v2_total = sum(categories[c].get('v2_count', 0) for c in categories.keys())
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=cat_names,
        y=[f"{v1_name[:20]} ({v1_total} comments)" if v1_name else "Video 1", 
           f"{v2_name[:20]} ({v2_total} comments)" if v2_name else "Video 2"],
        # Monochromatic Blue scale (Enterprise Style)
        colorscale=[
            [0, '#F0F9FF'],      # 0% - Very Light Blue
            [0.2, '#BAE6FD'],    # 20% - Light Blue
            [0.4, '#7DD3FC'],    # 40% - Sky Blue
            [0.6, '#38BDF8'],    # 60% - Bright Blue
            [0.8, '#0EA5E9'],    # 80% - Blue
            [1, '#0369A1']       # 100% - Dark Blue
        ],
        texttemplate="%{z:.0f}%",
        textfont=dict(color='#1F2937', size=12),
        showscale=True,
        colorbar=dict(
            title="Match %",
            ticksuffix="%",
            len=0.8
        ),
        hovertemplate='<b>%{x}</b><br>%{y}<br>Match: %{z:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Category Match Heatmap", height=280)
    fig.update_layout(margin=dict(t=40, b=20))
    
    return fig


def create_battle_trend_chart(
    v1_sentiments: list, 
    v2_sentiments: list, 
    v1_name: str, 
    v2_name: str
) -> go.Figure:
    """
    Create a dual-line chart showing sentiment trends for both videos.
    
    Args:
        v1_sentiments: List of sentiment results for video 1
        v2_sentiments: List of sentiment results for video 2
        v1_name: Name of video 1
        v2_name: Name of video 2
    
    Returns:
        Plotly figure with two trend lines
    """
    def extract_scores(sentiments):
        """Convert sentiment results to numeric scores"""
        scores = []
        for s in sentiments:
            if hasattr(s, 'label') and hasattr(s, 'score'):
                if s.label == 'positive':
                    scores.append(s.score)
                elif s.label == 'negative':
                    scores.append(-s.score)
                else:
                    scores.append(0)
            elif isinstance(s, dict):
                label = s.get('label', 'neutral')
                score = s.get('score', 0)
                if label == 'positive':
                    scores.append(score)
                elif label == 'negative':
                    scores.append(-score)
                else:
                    scores.append(0)
        return scores
    
    v1_scores = extract_scores(v1_sentiments)
    v2_scores = extract_scores(v2_sentiments)
    
    if not v1_scores and not v2_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="Sentiment verisi bulunamadı",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text_muted'])
        )
        _update_layout(fig, height=350)
        return fig
    
    fig = go.Figure()
    
    # Video 1 trend line
    if v1_scores:
        df1 = pd.DataFrame({'score': v1_scores})
        window = max(3, len(df1) // 10)
        df1['ma'] = df1['score'].rolling(window=window, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            y=df1['ma'],
            mode='lines',
            name=v1_name[:20],
            line=dict(color=COLORS['primary'], width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.15)',
            hovertemplate='%{y:.2f}<extra>' + v1_name[:15] + '</extra>'
        ))
    
    # Video 2 trend line
    if v2_scores:
        df2 = pd.DataFrame({'score': v2_scores})
        window = max(3, len(df2) // 10)
        df2['ma'] = df2['score'].rolling(window=window, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            y=df2['ma'],
            mode='lines',
            name=v2_name[:20],
            line=dict(color=COLORS['secondary'], width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.15)',
            hovertemplate='%{y:.2f}<extra>' + v2_name[:15] + '</extra>'
        ))
    
    _update_layout(fig, title="Duygu Trendi (Yorum Sırasına Göre)", height=350)
    fig.update_xaxes(
        title="Yorum Sırası (1 = ilk yorum)", 
        showgrid=False,
        dtick=10
    )
    fig.update_yaxes(
        title="Duygu Skoru (-1=Negatif, +1=Pozitif)", 
        range=[-1.1, 1.1], 
        zeroline=True, 
        zerolinecolor='rgba(255,255,255,0.3)',
        zerolinewidth=2
    )
    
    # Add annotation explaining the chart
    fig.add_annotation(
        text="📊 Her nokta bir yorumun duygu ortalaması",
        xref="paper", yref="paper",
        x=0.02, y=1.12, showarrow=False,
        font=dict(size=10, color=COLORS['text_muted']),
        align="left"
    )
    
    return fig


def create_keyword_bar_chart(keywords: Dict[str, int], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for keyword frequencies"""
    if not keywords:
        return go.Figure()
    
    # Handle both dict and list of dicts
    if isinstance(keywords, list):
        # If it's a list of video dicts, extract titles and count words
        text = " ".join([v.get('baslik', '') for v in keywords if isinstance(v, dict)])
        words = [w for w in text.split() if len(w) > 3]
        keywords = dict(Counter(words).most_common(top_n))
    
    sorted_kw = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    if not sorted_kw:
        return go.Figure()
    
    fig = go.Figure(go.Bar(
        x=list(sorted_kw.values()),
        y=list(sorted_kw.keys()),
        orientation='h',
        marker=dict(
            color=list(sorted_kw.values()),
            colorscale='Blues',
            showscale=False
        ),
        text=list(sorted_kw.values()),
        textposition='outside'
    ))
    
    _update_layout(fig, title=f"En Çok Kullanılan {len(sorted_kw)} Kelime", height=400)
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120)
    )
    
    return fig


def create_battle_comparison(v1_score, v2_score, v1_name, v2_name) -> go.Figure:
    """Simple bar chart comparing two video scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=v1_name[:20],
        x=[v1_name[:20]], 
        y=[v1_score],
        marker_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Bar(
        name=v2_name[:20],
        x=[v2_name[:20]], 
        y=[v2_score],
        marker_color=COLORS['secondary']
    ))
    
    _update_layout(fig, height=300)
    fig.update_layout(barmode='group')
    
    return fig


def create_sentiment_bubble_chart(positive: int, negative: int, neutral: int) -> go.Figure:
    """Bubble chart for sentiment distribution"""
    total = positive + negative + neutral
    if total == 0:
        total = 1
    
    labels = ['Pozitif', 'Negatif', 'Nötr']
    counts = [positive, negative, neutral]
    percentages = [c / total * 100 for c in counts]
    colors = [COLORS['success'], COLORS['danger'], COLORS['neutral']]
    
    # Bubble sizes (min 30, max 100)
    max_count = max(counts) if max(counts) > 0 else 1
    sizes = [max(30, min(100, c / max_count * 70 + 30)) for c in counts]
    
    fig = go.Figure()
    
    for i, (label, count, pct, color, size) in enumerate(zip(labels, counts, percentages, colors, sizes)):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[pct],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(color='white', width=2),
                opacity=0.85
            ),
            text=[f'{count}<br>({pct:.0f}%)'],
            textposition='middle center',
            textfont=dict(color='white', size=11),
            name=label,
            hovertemplate=f'<b>{label}</b><br>Sayı: {count}<br>Yüzde: {pct:.1f}%<extra></extra>'
        ))
    
    _update_layout(fig, title="Duygu Dağılımı (Baloncuk)", height=320)
    fig.update_layout(
        xaxis=dict(
            showgrid=False, 
            showticklabels=True,
            tickvals=[0, 1, 2],
            ticktext=labels
        ),
        yaxis=dict(title='Yüzde (%)', range=[0, max(percentages) * 1.3] if max(percentages) > 0 else [0, 100]),
        showlegend=False
    )
    
    return fig


def create_temporal_sentiment_chart(
    comments: List[Dict], 
    sentiments: list,
    title: str = "Zamana Bağlı Duygu Değişimi"
) -> go.Figure:
    """
    Create a simple line chart showing positive/negative percentages over time.
    
    Args:
        comments: List of comment dicts with 'timestamp' field
        sentiments: List of sentiment results matching comments
        title: Chart title
    
    Returns:
        Plotly figure with two lines (positive/negative %)
    """
    from datetime import datetime as dt
    from collections import defaultdict
    
    # Group comments by date
    daily_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
    
    for i, (comment, sentiment) in enumerate(zip(comments, sentiments)):
        timestamp = comment.get('timestamp', 0)
        
        # Skip if no valid timestamp
        if not timestamp or timestamp == 0:
            continue
        
        # Convert timestamp to date
        try:
            if isinstance(timestamp, (int, float)):
                date = dt.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            else:
                continue
        except (ValueError, OSError):
            continue
        
        # Get sentiment label
        if hasattr(sentiment, 'label'):
            label = sentiment.label
        elif isinstance(sentiment, dict):
            label = sentiment.get('label', 'neutral')
        else:
            continue
        
        daily_stats[date]['total'] += 1
        if label == 'positive':
            daily_stats[date]['positive'] += 1
        elif label == 'negative':
            daily_stats[date]['negative'] += 1
        else:
            daily_stats[date]['neutral'] += 1
    
    if not daily_stats:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Tarih bilgisi olan yorum bulunamadı",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text_muted'])
        )
        _update_layout(fig, height=350)
        return fig
    
    # Sort by date and calculate percentages
    sorted_dates = sorted(daily_stats.keys())
    pos_percentages = []
    neg_percentages = []
    
    for date in sorted_dates:
        stats = daily_stats[date]
        total = stats['total']
        pos_percentages.append(stats['positive'] / total * 100 if total > 0 else 0)
        neg_percentages.append(stats['negative'] / total * 100 if total > 0 else 0)
    
    fig = go.Figure()
    
    # Positive line - green
    fig.add_trace(go.Scatter(
        x=sorted_dates,
        y=pos_percentages,
        mode='lines+markers',
        name='Pozitif %',
        line=dict(color=COLORS['success'], width=4),
        marker=dict(size=10),
        hovertemplate='%{x}<br>Pozitif: %{y:.1f}%<extra></extra>'
    ))
    
    # Negative line - red
    fig.add_trace(go.Scatter(
        x=sorted_dates,
        y=neg_percentages,
        mode='lines+markers',
        name='Negatif %',
        line=dict(color=COLORS['danger'], width=4),
        marker=dict(size=10),
        hovertemplate='%{x}<br>Negatif: %{y:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title=title, height=400)
    fig.update_layout(
        xaxis=dict(
            title="Tarih",
            type='category',
            tickangle=-45,
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        yaxis=dict(
            title="Yüzde (%)",
            range=[0, 100],
            showgrid=True,
            gridcolor=COLORS['grid']
        ),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig


def create_category_pie_grid(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """
    Create a grid of pie charts showing match distribution for each category.
    Enterprise Style: Blue = Matched, Light Gray = Not matched
    """
    from plotly.subplots import make_subplots
    
    cat_names = list(categories.keys())
    num_cats = len(cat_names)
    
    if num_cats == 0:
        fig = go.Figure()
        fig.add_annotation(text="No categories found", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate grid size
    cols = min(3, num_cats)
    rows = (num_cats + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'pie'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[c[:20] for c in cat_names],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    # SUCCESS/FAIL COLORS: Green for matched, Red for not matched
    MATCHED_COLOR = '#2ECC71'  # Emerald Green - success/matched
    OTHER_COLOR = '#E74C3C'    # Vivid Red - fail/not matched
    
    for i, cat_name in enumerate(cat_names):
        cat_data = categories[cat_name]
        row = i // cols + 1
        col = i % cols + 1
        
        # V1 data - matched vs not matched
        v1_matched = cat_data.get('v1_count', 0)
        v1_total = cat_data.get('v1_total', v1_matched)
        v1_not_matched = max(0, v1_total - v1_matched) if v1_total > v1_matched else 0
        
        # If we don't have total, estimate from percent
        if v1_not_matched == 0 and cat_data.get('v1_percent', 0) > 0:
            v1_percent = cat_data['v1_percent']
            if v1_percent < 100:
                v1_not_matched = int(v1_matched * (100 - v1_percent) / v1_percent) if v1_percent > 0 else 0
        
        fig.add_trace(go.Pie(
            labels=['Matched', 'Other'],
            values=[v1_matched, max(1, v1_not_matched)],
            marker=dict(
                colors=[MATCHED_COLOR, OTHER_COLOR],  # Blue + Light Gray
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='percent',
            textfont=dict(size=12, color='#1F2937'),
            hole=0.4,
            name=cat_name[:15],
            hovertemplate=f'<b>{cat_name[:20]}</b><br>%{{label}}: %{{value}} comments<br>(%{{percent}})<extra></extra>'
        ), row=row, col=col)
    
    _update_layout(fig, title="Category Match Distribution", height=max(400, rows * 300))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.05,
            xanchor="center", 
            x=0.5,
            font=dict(size=12, color=COLORS['text'])
        ),
        margin=dict(t=60, b=60, l=60, r=60)
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color=COLORS['text'])
        annotation['y'] = annotation['y'] + 0.05
    
    return fig


def create_category_temporal_chart(
    categories: Dict[str, Dict],
    v1_comments: List[Dict],
    v2_comments: List[Dict],
    v1_sentiments: list,
    v2_sentiments: list,
    v1_name: str,
    v2_name: str
) -> go.Figure:
    """
    Create BAR CHARTS showing positive/negative sentiment counts over time for each category.
    Green bars = Positive, Red bars = Negative
    """
    from plotly.subplots import make_subplots
    from datetime import datetime
    
    cat_names = list(categories.keys())
    num_cats = len(cat_names)
    
    if num_cats == 0:
        fig = go.Figure()
        fig.add_annotation(text="Kategori bulunamadı", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate grid size
    cols = min(2, num_cats)
    rows = (num_cats + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{c[:25]}" for c in cat_names],
        vertical_spacing=0.20,
        horizontal_spacing=0.12
    )
    
    # Current date for filtering - STRICT filter
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Process sentiment by time for each video
    def get_sentiment_by_date(comments, sentiments):
        """Group sentiments by date, STRICTLY filtering out future dates"""
        date_sentiment = {}
        for i, c in enumerate(comments):
            if i >= len(sentiments):
                break
            
            ts = c.get('timestamp')
            if ts:
                try:
                    # Handle both Unix timestamp and string dates
                    if isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts)
                    else:
                        continue  # Skip if not a valid timestamp
                    
                    # STRICT filter: No future dates at all
                    if dt.year > current_year:
                        continue
                    if dt.year == current_year and dt.month > current_month:
                        continue
                    
                    # Only use valid years (2020-current)
                    if dt.year < 2020:
                        continue
                        
                    date = dt.strftime('%Y-%m')
                except:
                    continue
                
                if date not in date_sentiment:
                    date_sentiment[date] = {'pos': 0, 'neg': 0, 'total': 0}
                
                s = sentiments[i]
                label = s.label if hasattr(s, 'label') else s.get('label', 'neutral')
                
                if label == 'positive':
                    date_sentiment[date]['pos'] += 1
                elif label == 'negative':
                    date_sentiment[date]['neg'] += 1
                date_sentiment[date]['total'] += 1
        
        return date_sentiment
    
    # Get combined sentiment data
    v1_date_data = get_sentiment_by_date(v1_comments, v1_sentiments if v1_sentiments else [])
    v2_date_data = get_sentiment_by_date(v2_comments, v2_sentiments if v2_sentiments else [])
    
    # Combine all dates and sort, filter to only dates with data
    all_dates = sorted(set(list(v1_date_data.keys()) + list(v2_date_data.keys())))
    
    # If no valid dates, show message
    if not all_dates:
        fig = go.Figure()
        fig.add_annotation(
            text="Geçerli tarih verisi bulunamadı",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text_muted'])
        )
        _update_layout(fig, title="Zaman Trendi", height=300)
        return fig
    
    for i, cat_name in enumerate(cat_names):
        row = i // cols + 1
        col = i % cols + 1
        
        # Calculate positive/negative for this category over time
        pos_values = []
        neg_values = []
        
        for date in all_dates:
            v1_data = v1_date_data.get(date, {'pos': 0, 'neg': 0})
            v2_data = v2_date_data.get(date, {'pos': 0, 'neg': 0})
            pos_values.append(v1_data['pos'] + v2_data['pos'])
            neg_values.append(v1_data['neg'] + v2_data['neg'])
        
        # Positive BARS - GREEN
        fig.add_trace(go.Bar(
            x=all_dates,
            y=pos_values,
            name='Pozitif' if i == 0 else None,
            showlegend=(i == 0),
            marker=dict(color='#10B981', line=dict(width=0)),
            legendgroup='pos',
            hovertemplate='%{x}<br>Pozitif: %{y}<extra></extra>'
        ), row=row, col=col)
        
        # Negative BARS - RED
        fig.add_trace(go.Bar(
            x=all_dates,
            y=neg_values,
            name='Negatif' if i == 0 else None,
            showlegend=(i == 0),
            marker=dict(color='#EF4444', line=dict(width=0)),
            legendgroup='neg',
            hovertemplate='%{x}<br>Negatif: %{y}<extra></extra>'
        ), row=row, col=col)
    
    _update_layout(fig, title="Kategorilere Göre Zamana Bağlı Duygu Analizi", height=max(420, rows * 300))
    fig.update_layout(
        barmode='group',  # Side by side bars
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.08,
            xanchor="center", 
            x=0.5,
            font=dict(size=13)
        ),
        hovermode='x unified',
        margin=dict(t=110, b=60, l=50, r=40)
    )
    
    # Update all x-axes and y-axes
    fig.update_xaxes(
        tickfont=dict(size=10, color=COLORS['text_muted']),
        tickangle=-45,
        showgrid=False
    )
    fig.update_yaxes(
        tickfont=dict(size=11, color=COLORS['text_muted']),
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.1)',
        gridwidth=1
    )
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=COLORS['text'])
    
    return fig

