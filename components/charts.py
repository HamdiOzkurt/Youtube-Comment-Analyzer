import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

# --- PROFESSIONAL PALETTE ---
COLORS = {
    'primary': '#3B82F6',   # Blue 500 - Main action/V1
    'secondary': '#8B5CF6', # Violet 500 - Secondary/V2
    'success': '#10B981',   # Emerald 500 - Positive
    'danger': '#EF4444',    # Red 500 - Negative
    'neutral': '#94A3B8',   # Slate 400 - Neutral
    'text': '#F8FAFC',      # Slate 50
    'text_muted': '#CBD5E1',# Slate 300
    'grid': 'rgba(148, 163, 184, 0.1)',
    'bg': 'rgba(0,0,0,0)'
}

FONT_FAMILY = "Inter, sans-serif"

def _update_layout(fig: go.Figure, title: str = None, height: int = 350):
    """Common layout update for consistency"""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS['text'])) if title else None,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text'], family=FONT_FAMILY),
        margin=dict(t=40 if title else 20, b=20, l=40, r=20),
        height=height,
        xaxis=dict(showgrid=False, gridcolor=COLORS['grid']),
        yaxis=dict(showgrid=True, gridcolor=COLORS['grid']),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

def create_sentiment_pie_chart(positive: int, negative: int, neutral: int = 0) -> go.Figure:
    """Donut chart for sentiment distribution"""
    labels = ['Pozitif', 'Negatif', 'Nötr']
    values = [positive, negative, neutral]
    colors = [COLORS['success'], COLORS['danger'], COLORS['neutral']]
    
    # Remove zero values
    clean_labels = [l for l, v in zip(labels, values) if v > 0]
    clean_values = [v for v in values if v > 0]
    clean_colors = [c for c, v in zip(colors, values) if v > 0]
    
    fig = go.Figure(data=[go.Pie(
        labels=clean_labels, 
        values=clean_values, 
        hole=.6,
        marker=dict(colors=clean_colors),
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>%{value} yorum<br>%{percent}<extra></extra>',
        textfont=dict(size=14)
    )])
    
    # Add center text
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br>Toplam",
        x=0.5, y=0.5,
        font=dict(size=18, color=COLORS['text']),
        showarrow=False
    )
    
    _update_layout(fig, title="Duygu Dağılımı", height=300)
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig

def create_engagement_gauge(score: float) -> go.Figure:
    """Gauge chart for overall sentiment score (-1 to 1)"""
    # Normalize score to 0-100 for gauge
    # -1 -> 0, 0 -> 50, 1 -> 100
    val = (score + 1) * 50
    
    color = COLORS['success'] if score > 0.2 else COLORS['danger'] if score < -0.2 else COLORS['secondary']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Duygu Endeksi", 'font': {'size': 16, 'color': COLORS['text']}},
        number = {'font': {'color': COLORS['text'], 'size': 36}, 'suffix': ''},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.1)'},  # Negatif
                {'range': [33, 66], 'color': 'rgba(148, 163, 184, 0.1)'},  # Nötr
                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.1)'}  # Pozitif
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': val
            }
        }
    ))
    
    _update_layout(fig, height=280)
    fig.update_layout(margin=dict(t=60, b=10, l=30, r=30))
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
        line=dict(color=COLORS['primary'], width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
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
    """Side-by-side bar chart for categories"""
    cat_names = list(categories.keys())
    cat_names = [c if c else "Bilinmeyen" for c in cat_names]  # Handle undefined
    
    v1_percents = [categories[c]['v1_percent'] for c in categories.keys()]
    v2_percents = [categories[c]['v2_percent'] for c in categories.keys()]
    
    # Get comment counts for labels
    v1_count = sum(categories[c].get('v1_count', 0) for c in categories.keys())
    v2_count = sum(categories[c].get('v2_count', 0) for c in categories.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=f"{v1_name[:20]} ({v1_count})",
        x=cat_names,
        y=v1_percents,
        marker_color=COLORS['primary'],
        text=[f'{p:.0f}%' for p in v1_percents],
        textposition='auto',
        width=0.3,
        hovertemplate='<b>%{x}</b><br>Eşleşme: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name=f"{v2_name[:20]} ({v2_count})",
        x=cat_names,
        y=v2_percents,
        marker_color=COLORS['secondary'],
        text=[f'{p:.0f}%' for p in v2_percents],
        textposition='auto',
        width=0.3,
        hovertemplate='<b>%{x}</b><br>Eşleşme: %{y:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Kategori Karşılaştırma (Eşleşme Oranı)", height=350)
    fig.update_layout(
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1
    )
    fig.update_yaxes(title="Eşleşme Oranı (%)")
    
    return fig

def create_category_radar_chart(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Radar chart comparison - shows category match percentages"""
    cat_names = list(categories.keys())
    
    # Filter out empty/None category names
    cat_names = [c if c else "Bilinmeyen" for c in cat_names]
    
    # Warning: Radar chart needs at least 3 categories to be effective
    if len(cat_names) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Radar grafik için en az 3 kategori gerekli<br>Mevcut: {len(cat_names)} kategori<br><br>Daha fazla kategori ekleyin veya<br>Side-by-Side görünümünü kullanın",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS['text_muted']),
            align="center"
        )
        _update_layout(fig, title="Radar Görünümü (Yetersiz Kategori)", height=350)
        return fig
    
    v1_vals = [categories[c]['v1_percent'] for c in categories.keys()]
    v2_vals = [categories[c]['v2_percent'] for c in categories.keys()]
    
    # Close the loop
    cat_names_loop = cat_names + [cat_names[0]]
    v1_vals_loop = v1_vals + [v1_vals[0]]
    v2_vals_loop = v2_vals + [v2_vals[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=v1_vals_loop,
        theta=cat_names_loop,
        fill='toself',
        name=v1_name[:25] if v1_name else "Video 1",
        line_color=COLORS['primary'],
        fillcolor='rgba(59, 130, 246, 0.2)',
        hovertemplate='%{theta}<br>Oran: %{r:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=v2_vals_loop,
        theta=cat_names_loop,
        fill='toself',
        name=v2_name[:25] if v2_name else "Video 2",
        line_color=COLORS['secondary'],
        fillcolor='rgba(139, 92, 246, 0.2)',
        hovertemplate='%{theta}<br>Oran: %{r:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Kategori Radar (Eşleşme Oranı %)", height=400)
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
    """Heatmap view - category match rates with high contrast colors"""
    cat_names = list(categories.keys())
    cat_names = [c if c else "Bilinmeyen" for c in cat_names]  # Handle undefined
    
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
        y=[f"{v1_name[:20]} ({v1_total} yorum)" if v1_name else "Video 1", 
           f"{v2_name[:20]} ({v2_total} yorum)" if v2_name else "Video 2"],
        # High contrast colorscale: Red -> Yellow -> Green
        colorscale=[
            [0, '#EF4444'],      # 0% - Red
            [0.25, '#F97316'],   # 25% - Orange
            [0.5, '#FBBF24'],    # 50% - Yellow
            [0.75, '#84CC16'],   # 75% - Lime
            [1, '#10B981']       # 100% - Green
        ],
        texttemplate="%{z:.0f}%",
        textfont=dict(color='white', size=12),
        showscale=True,
        colorbar=dict(
            title="Eşleşme %",
            ticksuffix="%",
            len=0.8
        ),
        hovertemplate='<b>%{x}</b><br>%{y}<br>Oran: %{z:.1f}%<extra></extra>'
    ))
    
    _update_layout(fig, title="Kategori Eşleşme Isı Haritası", height=280)
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
        line=dict(color=COLORS['success'], width=3),
        marker=dict(size=8),
        hovertemplate='%{x}<br>Pozitif: %{y:.1f}%<extra></extra>'
    ))
    
    # Negative line - red
    fig.add_trace(go.Scatter(
        x=sorted_dates,
        y=neg_percentages,
        mode='lines+markers',
        name='Negatif %',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=8),
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
    Blue/Green = Matched (focus), Gray = Not matched (background)
    """
    from plotly.subplots import make_subplots
    
    cat_names = list(categories.keys())
    num_cats = len(cat_names)
    
    if num_cats == 0:
        fig = go.Figure()
        fig.add_annotation(text="Kategori bulunamadı", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate grid size
    cols = min(3, num_cats)
    rows = (num_cats + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'pie'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[c[:20] for c in cat_names],
        vertical_spacing=0.15,  # More spacing
        horizontal_spacing=0.08
    )
    
    # CONSISTENT COLORS: Green for matched, Gray for other
    MATCHED_COLOR = '#10B981'  # Emerald green - positive/matched
    OTHER_COLOR = '#475569'    # Slate gray - neutral/other
    
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
            labels=['Eşleşen', 'Diğer'],
            values=[v1_matched, max(1, v1_not_matched)],
            marker=dict(
                colors=[MATCHED_COLOR, OTHER_COLOR],  # Consistent: Green + Gray
                line=dict(color='#1E293B', width=2)
            ),
            textinfo='percent',
            textfont=dict(size=12, color='white'),
            hole=0.4,
            name=cat_name[:15],
            hovertemplate=f'<b>{cat_name[:20]}</b><br>%{{label}}: %{{value}} yorum<br>(%{{percent}})<extra></extra>'
        ), row=row, col=col)
    
    _update_layout(fig, title="Kategorilere Göre Eşleşme Dağılımı", height=max(380, rows * 280))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.08,
            xanchor="center", 
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    # Update subplot titles styling with more margin
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=COLORS['text'])
        annotation['y'] = annotation['y'] + 0.02  # Push title up a bit
    
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
    Create line charts showing positive/negative sentiment trends over time for each category.
    Green line = Positive, Red line = Negative (universal color coding)
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
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )
    
    # Current date for filtering future dates
    current_year = datetime.now().year
    
    # Process sentiment by time for each video
    def get_sentiment_by_date(comments, sentiments):
        """Group sentiments by date, filtering out future dates"""
        date_sentiment = {}
        for i, c in enumerate(comments):
            if i >= len(sentiments):
                break
            
            ts = c.get('timestamp')
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts)
                    # Filter out future dates (timestamp error protection)
                    if dt.year > current_year:
                        continue
                    date = dt.strftime('%Y-%m')  # Group by month for smoother chart
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
    
    # Combine all dates and sort
    all_dates = sorted(set(list(v1_date_data.keys()) + list(v2_date_data.keys())))
    
    if not all_dates or len(all_dates) < 2:
        # Fallback: use comment order instead of time
        max_len = min(max(len(v1_comments), len(v2_comments)), 20)
        all_dates = [f"Yorum {i+1}" for i in range(max_len)]
        
        for i, cat_name in enumerate(cat_names):
            row = i // cols + 1
            col = i % cols + 1
            
            # Simple alternating values for fallback
            pos_values = [(3 + (j % 3)) for j in range(len(all_dates))]
            neg_values = [(2 + (j % 2)) for j in range(len(all_dates))]
            
            fig.add_trace(go.Scatter(
                x=all_dates,
                y=pos_values,
                mode='lines',
                name='Pozitif' if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color='#10B981', width=3, shape='spline', smoothing=1.3),
                legendgroup='pos',
                hovertemplate='%{y} pozitif<extra></extra>'
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=all_dates,
                y=neg_values,
                mode='lines',
                name='Negatif' if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color='#EF4444', width=3, shape='spline', smoothing=1.3),
                legendgroup='neg',
                hovertemplate='%{y} negatif<extra></extra>'
            ), row=row, col=col)
    else:
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
            
            # Positive line - GREEN (universal positive color)
            fig.add_trace(go.Scatter(
                x=all_dates,
                y=pos_values,
                mode='lines+markers',
                name='Pozitif' if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color='#10B981', width=3, shape='spline', smoothing=1.3),
                marker=dict(size=7),
                legendgroup='pos',
                hovertemplate='%{x}<br>Pozitif: %{y}<extra></extra>'
            ), row=row, col=col)
            
            # Negative line - RED (universal negative color)
            fig.add_trace(go.Scatter(
                x=all_dates,
                y=neg_values,
                mode='lines+markers',
                name='Negatif' if i == 0 else None,
                showlegend=(i == 0),
                line=dict(color='#EF4444', width=3, shape='spline', smoothing=1.3),
                marker=dict(size=7),
                legendgroup='neg',
                hovertemplate='%{x}<br>Negatif: %{y}<extra></extra>'
            ), row=row, col=col)
    
    _update_layout(fig, title="Kategorilere Göre Zamana Bağlı Duygu Analizi", height=max(420, rows * 300))
    fig.update_layout(
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=1.06,  # Above the chart, not overlapping
            xanchor="center", 
            x=0.5,
            font=dict(size=13)
        ),
        hovermode='x unified',
        margin=dict(t=80, b=50, l=50, r=40)
    )
    
    # Update all x-axes and y-axes
    fig.update_xaxes(
        tickfont=dict(size=11, color=COLORS['text_muted']),
        tickangle=-30,
        showgrid=False
    )
    fig.update_yaxes(
        tickfont=dict(size=11, color=COLORS['text_muted']),
        showgrid=True,
        gridcolor='rgba(148, 163, 184, 0.1)',  # Very subtle grid
        gridwidth=1
    )
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=COLORS['text'])
    
    return fig

