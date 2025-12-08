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
    labels = ['Positive', 'Negative', 'Neutral']
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
        hoverinfo='label+value',
        textfont=dict(size=14)
    )])
    
    # Add center text
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=18, color=COLORS['text']),
        showarrow=False
    )
    
    _update_layout(fig, height=280)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
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
        title = {'text': "Sentiment Index", 'font': {'size': 14, 'color': COLORS['text_muted']}},
        number = {'font': {'color': COLORS['text'], 'size': 30}, 'suffix': ''},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 100], 'color': 'rgba(255,255,255,0.05)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': val
            }
        }
    ))
    
    _update_layout(fig, height=250)
    fig.update_layout(margin=dict(t=40, b=10, l=30, r=30))
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
    
    _update_layout(fig, title="Sentiment Trend Over Comments", height=300)
    fig.update_xaxes(title="Comment Sequence", showgrid=False)
    fig.update_yaxes(title="Sentiment Value", range=[-1.1, 1.1])
    
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
    v1_percents = [categories[c]['v1_percent'] for c in cat_names]
    v2_percents = [categories[c]['v2_percent'] for c in cat_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=v1_name[:15],
        x=cat_names,
        y=v1_percents,
        marker_color=COLORS['primary'],
        text=[f'{p:.0f}%' for p in v1_percents],
        textposition='auto',
        width=0.3
    ))
    
    fig.add_trace(go.Bar(
        name=v2_name[:15],
        x=cat_names,
        y=v2_percents,
        marker_color=COLORS['secondary'],
        text=[f'{p:.0f}%' for p in v2_percents],
        textposition='auto',
        width=0.3
    ))
    
    _update_layout(fig, height=350)
    fig.update_layout(
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1
    )
    fig.update_yaxes(title="Match Rate (%)")
    
    return fig

def create_category_radar_chart(categories: Dict[str, Dict], v1_name: str, v2_name: str) -> go.Figure:
    """Radar chart comparison"""
    cat_names = list(categories.keys())
    v1_vals = [categories[c]['v1_percent'] for c in cat_names]
    v2_vals = [categories[c]['v2_percent'] for c in cat_names]
    
    # Close the loop
    if cat_names:
        cat_names += [cat_names[0]]
        v1_vals += [v1_vals[0]]
        v2_vals += [v2_vals[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=v1_vals,
        theta=cat_names,
        fill='toself',
        name=v1_name[:15],
        line_color=COLORS['primary'],
        fillcolor='rgba(59, 130, 246, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=v2_vals,
        theta=cat_names,
        fill='toself',
        name=v2_name[:15],
        line_color=COLORS['secondary'],
        fillcolor='rgba(139, 92, 246, 0.2)'
    ))
    
    _update_layout(fig, height=400)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS['grid']),
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
    """Heatmap view"""
    cat_names = list(categories.keys())
    # Matrix: row=video, col=category
    z = [
        [categories[c]['v1_percent'] for c in cat_names],
        [categories[c]['v2_percent'] for c in cat_names]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=cat_names,
        y=[v1_name[:15], v2_name[:15]],
        colorscale='PuBu', # Purple-Blue similar to our theme
        texttemplate="%{z:.0f}%",
        showscale=False
    ))
    
    _update_layout(fig, height=250)
    fig.update_layout(margin=dict(t=20, b=20))
    
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
    
    _update_layout(fig, title=f"Top {len(sorted_kw)} Keywords", height=400)
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
    
    labels = ['Positive', 'Negative', 'Neutral']
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
            hovertemplate=f'<b>{label}</b><br>Count: {count}<br>Percent: {pct:.1f}%<extra></extra>'
        ))
    
    _update_layout(fig, title="Sentiment Distribution (Bubble)", height=320)
    fig.update_layout(
        xaxis=dict(
            showgrid=False, 
            showticklabels=True,
            tickvals=[0, 1, 2],
            ticktext=labels
        ),
        yaxis=dict(title='Percentage (%)', range=[0, max(percentages) * 1.3] if max(percentages) > 0 else [0, 100]),
        showlegend=False
    )
    
    return fig

