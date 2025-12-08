"""
YouTube Comment Analyzer 2.0 - Streamlit Dashboard
Minimal & Professional UI with Multi-Video Search (Ollama Integration)
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import sys
import requests
import plotly.graph_objects as go

# Sayfa yapılandırması
st.set_page_config(
    page_title="YCA",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modül imports
try:
    from nlp_processor import NLPProcessor
    from sentiment_analyzer import SentimentAnalyzer
    from ollama_llm import OllamaLLM
    from content_assistant import ContentAssistant
    from comment_worker import CommentWorker
    from main import BulkCommentScraper
    from search_worker import SearchWorker
    from data_manager import DataManager
    from components.charts import (
        create_sentiment_pie_chart,
        create_engagement_gauge,
        create_keyword_bar_chart,
        create_battle_comparison,
        create_timeline_from_comments,
        create_category_comparison_chart,
        create_category_radar_chart,
        create_winner_summary_chart,
        create_category_heatmap,
        create_sentiment_bubble_chart
    )
    from components.wordcloud_gen import generate_wordcloud, get_word_frequencies_from_texts
    from components.progress_bar import ProgressBar, create_battle_progress_callback
    from battle_analyzer import BattleAnalyzer
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()


# ============= THEME CONFIGURATION =============
THEMES = {
    "professional_dark": {
        "bg_color": "#0F172A",  # Slate 900
        "main_bg": "#0F172A",
        "sidebar_bg": "#1E293B", # Slate 800
        "card_bg": "rgba(30, 41, 59, 0.7)", # Transparent Slate 800
        "text_primary": "#F8FAFC", # Slate 50
        "text_secondary": "#94A3B8", # Slate 400
        "accent": "#3B82F6", # Blue 500
        "accent_hover": "#2563EB", # Blue 600
        "success": "#10B981", # Emerald 500
        "error": "#EF4444", # Red 500
        "warning": "#F59E0B", # Amber 500
        "border": "rgba(148, 163, 184, 0.1)"
    }
}

CURRENT_THEME = "professional_dark"

def inject_theme():
    """Professional UI Theme Injection"""
    t = THEMES[CURRENT_THEME]
    
    st.markdown(f"""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* GLOBAL RESET */
        .stApp {{
            background-color: {t["main_bg"]};
            font-family: 'Inter', sans-serif;
            color: {t["text_primary"]};
        }}
        
        /* SIDEBAR */
        [data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]};
            border-right: 1px solid {t["border"]};
        }}
        
        /* TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {{
            color: {t["text_primary"]} !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }}
        
        p, label, .stMarkdown {{
            color: {t["text_primary"]};
        }}

        .small-text {{
            font-size: 0.875rem;
            color: {t["text_secondary"]};
        }}
        
        /* CARDS & CONTAINERS */
        .glass-card {{
            background: {t["card_bg"]};
            backdrop-filter: blur(12px);
            border: 1px solid {t["border"]};
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .glass-card:hover {{
            border-color: {t["accent"]};
            transform: translateY(-2px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }}
        
        /* BUTTONS */
        .stButton > button {{
            background: linear-gradient(135deg, {t["accent"]} 0%, {t["accent_hover"]} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            letter-spacing: 0.02em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            font-size: 0.85rem;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
            width: 100%;
        }}
        
        .stButton > button:hover:not(:disabled) {{
            filter: brightness(1.1);
            box-shadow: 0 8px 15px -3px rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }}

        [data-testid="baseButton-secondary"] {{
            background: transparent !important;
            border: 1px solid {t["border"]} !important;
            color: {t["text_secondary"]} !important;
        }}

        [data-testid="baseButton-secondary"]:hover {{
            background: rgba(255,255,255,0.05) !important;
            border-color: {t["accent"]} !important;
            color: {t["accent"]} !important;
        }}
        
        /* INPUTS */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid {t["border"]};
            border-radius: 8px;
            color: {t["text_primary"]};
            padding: 10px 12px;
            transition: all 0.2s;
        }}
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {{
            border-color: {t["accent"]};
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }}
        
        /* METRICS */
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(to right, {t["accent"]}, #60A5FA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.875rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {t["text_secondary"]} !important;
        }}
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            border-bottom: 1px solid {t["border"]};
            padding-bottom: 0px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: nowrap;
            background-color: transparent;
            border: none;
            color: {t["text_secondary"]};
            font-weight: 500;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {t["accent"]};
            border-bottom: 2px solid {t["accent"]};
        }}
        
        /* PROGRESS BAR */
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, {t["accent"]}, #60A5FA);
        }}

        /* HIDE DEFAULT ELEMENTS */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
    </style>
    """, unsafe_allow_html=True)

# ============= SESSION STATE =============
def init_session_state():
    defaults = {
        # Single video analysis
        'single_video_data': None,
        'single_video_sentiment': None,
        # Multi-video analysis
        'multi_video_data': [],
        'multi_video_sentiment': None,
        # Battle mode
        'battle_video1': None,
        'battle_video2': None,
        # Navigation
        'page': 'home',
        'analysis_mode_index': 0,  # 0: Single, 1: Multi
        # Legacy compatibility
        'analyzed_data': None,
        'sentiment_results': None,
        'all_videos': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============= PAGES =============

def page_home():
    """Professional Home Dashboard"""
    
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; margin-bottom: 60px;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; 
                   background: linear-gradient(to right, #F8FAFC, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Insight Analytics
        </h1>
        <p style='font-size: 1.25rem; color: #94A3B8; max_width: 600px; margin: 0 auto;'>
            Advanced YouTube comment analysis powered by AI. 
            Extract sentiments, compare videos, and gain actionable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selection Components
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #3B82F6;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
                <span>🔍</span> Single Source Analysis
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-bottom: 1.5rem; line-height: 1.6;">
                Analyze individual video performance in depth.
                <ul style="margin-top: 12px; padding-left: 20px; color: #CBD5E1; font-size: 0.9rem;">
                    <li>Sentiment Breakdown</li>
                    <li>Keyword Extraction</li>
                    <li>Timeline Trends</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("START SINGLE ANALYSIS", type="primary", use_container_width=True, key="btn_single"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode_index = 0
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #8B5CF6;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
                <span>🚀</span> Competitive Intelligence
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-bottom: 1.5rem; line-height: 1.6;">
                Batch process multiple videos via search.
                <ul style="margin-top: 12px; padding-left: 20px; color: #CBD5E1; font-size: 0.9rem;">
                    <li>Batch Scraping</li>
                    <li>Cross-Video Comparison</li>
                    <li>Market Trend Analysis</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("START BATCH / SEARCH", type="secondary", use_container_width=True, key="btn_multi"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode_index = 1
            st.rerun()

    # Footer Metrics (Minimalist)
    st.markdown("<div style='height: 60px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 32px; color: #64748B; font-size: 0.85rem; padding-top: 24px; border-top: 1px solid rgba(148, 163, 184, 0.1); margin-top: auto;'>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="height: 6px; width: 6px; background-color: #10B981; border-radius: 50%;"></span>
            Model: <b style="color: #94A3B8;">Gemma-4b</b>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="height: 6px; width: 6px; background-color: #10B981; border-radius: 50%;"></span>
            Device: <b style="color: #94A3B8;">Local GPU</b>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="height: 6px; width: 6px; background-color: #10B981; border-radius: 50%;"></span>
            Engine: <b style="color: #94A3B8;">Ollama</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model", "Gemma-4b", delta="Active", delta_color="normal")
    with m2:
        st.metric("Processing", "CUDA Core", delta="Enabled", delta_color="normal")
    with m3:
        st.metric("Engine", "Ollama", delta="Local", delta_color="normal")
    with m4:
        st.metric("Status", "Operational", delta="Ready", delta_color="normal")


def page_analyze():
    """Professional Analysis Page"""
    
    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 1.5rem; color: #F8FAFC;'>Analysis Console</h2>
        <span style='background: rgba(59, 130, 246, 0.1); color: #60A5FA; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;'>
            v2.1
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure analysis_mode is in session_state
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Single Video"
    
    # Mode selection and Main Controls in one clean block
    c_mode, c_input = st.columns([1, 3], gap="medium")
    
    with c_mode:
        st.markdown("<label style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 4px; display: block;'>Operation Mode</label>", unsafe_allow_html=True)
        mode = st.radio(
            "Analysis Mode", 
            ["Single Video", "Multi-Video Batch"], 
            horizontal=False,
            label_visibility="collapsed",
            key="analysis_mode"
        )
    
    # Divider aligned
    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(148, 163, 184, 0.1);'>", unsafe_allow_html=True)
    
    if mode == "Single Video":
        analyze_single_video()
    else:
        analyze_multi_video()


def analyze_single_video():
    # Modern Form Layout
    with st.container():
        col1, col2, col3 = st.columns([5, 2, 2], gap="medium")
        
        with col1:
            url = st.text_input(
                "Video URL", 
                value=st.session_state.get('current_video_url', ''), 
                placeholder="https://youtube.com/watch?v=...",
                help="Paste the full YouTube video URL here"
            )
        
        with col2:
            count = st.number_input(
                "Sample Limit", 
                min_value=10, 
                max_value=10000, 
                value=500, 
                step=50,
                help="Maximum number of comments to fetch"
            )
        
        with col3:
            # Align button with inputs (approximate height adjustment)
            st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True)
            analyze_clicked = st.button(
                "START ANALYSIS", 
                type="primary", 
                use_container_width=True, 
                key="single_analyze_btn"
            )
    
    # Handle analysis in a separate block
    if analyze_clicked and url:
        run_single_analysis(url, count)
    
    # Display results if available (only for single video mode)
    if st.session_state.single_video_data:
        display_single_results()


def run_single_analysis(url, count):
    # Progress bar containers
    progress_container = st.empty()
    progress_bar = ProgressBar(progress_container)
    
    try:
        # Phase 1: Setup (0-5%)
        progress_bar.update(0.02, "Initializing connection...")
        worker = CommentWorker(max_workers=3, max_comments_per_video=count)
        
        # Phase 2: Fetching (5-20%) - animasyonlu ilerleme
        progress_bar.update(0.05, f"Fetching comments (limit: {count})...")
        results = worker.fetch_comments_from_url(url)
        
        if results and results.get('yorumlar'):
            comment_count = len(results['yorumlar'])
            progress_bar.update(0.20, f"SUCCESS: {comment_count} comments retrieved")
            
            st.session_state.single_video_data = results
            st.session_state.single_video_sentiment = None
            
            # Phase 3: Sentiment Analysis (20-95%) - per-comment progress
            comments = [c['metin'] for c in results['yorumlar']]
            analyzer = SentimentAnalyzer()
            
            # Analyze with progress callback
            sentiment_results = []
            total_comments = len(comments)
            
            for i, text in enumerate(comments):
                # Update progress: 20% to 95% range
                progress_pct = 0.20 + (i / total_comments) * 0.75
                progress_bar.update(progress_pct, f"Processing Sentiment: {i+1}/{total_comments}")
                
                result = analyzer.analyze(text)
                sentiment_results.append(result)
            
            st.session_state.single_video_sentiment = sentiment_results
            
            progress_bar.complete(f"ANALYSIS COMPLETE ({total_comments} items)")
            time.sleep(0.5)
            st.rerun()
        else:
            progress_bar.error("No Data Found")
            st.error("Could not retrieve comments. Please check URL privacy settings.")
    except Exception as e:
        progress_bar.error(f"Error: {str(e)[:50]}")
        st.error(f"System Error: {e}")


def analyze_multi_video():
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search Query", placeholder="e.g., 'python tutorial'")
    count = c2.number_input("Video Count", 1, 20, 5)
    per_vid = c3.number_input("Comments/Video", 10, 500, 100)
    
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    if st.button("START BATCH PROCESS", type="primary", use_container_width=True):
        if query:
            run_multi_analysis(query, count, per_vid, None)
    # Display multi-video results only
    if st.session_state.multi_video_data:
        display_multi_video_results()


def run_multi_analysis(query, count, per_vid, keywords_str):
    kw_list = [k.strip() for k in keywords_str.split(',')] if keywords_str else None
    
    # Progress bar container
    progress_container = st.empty()
    status_container = st.empty()
    progress_bar = ProgressBar(progress_container)
    
    try:
        # Phase 1: Search (0-20%)
        progress_bar.update(0.05, f"Searching for '{query}'...")
        status_container.info(f"Searching YouTube for '{query}'...")
        
        scraper = BulkCommentScraper()
        
        # Custom progress callback that updates both status and progress bar
        videos_found = [0]  # Use list for mutable reference
        
        def update_progress(msg):
            # Parse message to estimate progress
            if "video aranıyor" in msg.lower() or "searching" in msg.lower():
                progress_bar.update(0.1, f"Searching videos...")
            elif "url" in msg.lower():
                progress_bar.update(0.15, f"Found {videos_found[0]} video(s)...")
                videos_found[0] += 1
            elif "yorum" in msg.lower() or "comment" in msg.lower():
                # Comment fetching phase (20-60%)
                pct = 0.2 + (videos_found[0] / count) * 0.4
                progress_bar.update(min(0.6, pct), f"Fetching comments ({videos_found[0]}/{count})...")
            status_container.caption(msg)
        
        result = scraper.scrape_and_extract(
            search_query=query,
            video_limit=count,
            max_comments_per_video=per_vid,
            filter_keywords=kw_list,
            parallel_workers=3,
            progress_callback=update_progress
        )
        
        if result and result.get('videos'):
            st.session_state.multi_video_data = result['videos']
            
            # Merge comments for sentiment
            all_comments = []
            for v in result['videos']:
                for c in v['yorumlar']:
                    c['_video_title'] = v.get('baslik', 'Unknown')
                    all_comments.append(c)
            
            # Phase 3: Sentiment Analysis (60-95%)
            progress_bar.update(0.6, f"Analyzing sentiment ({len(all_comments)} comments)...")
            status_container.info(f"Running sentiment analysis on {len(all_comments)} comments...")
            
            comment_texts = [c['metin'] for c in all_comments]
            
            # Analyze with per-comment progress
            analyzer = SentimentAnalyzer()
            sentiment_results = []
            total = len(comment_texts)
            
            for i, text in enumerate(comment_texts):
                pct = 0.6 + (i / total) * 0.35
                if i % 10 == 0:  # Update every 10 comments to reduce overhead
                    progress_bar.update(pct, f"Sentiment: {i+1}/{total}")
                sentiment_results.append(analyzer.analyze(text))
            
            st.session_state.multi_video_sentiment = sentiment_results
            
            progress_bar.complete(f"Complete! {len(result['videos'])} videos, {len(all_comments)} comments")
            status_container.empty()
            
            time.sleep(0.5)
            st.rerun()
        else:
            progress_bar.error("No results found")
            st.error("No results found.")
    except Exception as e:
        progress_bar.error(f"Error: {str(e)[:50]}")
        st.error(f"Critical Error: {e}")


def display_single_results():
    data = st.session_state.single_video_data
    res = st.session_state.single_video_sentiment
    comments = data.get('yorumlar', [])
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Video Header
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 32px;'>
        <h3 style='font-size: 1.5rem; color: #F8FAFC; margin-bottom: 8px;'>{data.get('baslik', 'Video Title')}</h3>
        <span style='color: #64748B; font-size: 0.9rem; background: rgba(30, 41, 59, 0.5); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(148, 163, 184, 0.1);'>
            {data.get('kanal', 'Unknown Channel')} &bull; {data.get('sure', '0')}s
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    
    sentiment_score = 0
    if res:
        stats = SentimentAnalyzer().get_summary_stats(res)
        sentiment_score = stats['sentiment_score']
    
    def metric_card(label, value, color="#3B82F6"):
        return f"""
        <div style='background: rgba(30, 41, 59, 0.4); border-left: 3px solid {color}; padding: 16px; border-radius: 8px;'>
            <div style='color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;'>{label}</div>
            <div style='color: #F8FAFC; font-size: 1.5rem; font-weight: 700; margin-top: 4px;'>{value}</div>
        </div>
        """

    with c1: st.markdown(metric_card("Total Comments", len(comments), "#3B82F6"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Total Likes", f"{sum(c.get('begeni', 0) for c in comments):,}", "#8B5CF6"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Video Views", f"{data.get('goruntulenme', 0):,}", "#10B981"), unsafe_allow_html=True)
    
    # Dynamic color for sentiment
    s_color = "#10B981" if sentiment_score > 0 else "#EF4444" if sentiment_score < 0 else "#94A3B8"
    with c4: st.markdown(metric_card("Sentiment Score", f"{sentiment_score:.2f}", s_color), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #F8FAFC; margin-bottom: 20px; border-left: 4px solid #F59E0B; padding-left: 12px;'>Visual Insights</h3>", unsafe_allow_html=True)
    display_tabs(comments, res, data.get('baslik', ''))


def display_multi_video_results():
    videos = st.session_state.multi_video_data
    all_comments = []
    for v in videos:
        all_comments.extend(v.get('yorumlar', []))
    
    st.divider()
    st.subheader(f"Batch Results ({len(videos)} Videos)")
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comments", len(all_comments))
    c2.metric("Total Likes", f"{sum(c.get('begeni', 0) for c in all_comments):,}")
    c3.metric("Videos Scanned", len(videos))
    
    if st.session_state.multi_video_sentiment:
        stats = SentimentAnalyzer().get_summary_stats(st.session_state.multi_video_sentiment)
        c4.metric("Overall Sentiment", f"{stats['sentiment_score']:.2f}")

    with st.expander("SOURCE LIST"):
        for i, v in enumerate(videos, 1):
            st.write(f"**{i}. {v.get('baslik')}** - {len(v['yorumlar'])} comments")

    display_tabs(all_comments, st.session_state.multi_video_sentiment, "Multi-Video Analysis")


def display_tabs(comments, sentiment_results, title_context):
    tabs = st.tabs(["DASHBOARD", "TIMELINE", "DATA FEED", "WORD CLOUD", "AI SUMMARY"])
    
    with tabs[0]:
        if sentiment_results:
            c1, c2 = st.columns(2)
            dist = SentimentAnalyzer().get_sentiment_distribution(sentiment_results)
            c1.plotly_chart(create_sentiment_pie_chart(dist['positive'], dist['negative'], dist['neutral']), use_container_width=True)
            
            stats = SentimentAnalyzer().get_summary_stats(sentiment_results)
            c2.plotly_chart(create_engagement_gauge(stats['sentiment_score']), use_container_width=True)
        else:
            st.info("No sentiment data available.")

    with tabs[1]:
        try:
            st.plotly_chart(create_timeline_from_comments(comments, sentiment_results), use_container_width=True)
        except:
            st.caption("Insufficient time data.")

    with tabs[2]:
        # --- DATA FEED REDESIGN ---
        
        # 1. Pagination Controls (Compact)
        col_controls, col_stats = st.columns([2, 3])
        
        with col_controls:
            c_page, c_limit = st.columns(2)
            # Fixed: Use proper labels with label_visibility, and number_input for custom values
            per_page = c_limit.number_input("Per Page", min_value=10, max_value=500, value=20, step=10, label_visibility="collapsed", key="pp_select")
            
            total_pages = max(1, (len(comments)-1)//per_page + 1)
            page = c_page.number_input("Page", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed", key="page_input")
            
            st.caption(f"Page {page} of {total_pages} | Total: {len(comments)}")

        start = (page-1)*per_page
        end = start + per_page
        current_batch = comments[start:end]

        # 2. Mini Bar Chart for Current Page (User Request)
        with col_stats:
            # Calculate sentiment for current page if available
            if sentiment_results:
                batch_indices = range(start, min(end, len(sentiment_results)))
                batch_sentiments = [sentiment_results[i] for i in batch_indices]
                dist = SentimentAnalyzer().get_sentiment_distribution(batch_sentiments)
                
                # Mini stacked bar
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Bar(
                    x=[dist['positive']], y=[''], orientation='h', 
                    marker_color='#10B981', name='Pos', hoverinfo='x'
                ))
                fig_mini.add_trace(go.Bar(
                    x=[dist['negative']], y=[''], orientation='h', 
                    marker_color='#EF4444', name='Neg', hoverinfo='x'
                ))
                fig_mini.add_trace(go.Bar(
                    x=[dist['neutral']], y=[''], orientation='h', 
                    marker_color='#94A3B8', name='Neu', hoverinfo='x'
                ))
                fig_mini.update_layout(
                    barmode='stack', 
                    height=40, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False)
                )
                st.plotly_chart(fig_mini, key="mini_chart", config={'displayModeBar': False})
                st.caption("Page Sentiment Distribution")

        st.markdown("---")

        # 3. Comments List (Styled Cards)
        for c in current_batch:
             # Try to find corresponding sentiment if available
            sentiment_color = "#94A3B8" # Default gray
            score = 0
            if sentiment_results:
                # Assuming simple mapping by index, but comments list might be filtered/sorted in future.
                # Since comments arg is passed directly, index matching should work for now.
                # Ideally, comments should have ID or attached sentiment.
                # For safety, let's just show raw text styled.
                pass

            st.markdown(f"""
            <div class="glass-card" style="padding: 16px; margin-bottom: 12px; border-left: 4px solid #3B82F6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #60A5FA; font-weight: 600; font-size: 0.9rem;">{c.get('yazar', 'User')}</span>
                    <span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">
                        ❤️ {c.get('begeni', 0)}
                    </span>
                </div>
                <div style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">
                    {c.get('metin', '')}
                </div>
                {f'<div style="margin-top:8px; font-size:0.75rem; color:#64748B;">📺 {c["_video_title"]}</div>' if '_video_title' in c else ''}
            </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.subheader("Keyword Frequency")
        texts = [c.get('metin', '') for c in comments]
        freqs = get_word_frequencies_from_texts(texts)
        if freqs:
            img = generate_wordcloud(word_frequencies=freqs, width=1200, height=600)
            if img:
                st.image(img, use_container_width=True)
        else:
            st.info("Insufficient data.")

    with tabs[4]:
        st.markdown("### AI Executive Summary")
        st.caption("Model: **gemma3:4b** via Ollama Local")
        
        # --- FIX: Session State Persistence ---
        summary_key = f"summary_{hash(title_context)}"
        
        if summary_key not in st.session_state:
            st.session_state[summary_key] = None
            
        if st.button("Generate Summary", type="primary", key="btn_gen_summary"):
            with st.spinner("Processing..."):
                try:
                    ollama = OllamaLLM(model_name="gemma3:4b")
                    if not ollama.check_connection():
                        st.error("Ollama connection failed.")
                    else:
                        sample = [c.get('metin', '') for c in comments[:100]]
                        res = ollama.summarize_comments(sample, title_context)
                        st.session_state[summary_key] = res.summary
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Display if exists
        if st.session_state[summary_key]:
            st.markdown(f"""
            <div class="glass-card" style="border: 1px solid #10B981;">
                {st.session_state[summary_key]}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Clear Summary", key="btn_clr_summary"):
                st.session_state[summary_key] = None
                st.rerun()


def page_battle():
    st.title("Competitive Battle Mode")
    
    # --- Video URL Girişleri ---
    st.markdown("### Source Configuration")
    c1, c2 = st.columns(2)
    u1 = c1.text_input("Competitor A (URL)", key="battle_url1", placeholder="https://youtube.com/watch?v=...")
    u2 = c2.text_input("Competitor B (URL)", key="battle_url2", placeholder="https://youtube.com/watch?v=...")
    
    # --- Kategori Tanımlama ---
    st.markdown("### Evaluation Criteria")

    if 'battle_categories' not in st.session_state:
        st.session_state.battle_categories = [{"name": "", "desc": ""}]
    
    categories_to_remove = []
    for i, cat in enumerate(st.session_state.battle_categories):
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            cat['name'] = st.text_input(f"Category", value=cat['name'], key=f"cat_name_{i}", placeholder="e.g. Positive Feedback")
        with col2:
            cat['desc'] = st.text_input(f"Description / Rules", value=cat['desc'], key=f"cat_desc_{i}", placeholder="What counts as this category?")
        with col3:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("x", key=f"remove_cat_{i}"):
                categories_to_remove.append(i)
    
    for idx in sorted(categories_to_remove, reverse=True):
        st.session_state.battle_categories.pop(idx)
        st.rerun()
    
    if st.button("+ Add Category"):
        st.session_state.battle_categories.append({"name": "", "desc": ""})
        st.rerun()
    
    st.markdown("---")
    
    max_comments = st.number_input("Analysis Depth (Comments per Video)", min_value=10, max_value=10000, value=50, step=10)
    
    if st.button("INITIATE BATTLE COMPARISON", type="primary", use_container_width=True):
        if u1 and u2:
            valid_categories = {c['name']: c['desc'] for c in st.session_state.battle_categories if c['name'] and c['desc']}
            
            if len(valid_categories) < 1:
                st.error("Define at least 1 category.")
                return
            
            progress_container = st.empty()
            status_container = st.empty()
            
            try:
                status_container.info("Fetching source data...")
                worker = CommentWorker(max_workers=2, max_comments_per_video=max_comments)
                
                v1_data = worker.fetch_comments_from_url(u1)
                v2_data = worker.fetch_comments_from_url(u2)
                
                if not v1_data or not v2_data:
                    st.error("Could not fetch data.")
                    return
                
                v1_comments = [c['metin'] for c in v1_data.get('yorumlar', [])]
                v2_comments = [c['metin'] for c in v2_data.get('yorumlar', [])]
                
                status_container.info("Running AI Classification...")
                progress_bar = ProgressBar(progress_container)
                
                analyzer = BattleAnalyzer(model_name="gemma3:4b")
                
                if not analyzer.check_connection():
                    st.error("Ollama connection failed.")
                    return
                
                total_cats = len(valid_categories)
                progress_callback = create_battle_progress_callback(progress_bar, total_cats, max_comments)
                
                result = analyzer.compare_videos(
                    v1_comments, v2_comments,
                    v1_data.get('baslik', 'Video 1'),
                    v2_data.get('baslik', 'Video 2'),
                    valid_categories,
                    max_comments_per_video=max_comments,
                    progress_callback=progress_callback
                )
                
                progress_bar.complete("Analysis Complete")
                
                st.session_state.battle_result = result
                st.session_state.battle_video1 = v1_data
                st.session_state.battle_video2 = v2_data
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter both URLs.")
    
    if 'battle_result' in st.session_state and st.session_state.battle_result:
        result = st.session_state.battle_result
        v1 = st.session_state.battle_video1
        v2 = st.session_state.battle_video2
        
        st.markdown("---")
        st.subheader("Comparison Results")
        
        if result.winner != "Berabere":
            st.success(f"**Leader: {result.winner}**")
        
        c1, c2 = st.columns(2)
        c1.metric(v1.get('baslik', 'Video 1')[:25], f"{result.video1_total_comments} comments")
        c2.metric(v2.get('baslik', 'Video 2')[:25], f"{result.video2_total_comments} comments")
        
        st.markdown("### Category Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Side-by-Side", "Radar View", "Win Share", "Heatmap"])
        
        with tab1:
            fig = create_category_comparison_chart(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = create_category_radar_chart(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = create_winner_summary_chart(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            fig = create_category_heatmap(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Category details
        st.markdown("### Detailed Breakdown")
        for cat_name, cat_data in result.categories.items():
            with st.expander(f"**{cat_name}** - V1: {cat_data['v1_percent']:.1f}% | V2: {cat_data['v2_percent']:.1f}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{v1.get('baslik', 'Video 1')[:30]}**")
                    for sample in cat_data['v1_samples'][:2]:
                        st.markdown(f"> _{sample[:100]}..._")
                with col2:
                    st.markdown(f"**{v2.get('baslik', 'Video 2')[:30]}**")
                    for sample in cat_data['v2_samples'][:2]:
                        st.markdown(f"> _{sample[:100]}..._")
        
        st.markdown("### Export Data")
        tab_v1, tab_v2 = st.tabs([f"{v1.get('baslik', 'Video 1')[:20]}", f"{v2.get('baslik', 'Video 2')[:20]}"])
        
        with tab_v1:
            if result.v1_classifications:
                df_v1 = pd.DataFrame(result.v1_classifications)
                st.dataframe(df_v1, use_container_width=True, height=300)
        
        with tab_v2:
            if result.v2_classifications:
                df_v2 = pd.DataFrame(result.v2_classifications)
                st.dataframe(df_v2, use_container_width=True, height=300)
        
        # Reset button
        if st.button("🔄 Yeni Karşılaştırma"):
            st.session_state.battle_result = None
            st.session_state.battle_video1 = None
            st.session_state.battle_video2 = None
            st.rerun()


def page_stats():
    st.title("📈 İstatistikler")
    
    # Check for any data
    has_single = st.session_state.single_video_data is not None
    has_multi = st.session_state.multi_video_data and len(st.session_state.multi_video_data) > 0
    
    if not has_single and not has_multi:
        st.info("Henüz analiz verisi yok. Önce bir video analiz edin.")
        return
    
    # Select data source
    if has_single and has_multi:
        source = st.radio("Veri Kaynağı", ["Tek Video", "Çoklu Video"], horizontal=True)
        use_multi = source == "Çoklu Video"
    else:
        use_multi = has_multi
    
    if use_multi:
        videos = st.session_state.multi_video_data
        all_comments = []
        for v in videos:
            for c in v.get('yorumlar', []):
                c['_video_title'] = v.get('baslik', 'Unknown')
                all_comments.append(c)
        sentiment = st.session_state.multi_video_sentiment
        title_text = f"{len(videos)} Video Analizi"
    else:
        data = st.session_state.single_video_data
        all_comments = data.get('yorumlar', [])
        sentiment = st.session_state.single_video_sentiment
        title_text = data.get('baslik', 'Tek Video')[:40]
    
    st.subheader(f"📊 {title_text}")
    st.divider()
    
    # --- GENEL METRİKLER ---
    st.markdown("### 📌 Genel Metrikler")
    c1, c2, c3, c4 = st.columns(4)
    
    total_comments = len(all_comments)
    total_likes = sum(c.get('begeni', 0) for c in all_comments)
    avg_likes = total_likes / total_comments if total_comments > 0 else 0
    
    c1.metric("Toplam Yorum", f"{total_comments:,}")
    c2.metric("Toplam Beğeni", f"{total_likes:,}")
    c3.metric("Ortalama Beğeni/Yorum", f"{avg_likes:.1f}")
    
    if sentiment:
        stats = SentimentAnalyzer().get_summary_stats(sentiment)
        c4.metric("Duygu Skoru", f"{stats['sentiment_score']:.2f}")
    else:
        c4.metric("Duygu Skoru", "N/A")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- DUYGU DAĞILIMI ---
    st.markdown("### 🎭 Duygu Dağılımı")
    if sentiment:
        stats = SentimentAnalyzer().get_summary_stats(sentiment)
        col1, col2 = st.columns([1, 1])
        
        # Tabs for Pie vs Bubble chart
        tab_pie, tab_bubble = st.tabs(["🥧 Pasta Grafiği", "🫧 Balon Grafiği"])
        
        with tab_pie:
            col1, col2 = st.columns([1, 1])
            with col1:
                fig = create_sentiment_pie_chart(
                    stats.get('positive_count', 0),
                    stats.get('negative_count', 0),
                    stats.get('neutral_count', 0)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Detaylar")
                total = stats.get('total_analyzed', 1)
                pos_pct = (stats.get('positive_count', 0) / total * 100) if total > 0 else 0
                neg_pct = (stats.get('negative_count', 0) / total * 100) if total > 0 else 0
                neu_pct = (stats.get('neutral_count', 0) / total * 100) if total > 0 else 0
                
                st.markdown(f"- **Pozitif:** {stats.get('positive_count', 0)} yorum ({pos_pct:.1f}%)")
                st.markdown(f"- **Negatif:** {stats.get('negative_count', 0)} yorum ({neg_pct:.1f}%)")
                if 'neutral_count' in stats:
                    st.markdown(f"- **Nötr:** {stats.get('neutral_count', 0)} yorum ({neu_pct:.1f}%)")
                st.markdown(f"- **Ortalama Güven:** {stats.get('average_confidence', 0):.2f}")
        
        with tab_bubble:
            fig = create_sentiment_bubble_chart(
                stats.get('positive_count', 0),
                stats.get('negative_count', 0),
                stats.get('neutral_count', 0)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Duygu analizi verisi yok.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- VIDEO BAZLI DAĞILIM (Çoklu video için) ---
    if use_multi and len(videos) > 1:
        st.markdown("### 📹 Video Bazlı Yorum Dağılımı")
        video_data = {v.get('baslik', f'Video {i}')[:25]: len(v.get('yorumlar', [])) for i, v in enumerate(videos)}
        
        fig = create_keyword_bar_chart(video_data, top_n=len(video_data))
        st.plotly_chart(fig, use_container_width=True)
    
    # --- EN POPÜLER YORUMLAR ---
    st.markdown("### ⭐ En Çok Beğenilen Yorumlar")
    sorted_comments = sorted(all_comments, key=lambda x: x.get('begeni', 0), reverse=True)[:5]
    
    for i, c in enumerate(sorted_comments, 1):
        with st.expander(f"**#{i}** - {c.get('begeni', 0)} beğeni | {c.get('yazar', 'Anonim')}"):
            st.write(c.get('metin', ''))
            if '_video_title' in c:
                st.caption(f"📺 {c['_video_title']}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- KELİME FREKANSLARI ---
    st.markdown("### 🔤 En Sık Kullanılan Kelimeler")
    texts = [c.get('metin', '') for c in all_comments]
    freqs = get_word_frequencies_from_texts(texts, top_n=20)
    
    if freqs:
        fig = create_keyword_bar_chart(freqs, top_n=15)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Yeterli kelime verisi yok.")





def page_settings():
    st.title("Ayarlar")
    st.markdown("### AI Model Yapılandırması")
    st.success("✅ **Ollama (gemma3:4b)** aktif.")
    st.info("Model değiştirmek için `ollama pull <model_adi>` kullanın ve kodu güncelleyin.")


# ============= MAIN =============
def main():
    inject_theme()
    init_session_state()
    
    with st.sidebar:
        st.title("YCA Studio")
        
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        
        if st.button("Ana Sayfa", use_container_width=True): st.session_state.page = 'home'; st.rerun()
        if st.button("Video Analiz", use_container_width=True): st.session_state.page = 'analyze'; st.rerun()
        if st.button("Battle Mode", use_container_width=True): st.session_state.page = 'battle'; st.rerun()
        if st.button("İstatistikler", use_container_width=True): st.session_state.page = 'stats'; st.rerun()
        if st.button("Ayarlar", use_container_width=True): st.session_state.page = 'settings'; st.rerun()
        
        st.markdown("<div style='flex-grow: 1'></div>", unsafe_allow_html=True)
        st.divider()
        st.caption("Powered by **Ollama**")
        st.caption("Model: gemma3:4b")

    if st.session_state.page == 'home': page_home()
    elif st.session_state.page == 'analyze': page_analyze()
    elif st.session_state.page == 'battle': page_battle()
    elif st.session_state.page == 'stats': page_stats()
    elif st.session_state.page == 'settings': page_settings()

if __name__ == "__main__":
    main()
