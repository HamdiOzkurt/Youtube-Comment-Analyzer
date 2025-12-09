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
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

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
THEME = {
    "bg_color": "#0F172A",  # Slate 900
    "main_bg": "#0F172A",
    "sidebar_bg": "#1E293B", # Slate 800
    "card_bg": "rgba(30, 41, 59, 0.7)", # Transparent Slate 800
    "card_bg_solid": "#1E293B",
    "text_primary": "#F8FAFC", # Slate 50
    "text_secondary": "#94A3B8", # Slate 400
    "text_muted": "#64748B", # Slate 500
    "accent": "#3B82F6", # Blue 500
    "accent_secondary": "#8B5CF6", # Purple 500
    "accent_hover": "#2563EB", # Blue 600
    "success": "#10B981", # Emerald 500
    "error": "#EF4444", # Red 500
    "warning": "#F59E0B", # Amber 500
    "border": "rgba(148, 163, 184, 0.1)",
    "input_bg": "rgba(30, 41, 59, 0.5)",
    "hover_bg": "rgba(59, 130, 246, 0.1)"
}


def inject_theme():
    """Professional UI Theme Injection"""
    t = THEME
    
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
        
        /* Remove top padding/margin from main content */
        .block-container {{
            padding-top: 1rem !important;
        }}
        
        [data-testid="stHeader"] {{
            display: none !important;
        }}
        
        /* SIDEBAR TOGGLE - Keep visible */
        [data-testid="collapsedControl"] {{
            display: flex !important;
        }}
        
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
    
    # ============ HERO SECTION ============
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 50px 0;'>
        <div style='margin-bottom: 16px;'>
            <span style='background: linear-gradient(135deg, #3B82F6, #8B5CF6); padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; color: white; font-weight: 500;'>
                AI-Powered Analytics
            </span>
        </div>
        <h1 style='font-size: 3rem; font-weight: 800; margin: 0 0 16px 0; 
                   background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            YouTube Comment Analyzer
        </h1>
        <p style='font-size: 1.1rem; color: #64748B; max-width: 500px; margin: 0 auto; line-height: 1.6;'>
            Extract sentiment insights, discover trends, and understand your audience with local AI processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ FEATURE BADGES ============
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 24px; margin-bottom: 48px; flex-wrap: wrap;'>
        <div style='display: flex; align-items: center; gap: 8px; color: #64748B; font-size: 0.85rem;'>
            <span style='color: #10B981; font-size: 1.1rem;'>●</span> Local Processing
        </div>
        <div style='display: flex; align-items: center; gap: 8px; color: #64748B; font-size: 0.85rem;'>
            <span style='color: #3B82F6; font-size: 1.1rem;'>●</span> GPU Accelerated
        </div>
        <div style='display: flex; align-items: center; gap: 8px; color: #64748B; font-size: 0.85rem;'>
            <span style='color: #8B5CF6; font-size: 1.1rem;'>●</span> No API Keys
        </div>
        <div style='display: flex; align-items: center; gap: 8px; color: #64748B; font-size: 0.85rem;'>
            <span style='color: #F59E0B; font-size: 1.1rem;'>●</span> Privacy First
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ MAIN ACTION CARDS ============
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #3B82F6; min-height: 200px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="background: linear-gradient(135deg, #3B82F6, #1D4ED8); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-weight: 700; font-size: 1.2rem;">S</span>
                </div>
                <div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">Single Video</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Deep dive analysis</div>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin-bottom: 16px;">
                Analyze comments from a single YouTube video. Get sentiment distribution, keyword extraction, and AI-generated insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("ANALYZE VIDEO", type="primary", use_container_width=True, key="btn_single"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode = "Single Video"
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #8B5CF6; min-height: 200px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="background: linear-gradient(135deg, #8B5CF6, #6D28D9); width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-weight: 700; font-size: 1.2rem;">B</span>
                </div>
                <div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">Batch Search</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Multi-video comparison</div>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin-bottom: 16px;">
                Search YouTube and analyze multiple videos at once. Compare sentiment across competitors and discover market trends.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("BATCH SEARCH", type="secondary", use_container_width=True, key="btn_multi"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode = "Multi-Video Batch"
            st.rerun()
    
    # ============ QUICK START GUIDE ============
    st.markdown("<div style='height: 48px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 24px;'>
        <span style='font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 1px;'>How It Works</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(59, 130, 246, 0.1); width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto;'>
                <span style='color: #3B82F6; font-weight: 700; font-size: 1.2rem;'>1</span>
            </div>
            <div style='font-weight: 600; color: #F8FAFC; margin-bottom: 4px;'>Paste URL</div>
            <div style='font-size: 0.85rem; color: #64748B;'>Enter any YouTube video link</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(139, 92, 246, 0.1); width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto;'>
                <span style='color: #8B5CF6; font-weight: 700; font-size: 1.2rem;'>2</span>
            </div>
            <div style='font-weight: 600; color: #F8FAFC; margin-bottom: 4px;'>Process</div>
            <div style='font-size: 0.85rem; color: #64748B;'>AI analyzes comments locally</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(16, 185, 129, 0.1); width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto;'>
                <span style='color: #10B981; font-weight: 700; font-size: 1.2rem;'>3</span>
            </div>
            <div style='font-weight: 600; color: #F8FAFC; margin-bottom: 4px;'>Insights</div>
            <div style='font-size: 0.85rem; color: #64748B;'>View charts, trends & reports</div>
        </div>
        """, unsafe_allow_html=True)




def page_analyze():
    """Professional Analysis Page"""
    
    st.markdown("""
    <div style='margin-bottom: 28px;'>
        <h2 style='font-size: 1.8rem; margin: 0; color: #F8FAFC;'>Analysis Console</h2>
        <p style='color: #64748B; font-size: 0.9rem; margin-top: 4px;'>Extract insights from YouTube video comments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure analysis_mode is in session_state
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Single Video"
    
    # --- STEP 1: Mode Selection ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='background: #3B82F6; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>1</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>Select Mode</span>
    </div>
    """, unsafe_allow_html=True)
    
    mode = st.radio(
        "Analysis Mode", 
        ["Single Video", "Multi-Video Batch"], 
        horizontal=True,
        label_visibility="collapsed",
        key="analysis_mode"
    )
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    if mode == "Single Video":
        analyze_single_video()
    else:
        analyze_multi_video()


def analyze_single_video():
    # --- STEP 2: Input Configuration ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='background: #8B5CF6; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>2</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>Input Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([4, 1], gap="medium")
        
        with col1:
            url = st.text_input(
                "Video URL", 
                value=st.session_state.get('current_video_url', ''), 
                placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed"
            )
        
        with col2:
            count = st.number_input(
                "Limit", 
                min_value=10, 
                max_value=10000, 
                value=500, 
                step=50,
                label_visibility="collapsed"
            )
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("START ANALYSIS", type="primary", use_container_width=True, key="single_analyze_btn")
    
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
                        
                        # Calculate sentiment distribution for summary context
                        sentiment_dist = None
                        if sentiment_results:
                            analyzer = SentimentAnalyzer()
                            dist = analyzer.get_sentiment_distribution(sentiment_results)
                            total = dist.get('positive', 0) + dist.get('negative', 0) + dist.get('neutral', 0)
                            if total > 0:
                                sentiment_dist = {
                                    'positive': int(dist.get('positive', 0) / total * 100),
                                    'negative': int(dist.get('negative', 0) / total * 100),
                                    'neutral': int(dist.get('neutral', 0) / total * 100)
                                }
                        
                        res = ollama.summarize_comments(sample, title_context, sentiment_distribution=sentiment_dist)
                        st.session_state[summary_key] = res.summary
                except Exception as e:
                    st.error(f"Error: {e}")
        
        
        # Display if exists
        if st.session_state[summary_key]:
            # Use container with custom CSS
            with st.container():
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); 
                            border-left: 4px solid #10B981; 
                            border-radius: 8px; 
                            padding: 20px; 
                            margin: 16px 0;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                        <span style="font-size: 1.2rem;">📝</span>
                        <span style="font-size: 1.1rem; font-weight: 600; color: #10B981;">AI Generated Summary</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(st.session_state[summary_key])
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            if st.button("Clear Summary", key="btn_clr_summary"):
                st.session_state[summary_key] = None
                st.rerun()


def page_battle():
    # Clean header
    st.markdown("""
    <div style='margin-bottom: 32px;'>
        <h2 style='font-size: 1.8rem; margin: 0; color: #F8FAFC;'>Competitive Battle</h2>
        <p style='color: #64748B; font-size: 0.9rem; margin-top: 4px;'>Compare two videos using AI-powered evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- STEP 1: Video URL Inputs ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='background: #3B82F6; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>1</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>Source Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    u1 = c1.text_input("Competitor A", key="battle_url1", placeholder="https://youtube.com/watch?v=...")
    u2 = c2.text_input("Competitor B", key="battle_url2", placeholder="https://youtube.com/watch?v=...")
    
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    
    # --- STEP 2: Evaluation Criteria ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
        <span style='background: #8B5CF6; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>2</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>Evaluation Criteria</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Define categories to classify and compare comments")

    if 'battle_categories' not in st.session_state:
        st.session_state.battle_categories = [{"name": "", "desc": ""}]
    
    # Use expander for category details to reduce visual clutter
    with st.expander("Manage Criteria", expanded=len(st.session_state.battle_categories) <= 2):
        categories_to_remove = []
        for i, cat in enumerate(st.session_state.battle_categories):
            col1, col2, col3 = st.columns([2, 5, 1])
            with col1:
                cat['name'] = st.text_input(f"Category #{i+1}", value=cat['name'], key=f"cat_name_{i}", placeholder="e.g. Features")
            with col2:
                cat['desc'] = st.text_input(f"Criteria", value=cat['desc'], key=f"cat_desc_{i}", placeholder="Brief description")
            with col3:
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"remove_cat_{i}"):
                    categories_to_remove.append(i)
        
        for idx in sorted(categories_to_remove, reverse=True):
            st.session_state.battle_categories.pop(idx)
            st.rerun()
        
        if st.button("+ Add Criterion"):
            st.session_state.battle_categories.append({"name": "", "desc": ""})
            st.rerun()
    
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Compact controls
    col_depth, col_btn = st.columns([2, 3])
    with col_depth:
        max_comments = st.number_input("Sample Size", min_value=10, max_value=500, value=50, step=10, help="Comments per video")
    with col_btn:
        st.markdown("<div style='height: 29px'></div>", unsafe_allow_html=True)
        start_battle = st.button("START BATTLE", type="primary", use_container_width=True)
    
    if start_battle:
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
    st.title("📈 İleri Düzey Metin Madenciliği")
    
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
    
    st.markdown(f"""
    <div style='margin-bottom: 24px;'>
        <h2 style='font-size: 1.6rem; margin: 0; color: #F8FAFC;'>{title_text}</h2>
        <p style='color: #64748B; font-size: 0.85rem; margin-top: 4px;'>Toplam {len(all_comments)} yorum analiz ediliyor</p>
    </div>
    """, unsafe_allow_html=True)
    
    import numpy as np
    import re
    from collections import Counter
    
    texts = [c.get('metin', '') for c in all_comments]
    comment_lengths = [len(t) for t in texts]
    likes = [c.get('begeni', 0) for c in all_comments]
    
    # === GENEL METRİKLER ===
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 16px 0 12px 0;'>
        <span style='background: #3B82F6; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>📊</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #F8FAFC;'>Genel Metrikler</span>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam Yorum", f"{len(all_comments):,}")
    m2.metric("Toplam Beğeni", f"{sum(likes):,}")
    m3.metric("Ort. Uzunluk", f"{np.mean(comment_lengths):.0f} char")
    if sentiment:
        stats = SentimentAnalyzer().get_summary_stats(sentiment)
        m4.metric("Duygu Skoru", f"{stats['sentiment_score']:.2f}")
    else:
        m4.metric("Duygu Skoru", "N/A")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 1. Bİ-GRAM ANALİZİ ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #EC4899, #F59E0B); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>1️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #EC4899, #F59E0B); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Bi-Gram Analizi (İkili Kelime Öbekleri)</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Yorumlarda yan yana en çok kullanılan kelime çiftleri")
    
    try:
        # Clean texts for vectorization
        cleaned_texts = [t.lower() for t in texts if len(t) > 10]
        
        if cleaned_texts:
            vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=15, stop_words=None)
            bigram_matrix = vectorizer.fit_transform(cleaned_texts)
            bigram_counts = bigram_matrix.sum(axis=0).A1
            bigram_names = vectorizer.get_feature_names_out()
            
            # Sort by frequency
            sorted_indices = np.argsort(bigram_counts)[::-1]
            top_bigrams = [(bigram_names[i], bigram_counts[i]) for i in sorted_indices[:12]]
            
            if top_bigrams:
                bigrams = [b[0] for b in top_bigrams]
                counts = [b[1] for b in top_bigrams]
                
                # Plasma gradient colors
                colors = px.colors.sequential.Plasma[:len(bigrams)]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=bigrams[::-1],
                    x=counts[::-1],
                    orientation='h',
                    marker=dict(
                        color=colors[::-1],
                        line=dict(width=1, color='rgba(255,255,255,0.5)')
                    ),
                    text=[f" {c}" for c in counts[::-1]],
                    textposition='outside',
                    textfont=dict(size=11, color='#E2E8F0'),
                    hovertemplate='<b>%{y}</b><br>Kullanım: %{x}<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Kullanım Sayısı",
                    yaxis_title="",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0', size=12),
                    height=420,
                    margin=dict(l=20, r=60, t=10, b=40),
                    hoverlabel=dict(bgcolor="#1E293B", bordercolor="#EC4899")
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', showgrid=True, zeroline=False)
                fig.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Yeterli bi-gram verisi yok.")
        else:
            st.info("Yeterli metin verisi yok.")
    except Exception as e:
        st.warning(f"Bi-gram analizi yapılamadı: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 2. ETKİLEŞİM MATRİSİ (Bubble Chart) ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #10B981, #06B6D4); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>2️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #10B981, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Etkileşim Matrisi (Duygu × Beğeni × Uzunluk)</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Baloncuk boyutu = yorum uzunluğu, X = duygu skoru, Y = beğeni sayısı")
    
    if sentiment and len(all_comments) > 0:
        bubble_data = []
        for i, c in enumerate(all_comments):
            if i < len(sentiment):
                sent = sentiment[i]
                # Convert sentiment to score (-1 to 1)
                if sent.label == 'positive':
                    score = sent.score
                elif sent.label == 'negative':
                    score = -sent.score
                else:
                    score = 0
                
                bubble_data.append({
                    'sentiment_score': score,
                    'likes': c.get('begeni', 0),
                    'length': len(c.get('metin', '')),
                    'label': sent.label.capitalize(),
                    'author': c.get('yazar', 'Anonim')[:20],
                    'text': c.get('metin', '')[:80]
                })
        
        if bubble_data:
            import pandas as pd
            df_bubble = pd.DataFrame(bubble_data)
            
            # Color mapping
            color_map = {'Positive': '#10B981', 'Neutral': '#3B82F6', 'Negative': '#EF4444'}
            
            fig = go.Figure()
            
            for label in ['Positive', 'Neutral', 'Negative']:
                df_subset = df_bubble[df_bubble['label'] == label]
                if len(df_subset) > 0:
                    fig.add_trace(go.Scatter(
                        x=df_subset['sentiment_score'],
                        y=df_subset['likes'],
                        mode='markers',
                        name=f"{'🟢' if label == 'Positive' else '🔵' if label == 'Neutral' else '🔴'} {label}",
                        marker=dict(
                            size=np.clip(df_subset['length'] / 10, 8, 50),
                            color=color_map[label],
                            opacity=0.7,
                            line=dict(width=1, color='rgba(255,255,255,0.5)')
                        ),
                        customdata=df_subset[['author', 'text', 'length']].values,
                        hovertemplate='<b>👤 %{customdata[0]}</b><br>' +
                                      'Duygu Skoru: %{x:.2f}<br>' +
                                      'Beğeni: %{y}<br>' +
                                      'Uzunluk: %{customdata[2]} char<br>' +
                                      '<i>%{customdata[1]}...</i><extra></extra>'
                    ))
            
            fig.update_layout(
                xaxis_title="Duygu Skoru (-1 = Negatif, +1 = Pozitif)",
                yaxis_title="Beğeni Sayısı",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', size=12),
                height=450,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'
                ),
                margin=dict(l=20, r=20, t=50, b=40),
                hoverlabel=dict(bgcolor="#1E293B", bordercolor="#64748B")
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', showgrid=True, zeroline=True, zerolinecolor='rgba(255,255,255,0.2)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.08)', showgrid=True, zeroline=False)
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Duygu analizi verisi gerekli. Önce bir video analiz edin.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 3. EN ETKİLİ YORUMCULAR ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #8B5CF6, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>3️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #8B5CF6, #6366F1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>En Etkili Yorumcular</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("En çok beğeni alan yorumcular ve etkileşim metrikleri")
    
    # Aggregate author stats
    author_stats = {}
    for c in all_comments:
        author = c.get('yazar', 'Anonim')
        likes = c.get('begeni', 0)
        length = len(c.get('metin', ''))
        
        if author not in author_stats:
            author_stats[author] = {'total_likes': 0, 'comment_count': 0, 'total_length': 0}
        
        author_stats[author]['total_likes'] += likes
        author_stats[author]['comment_count'] += 1
        author_stats[author]['total_length'] += length
    
    # Sort by total likes
    sorted_authors = sorted(author_stats.items(), key=lambda x: x[1]['total_likes'], reverse=True)[:10]
    
    if sorted_authors:
        col_chart, col_table = st.columns([1.5, 1])
        
        with col_chart:
            authors = [a[0][:15] + '...' if len(a[0]) > 15 else a[0] for a in sorted_authors]
            total_likes = [a[1]['total_likes'] for a in sorted_authors]
            
            # Sunset gradient colors
            colors = px.colors.sequential.Sunset[:len(authors)]
            if len(colors) < len(authors):
                colors = px.colors.sequential.Sunset * 2
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=authors[::-1],
                x=total_likes[::-1],
                orientation='h',
                marker=dict(
                    color=colors[:len(authors)][::-1],
                    line=dict(width=1, color='rgba(255,255,255,0.5)')
                ),
                text=[f" {l:,}" for l in total_likes[::-1]],
                textposition='outside',
                textfont=dict(size=11, color='#E2E8F0'),
                hovertemplate='<b>%{y}</b><br>Toplam Beğeni: %{x:,}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Toplam Beğeni",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0', size=12),
                height=380,
                margin=dict(l=20, r=60, t=10, b=40),
                hoverlabel=dict(bgcolor="#1E293B", bordercolor="#8B5CF6")
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', showgrid=True, zeroline=False)
            fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.markdown("**📊 Detaylı Metrikler**")
            
            # Top 5 with detailed metrics
            for i, (author, stats) in enumerate(sorted_authors[:5], 1):
                avg_likes = stats['total_likes'] / stats['comment_count'] if stats['comment_count'] > 0 else 0
                avg_length = stats['total_length'] / stats['comment_count'] if stats['comment_count'] > 0 else 0
                
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                
                st.markdown(f"""
                <div style='background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); 
                            padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #C4B5FD; font-weight: 600;'>{medal} {author[:18]}</span>
                        <span style='color: #A78BFA; font-size: 0.85rem;'>❤️ {stats['total_likes']:,}</span>
                    </div>
                    <div style='color: #94A3B8; font-size: 0.7rem; margin-top: 4px;'>
                        {stats['comment_count']} yorum • Ort: {avg_likes:.1f} beğeni • {avg_length:.0f} char
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Yeterli yazar verisi yok.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    # ========== 4. SORU ANALİZİ (Donut Chart) ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #F59E0B, #EF4444); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>4️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #F59E0B, #EF4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Soru Analizi</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Soru işareti veya soru kalıbı içeren yorumların oranı")
    
    question_patterns = ['?', 'nasıl', 'neden', 'ne zaman', 'nerede', 'kim', 'hangi', 'kaç', 'mi ', 'mı ', 'mu ', 'mü ']
    
    question_count = 0
    for c in all_comments:
        text_lower = c.get('metin', '').lower()
        if any(pattern in text_lower for pattern in question_patterns):
            question_count += 1
    
    non_question_count = len(all_comments) - question_count
    
    col_donut, col_stats = st.columns([1.5, 1])
    
    with col_donut:
        fig = go.Figure(go.Pie(
            labels=['❓ Soru İçeren', '💬 Normal Yorum'],
            values=[question_count, non_question_count],
            hole=0.6,
            marker=dict(
                colors=['#F59E0B', '#3B82F6'],
                line=dict(width=2, color='#0F172A')
            ),
            textinfo='percent+label',
            textposition='outside',
            textfont=dict(size=13, color='#E2E8F0'),
            hovertemplate='<b>%{label}</b><br>Sayı: %{value}<br>Oran: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f"<b>{question_count}</b><br><span style='font-size:12px'>Soru</span>",
                x=0.5, y=0.5, font_size=24, font_color='#F59E0B', showarrow=False
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        question_pct = (question_count / len(all_comments) * 100) if len(all_comments) > 0 else 0
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)); 
                    border: 1px solid rgba(245, 158, 11, 0.3); padding: 24px; border-radius: 12px; margin-top: 20px;'>
            <div style='color: #94A3B8; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px;'>Soru Oranı</div>
            <div style='color: #FBBF24; font-size: 2.8rem; font-weight: 800; line-height: 1;'>{question_pct:.1f}%</div>
            <div style='color: #64748B; font-size: 0.8rem; margin-top: 12px;'>
                {question_count} soru / {len(all_comments)} yorum
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Insight
        if question_pct > 30:
            insight = "⚠️ Kitle çok soru soruyor! Daha fazla açıklama veya FAQ gerekebilir."
            insight_color = "#F59E0B"
        elif question_pct > 15:
            insight = "💡 Orta düzeyde soru var. İçeriğiniz anlaşılır görünüyor."
            insight_color = "#3B82F6"
        else:
            insight = "✅ Çok az soru var. İçeriğiniz açık ve net!"
            insight_color = "#10B981"
        
        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.5); border-left: 3px solid {insight_color}; 
                    padding: 12px 16px; margin-top: 16px; border-radius: 6px;'>
            <span style='color: #E2E8F0; font-size: 0.85rem;'>{insight}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 5. EN ÇOK KULLANILAN EMOJİLER ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #06B6D4, #3B82F6); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>5️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; background: linear-gradient(135deg, #06B6D4, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>TF-IDF Anahtar Kelime Analizi</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("En ayırt edici ve önemli kelimeler (sadece sık kullanılan değil, gerçekten anlamlı olanlar)")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import nltk
        from nltk.corpus import stopwords
        
        # Download Turkish stopwords if not already downloaded
        try:
            turkish_stopwords = list(stopwords.words('turkish'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            turkish_stopwords = list(stopwords.words('turkish'))
        
        # Clean texts
        cleaned_texts = [t.lower().strip() for t in texts if len(t) > 20]
        
        if len(cleaned_texts) > 5:
            # TF-IDF Vectorizer with Turkish stopwords
            tfidf = TfidfVectorizer(
                max_features=30,
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.7,
                stop_words=turkish_stopwords,
                token_pattern=r'(?u)\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b'  # At least 3 chars, Turkish letters
            )
            tfidf_matrix = tfidf.fit_transform(cleaned_texts)
            
            # Get feature names and their average TF-IDF scores
            feature_names = tfidf.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Sort by score
            sorted_indices = np.argsort(avg_scores)[::-1]
            top_words = [(feature_names[i], avg_scores[i]) for i in sorted_indices[:15]]
            
            if top_words:
                words = [w[0] for w in top_words]
                scores = [w[1] for w in top_words]
                
                # Viridis gradient
                colors = px.colors.sequential.Viridis[:len(words)]
                if len(colors) < len(words):
                    colors = px.colors.sequential.Viridis * 2
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=words[::-1],
                    x=scores[::-1],
                    orientation='h',
                    marker=dict(
                        color=colors[:len(words)][::-1],
                        line=dict(width=1, color='rgba(255,255,255,0.5)')
                    ),
                    text=[f" {s:.4f}" for s in scores[::-1]],
                    textposition='outside',
                    textfont=dict(size=11, color='#E2E8F0'),
                    hovertemplate='<b>%{y}</b><br>TF-IDF Skoru: %{x:.4f}<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="TF-IDF Skoru (Yüksek = Daha Önemli)",
                    yaxis_title="",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0', size=12),
                    height=420,
                    margin=dict(l=20, r=80, t=10, b=40),
                    hoverlabel=dict(bgcolor="#1E293B", bordercolor="#06B6D4")
                )
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.08)', showgrid=True, zeroline=False)
                fig.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight
                st.markdown(f"""
                <div style='background: rgba(30, 41, 59, 0.5); border-left: 3px solid #06B6D4; 
                            padding: 12px 16px; margin-top: 8px; border-radius: 6px;'>
                    <span style='color: #E2E8F0; font-size: 0.85rem;'>
                        💡 <b>TF-IDF</b> (Term Frequency-Inverse Document Frequency), bir kelimenin hem sıklığını hem de "benzersizliğini" ölçer. 
                        Yüksek skorlu kelimeler, bu yorumlarda özellikle öne çıkan ve anlamlı terimlerdir.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Yeterli TF-IDF verisi hesaplanamadı.")
        else:
            st.info("TF-IDF analizi için en az 5 yorum gerekli.")
    except Exception as e:
        st.warning(f"TF-IDF analizi yapılamadı: {e}")


# ============= MAIN =============
def main():
    inject_theme()
    init_session_state()
    
    with st.sidebar:
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 20px 0 10px 0;'>
            <h2 style='font-size: 1.5rem; margin: 0; background: linear-gradient(135deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                YCA Studio
            </h2>
            <p style='font-size: 0.75rem; color: #64748B; margin-top: 4px;'>Insight Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        # Clean navigation without emojis
        pages = [
            ("Home", "home"),
            ("Analysis", "analyze"),
            ("Battle", "battle"),
            ("Statistics", "stats")
        ]
        
        for name, key in pages:
            is_active = st.session_state.page == key
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(name, use_container_width=True, type=btn_type, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
        
        # Footer
        st.markdown("<div style='flex-grow: 1'></div>", unsafe_allow_html=True)
        st.divider()
        

    if st.session_state.page == 'home': page_home()
    elif st.session_state.page == 'analyze': page_analyze()
    elif st.session_state.page == 'battle': page_battle()
    elif st.session_state.page == 'stats': page_stats()

if __name__ == "__main__":
    main()
