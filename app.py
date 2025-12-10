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

# Add src to path for cleaner structure
sys.path.append(os.path.abspath("src"))

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
        create_sentiment_bubble_chart,
        create_battle_trend_chart,
        create_temporal_sentiment_chart,
        create_category_pie_grid,
        create_category_temporal_chart
    )
    from components.wordcloud_gen import generate_wordcloud, get_word_frequencies_from_texts, generate_sentiment_wordcloud
    from components.progress_bar import ProgressBar, create_battle_progress_callback
    from battle_analyzer import BattleAnalyzer
except ImportError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()


# ============= THEME CONFIGURATION (LIGHT MODE) =============
THEME = {
    "bg_color": "#FFFFFF",           # Pure White
    "main_bg": "#F8F9FA",            # Light Gray Background
    "sidebar_bg": "#FFFFFF",         # White Sidebar
    "card_bg": "#FFFFFF",            # White Cards
    "card_bg_solid": "#FFFFFF",
    "text_primary": "#1F2937",       # Dark Gray (almost black)
    "text_secondary": "#4B5563",     # Medium Gray
    "text_muted": "#6B7280",         # Light Gray Text
    "accent": "#4169E1",             # Royal Blue (Logo Color)
    "accent_secondary": "#6366F1",   # Indigo
    "accent_hover": "#3B5ED9",       # Darker Royal Blue
    "success": "#059669",            # Emerald 600 (darker for light bg)
    "error": "#DC2626",              # Red 600
    "warning": "#D97706",            # Amber 600
    "border": "rgba(0, 0, 0, 0.08)", # Light border
    "input_bg": "#F9FAFB",           # Very light gray input
    "hover_bg": "rgba(65, 105, 225, 0.08)"  # Light Royal Blue hover
}


def inject_theme():
    """Professional UI Theme Injection - LIGHT MODE"""
    t = THEME
    
    st.markdown(f"""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* GLOBAL RESET - LIGHT MODE */
        .stApp {{
            background-color: {t["main_bg"]};
            font-family: 'Inter', sans-serif;
            color: {t["text_primary"]};
        }}
        
        /* MAIN CONTAINER - Wider and centered */
        .block-container {{
            padding-top: 2.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1280px !important;
            margin: 0 auto !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}
        
        /* SIDEBAR - Light Mode with clear separation */
        [data-testid="stSidebar"] {{
            background-color: #F8F9FA;
            border-right: 1px solid #E5E7EB;
            padding-top: 1rem;
            box-shadow: 2px 0 12px rgba(0,0,0,0.03);
        }}
        
        /* Sidebar buttons - More spacing between items */
        [data-testid="stSidebar"] .stButton {{
            margin-bottom: 8px !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            color: {t["text_secondary"]} !important;
            text-transform: none !important;
            font-weight: 500 !important;
            letter-spacing: 0 !important;
            padding: 14px 18px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
            margin-bottom: 4px !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {t["hover_bg"]} !important;
            color: {t["accent"]} !important;
            border-left: 4px solid {t["accent"]} !important;
            padding-left: 14px !important;
        }}
        
        /* Active sidebar button */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: {t["hover_bg"]} !important;
            color: {t["accent"]} !important;
            border-left: 4px solid {t["accent"]} !important;
            border-radius: 10px !important;
        }}
        
        /* Remove sidebar dividers */
        [data-testid="stSidebar"] hr {{
            display: none !important;
        }}
        
        /* TYPOGRAPHY - Light Mode */
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
            color: {t["text_muted"]};
        }}
        
        /* Captions/hints - Light Mode readable */
        .stCaption, [data-testid="stCaption"] {{
            color: {t["text_muted"]} !important;
        }}
        
        /* CARDS & CONTAINERS - Light Mode with enhanced SHADOWS */
        .glass-card {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .glass-card:hover {{
            border-color: {t["accent"]};
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        }}
        
        /* KPI/METRIC CARDS - Modern Light Style */
        .metric-card {{
            background: {t["card_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }}
        
        .metric-card:hover {{
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }}
        
        /* COMMENT CARD - Styled for Data Feed */
        .comment-card {{
            background: {t["card_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }}
        
        .comment-card:hover {{
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
            border-color: {t["accent"]};
        }}
        
        .comment-author {{
            color: {t["accent"]};
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 8px;
        }}
        
        .comment-text {{
            color: {t["text_primary"]};
            font-size: 0.95rem;
            line-height: 1.6;
        }}
        
        .comment-meta {{
            color: {t["text_muted"]};
            font-size: 0.8rem;
            margin-top: 10px;
        }}
        
        /* AI SUMMARY BOX - Light Mode */
        .ai-summary-box {{
            background: linear-gradient(135deg, rgba(65, 105, 225, 0.05), rgba(99, 102, 241, 0.05));
            border: 1px solid rgba(65, 105, 225, 0.2);
            border-left: 4px solid {t["accent"]};
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
        }}
        
        .ai-summary-box h4 {{
            color: {t["accent"]} !important;
            font-size: 1.1rem;
            margin-bottom: 16px;
        }}
        
        .ai-summary-box p, .ai-summary-box li {{
            color: {t["text_primary"]};
            line-height: 1.7;
        }}
        
        /* Home page cards container limit */
        .home-cards-container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        
        /* FORM CONTAINER - Light Mode */
        .form-container {{
            max-width: 800px;
            margin: 0 auto;
            background: {t["card_bg"]};
            border-radius: 16px;
            padding: 32px;
            border: 1px solid {t["border"]};
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        }}
        
        /* BUTTONS - Royal Blue */
        .stButton > button {{
            background: linear-gradient(135deg, {t["accent"]} 0%, {t["accent_hover"]} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.875rem 1.5rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            font-size: 0.875rem;
            box-shadow: 0 4px 15px -3px rgba(65, 105, 225, 0.3);
            max-width: 600px;
            height: 48px;
        }}
        
        .stButton > button:hover:not(:disabled) {{
            filter: brightness(1.1);
            box-shadow: 0 8px 25px -5px rgba(65, 105, 225, 0.4);
            transform: translateY(-2px);
        }}
        
        .stButton > button:active:not(:disabled) {{
            transform: translateY(0);
        }}
        
        .stButton > button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
            box-shadow: none !important;
        }}

        /* Secondary/Outlined buttons */
        [data-testid="baseButton-secondary"] {{
            background: transparent !important;
            border: 2px solid {t["accent"]} !important;
            color: {t["accent"]} !important;
            box-shadow: none !important;
        }}

        [data-testid="baseButton-secondary"]:hover {{
            background: {t["hover_bg"]} !important;
            border-color: {t["accent"]} !important;
            transform: translateY(-2px) !important;
        }}
        
        /* DELETE/CLOSE BUTTONS - Red theme */
        .delete-btn button, 
        button[data-testid*="delete"],
        button:has(svg[data-testid*="close"]) {{
            background: transparent !important;
            border: 1px solid {t["error"]} !important;
            color: {t["error"]} !important;
            box-shadow: none !important;
        }}
        
        .delete-btn button:hover {{
            background: rgba(220, 38, 38, 0.08) !important;
        }}
        
        /* ============================================= */
        /* ULTRA AGGRESSIVE INPUT STYLES - FORCE WHITE  */
        /* ============================================= */
        
        /* Target ALL possible input elements */
        input, 
        textarea,
        input[type="text"],
        input[type="number"],
        input[type="email"],
        input[type="password"],
        input[type="search"],
        input[type="url"],
        input[type="tel"] {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            color: #333333 !important;
            -webkit-text-fill-color: #333333 !important;
        }}
        
        /* Streamlit specific - Text Input */
        .stTextInput input,
        .stTextInput > div > div > input,
        [data-testid="stTextInput"] input,
        [data-baseweb="input"] input {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            border-radius: 8px !important;
            color: #333333 !important;
            -webkit-text-fill-color: #333333 !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
            height: 46px !important;
            min-height: 46px !important;
        }}
        
        /* Streamlit specific - Number Input (THE PROBLEM) */
        .stNumberInput input,
        .stNumberInput > div > div > input,
        .stNumberInput input[type="number"],
        [data-testid="stNumberInput"] input,
        [data-testid="stNumberInput-StepUp"],
        [data-testid="stNumberInput-StepDown"],
        [data-baseweb="input"] input[type="number"],
        div[data-baseweb="base-input"] input {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            border-radius: 8px !important;
            color: #333333 !important;
            -webkit-text-fill-color: #333333 !important;
            padding: 12px 16px !important;
            font-size: 1rem !important;
            height: 46px !important;
        }}
        
        /* Number input container and wrapper */
        .stNumberInput > div,
        .stNumberInput [data-baseweb="input"],
        [data-testid="stNumberInput"] > div {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border-radius: 8px !important;
        }}
        
        /* BaseWeb input wrapper (Streamlit uses this) */
        [data-baseweb="input"],
        [data-baseweb="base-input"] {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            border-radius: 8px !important;
        }}
        
        /* Number input step buttons */
        .stNumberInput button,
        [data-testid="stNumberInput"] button {{
            background-color: #F3F4F6 !important;
            border: 1px solid #E0E0E0 !important;
            color: #333333 !important;
        }}
        
        .stNumberInput button:hover {{
            background-color: #E5E7EB !important;
        }}
        
        /* TextArea */
        .stTextArea textarea,
        [data-testid="stTextArea"] textarea {{
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            border-radius: 8px !important;
            color: #333333 !important;
            -webkit-text-fill-color: #333333 !important;
        }}
        
        /* Input placeholder text */
        input::placeholder,
        textarea::placeholder {{
            color: #9CA3AF !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #9CA3AF !important;
        }}
        
        /* Focus states - Blue border */
        input:focus,
        textarea:focus,
        .stTextInput input:focus,
        .stNumberInput input:focus,
        [data-baseweb="input"]:focus-within {{
            border-color: #4A90E2 !important;
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.15) !important;
            outline: none !important;
            background-color: #FFFFFF !important;
        }}
        
        /* Remove any dark overlays */
        .stTextInput > div,
        .stNumberInput > div,
        [data-testid="stTextInput"] > div,
        [data-testid="stNumberInput"] > div {{
            background-color: transparent !important;
        }}
        
        /* Selectbox */
        .stSelectbox > div > div,
        [data-testid="stSelectbox"] > div > div {{
            background-color: #FFFFFF !important;
            border: 1px solid #E0E0E0 !important;
            border-radius: 8px !important;
            color: #333333 !important;
        }}
        
        /* ============================================= */
        /* END INPUT STYLES                             */
        /* ============================================= */
        
        /* INPUT GROUP - Unified look */
        [data-testid="column"]:first-child .stTextInput > div > div > input {{
            border-top-right-radius: 0 !important;
            border-bottom-right-radius: 0 !important;
            border-right: 1px solid #E0E0E0 !important;
        }}
        
        [data-testid="column"]:last-child .stNumberInput > div > div > input {{
            border-top-left-radius: 0 !important;
            border-bottom-left-radius: 0 !important;
        }}
        
        /* Horizontal row alignment */
        [data-testid="stHorizontalBlock"] {{
            align-items: flex-end !important;
        }}
        
        /* Main buttons same height */
        .stButton > button {{
            height: 48px !important;
            min-height: 48px !important;
        }}
        
        /* RADIO BUTTONS - Modern Segmented Control */
        .stRadio > div {{
            display: inline-flex !important;
            gap: 4px !important;
            background: #F3F4F6 !important;
            border-radius: 24px !important;
            padding: 4px !important;
            border: 1px solid #E5E7EB !important;
            max-width: 450px !important;
        }}
        
        .stRadio > div > label {{
            flex: 1 !important;
            padding: 12px 24px !important;
            border-radius: 20px !important;
            cursor: pointer !important;
            transition: all 0.25s ease !important;
            text-align: center !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            color: #4B5563 !important;
            margin: 0 !important;
            white-space: nowrap !important;
            background: transparent !important;
        }}
        
        .stRadio > div > label:hover {{
            background: rgba(65, 105, 225, 0.08) !important;
            color: {t["accent"]} !important;
        }}
        
        /* Selected state for radio - Blue background, white text */
        .stRadio > div > label[data-baseweb="radio"] input:checked + div,
        .stRadio [data-baseweb="radio"] > div:first-child[aria-checked="true"],
        .stRadio div[role="radiogroup"] > label:has(input:checked) {{
            background: linear-gradient(135deg, #4A90E2, #357ABD) !important;
            color: white !important;
            box-shadow: 0 4px 12px -2px rgba(74, 144, 226, 0.4) !important;
        }}
        
        /* Alternative selector for checked state */
        .stRadio label[data-baseweb="radio"]:has(input:checked) {{
            background: linear-gradient(135deg, #4A90E2, #357ABD) !important;
            color: white !important;
            box-shadow: 0 4px 12px -2px rgba(74, 144, 226, 0.4) !important;
        }}
        
        /* Hide the actual radio circle */
        .stRadio > div > label > div:first-child,
        .stRadio [data-baseweb="radio"] > div > div:first-child {{
            display: none !important;
        }}
        
        /* METRICS - Light Mode */
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: {t["accent"]} !important;
            -webkit-text-fill-color: {t["accent"]} !important;
            line-height: 1.2 !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.7rem !important;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {t["text_muted"]} !important;
            opacity: 0.8;
            margin-bottom: 4px !important;
        }}
        
        /* TABS - Light Mode */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 2px solid {t["border"]};
            padding-bottom: 0px;
            background: transparent;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            white-space: nowrap;
            background-color: transparent;
            border: none;
            border-radius: 8px 8px 0 0;
            color: {t["text_secondary"]};
            font-weight: 500;
            padding: 0 20px;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background: {t["hover_bg"]};
            color: {t["accent"]};
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {t["accent"]} !important;
            border-bottom: 4px solid {t["accent"]} !important;
            background: {t["hover_bg"]};
            margin-bottom: -2px;
        }}
        
        /* EXPANDERS - Light Mode */
        .streamlit-expanderHeader {{
            background: {t["input_bg"]};
            border-radius: 10px;
            border: 1px solid {t["border"]};
            transition: all 0.2s ease;
        }}
        
        .streamlit-expanderHeader:hover {{
            border-color: {t["accent"]};
            background: {t["hover_bg"]};
        }}
        
        /* PROGRESS BAR */
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, {t["accent"]}, {t["accent_secondary"]});
            border-radius: 10px;
        }}
        
        /* SPINNER */
        .stSpinner > div {{
            border-color: {t["accent"]} transparent transparent transparent !important;
        }}

        /* DATAFRAMES - Light Mode */
        [data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {t["border"]};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }}

        /* HIDE DEFAULT ELEMENTS */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        [data-testid="stHeader"] {{
            display: none !important;
        }}
        
        /* SIDEBAR TOGGLE - Keep visible */
        [data-testid="collapsedControl"] {{
            display: flex !important;
        }}
        
        /* TIP BOX - Light Mode */
        .tip-box {{
            background: linear-gradient(135deg, rgba(65, 105, 225, 0.06), rgba(99, 102, 241, 0.06));
            border: 1px solid rgba(65, 105, 225, 0.15);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .tip-box h4 {{
            color: {t["accent"]} !important;
            margin-bottom: 12px;
        }}
        
        /* FEATURE CARDS - Light Mode */
        .feature-card {{
            background: {t["card_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            border-color: {t["accent"]};
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .feature-icon {{
            font-size: 2.5rem;
            margin-bottom: 12px;
        }}
        
        /* SENTIMENT STATUS BADGES */
        .sentiment-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .sentiment-badge.positive {{
            background: rgba(5, 150, 105, 0.1);
            color: {t["success"]};
            border: 1px solid rgba(5, 150, 105, 0.3);
        }}
        
        .sentiment-badge.negative {{
            background: rgba(220, 38, 38, 0.1);
            color: {t["error"]};
            border: 1px solid rgba(220, 38, 38, 0.3);
        }}
        
        .sentiment-badge.neutral {{
            background: rgba(107, 114, 128, 0.1);
            color: {t["text_muted"]};
            border: 1px solid rgba(107, 114, 128, 0.3);
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
    """Professional Home Dashboard - LIGHT MODE"""
    
    # ============ HERO SECTION ============
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 50px 0;'>
        <div style='margin-bottom: 16px;'>
            <span style='background: linear-gradient(135deg, #4169E1, #6366F1); padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; color: white; font-weight: 500;'>
                AI-Powered Analytics
            </span>
        </div>
        <h1 style='font-size: 3rem; font-weight: 800; margin: 0 0 16px 0; 
                   background: linear-gradient(135deg, #1F2937 0%, #4B5563 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            YouTube Comment Analyzer
        </h1>
        <p style='font-size: 1.1rem; color: #6B7280; max-width: 500px; margin: 0 auto; line-height: 1.6;'>
            Extract sentiment insights, discover trends, and understand your audience with local AI processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ FEATURE BADGES ============
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 16px; margin-bottom: 48px; flex-wrap: wrap;'>
        <div style='display: flex; align-items: center; gap: 10px; background: rgba(5, 150, 105, 0.08); border: 1px solid rgba(5, 150, 105, 0.25); border-radius: 20px; padding: 8px 16px;'>
            <span style='color: #059669; font-size: 1rem;'>✓</span>
            <span style='color: #4B5563; font-size: 0.875rem; font-weight: 500;'>Yerel İşleme</span>
        </div>
        <div style='display: flex; align-items: center; gap: 10px; background: rgba(65, 105, 225, 0.08); border: 1px solid rgba(65, 105, 225, 0.25); border-radius: 20px; padding: 8px 16px;'>
            <span style='color: #4169E1; font-size: 1rem;'>⚡</span>
            <span style='color: #4B5563; font-size: 0.875rem; font-weight: 500;'>GPU Hızlandırma</span>
        </div>
        <div style='display: flex; align-items: center; gap: 10px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 20px; padding: 8px 16px;'>
            <span style='color: #6366F1; font-size: 1rem;'>🔑</span>
            <span style='color: #4B5563; font-size: 0.875rem; font-weight: 500;'>API Gerektirmez</span>
        </div>
        <div style='display: flex; align-items: center; gap: 10px; background: rgba(217, 119, 6, 0.08); border: 1px solid rgba(217, 119, 6, 0.25); border-radius: 20px; padding: 8px 16px;'>
            <span style='color: #D97706; font-size: 1rem;'>🔒</span>
            <span style='color: #4B5563; font-size: 0.875rem; font-weight: 500;'>Gizlilik Öncelikli</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ MAIN ACTION CARDS ============
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #4169E1; min-height: 200px; background: #FFFFFF;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="background: linear-gradient(135deg, #4169E1, #3B5ED9); width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1F2937;">Tekil Video</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Derinlemesine analiz</div>
                </div>
            </div>
            <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.7; margin-bottom: 16px;">
                Tek bir YouTube videosunun yorumlarını analiz edin. Duygu dağılımı, anahtar kelime çıkarımı ve AI destekli içgörüler alın.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("VİDEO ANALİZ ET", type="primary", use_container_width=True, key="btn_single"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode = "Single Video"
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #6366F1; min-height: 200px; background: #FFFFFF;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <div style="background: linear-gradient(135deg, #6366F1, #4F46E5); width: 48px; height: 48px; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #1F2937;">Toplu Arama</div>
                    <div style="font-size: 0.8rem; color: #6B7280;">Çoklu video karşılaştırma</div>
                </div>
            </div>
            <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.7; margin-bottom: 16px;">
                YouTube'da arama yapın ve birden fazla videoyu aynı anda analiz edin. Rakiplerin duygu durumlarını karşılaştırın.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("TOPLU ARAMA", type="secondary", use_container_width=True, key="btn_multi"):
            st.session_state.page = 'analyze'
            st.session_state.analysis_mode = "Multi-Video Batch"
            st.rerun()
    
    # ============ QUICK START GUIDE ============
    st.markdown("<div style='height: 48px'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 24px;'>
        <span style='font-size: 0.85rem; color: #6B7280; text-transform: uppercase; letter-spacing: 2px;'>Nasıl Çalışır?</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(65, 105, 225, 0.1); width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto; border: 1px solid rgba(65, 105, 225, 0.25);'>
                <span style='color: #4169E1; font-weight: 700; font-size: 1.4rem;'>1</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 8px; font-size: 1rem;'>URL Yapıştır</div>
            <div style='font-size: 0.9rem; color: #6B7280; line-height: 1.5;'>Herhangi bir YouTube video linkini girin</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(99, 102, 241, 0.1); width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto; border: 1px solid rgba(99, 102, 241, 0.25);'>
                <span style='color: #6366F1; font-weight: 700; font-size: 1.4rem;'>2</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 8px; font-size: 1rem;'>İşle</div>
            <div style='font-size: 0.9rem; color: #6B7280; line-height: 1.5;'>AI yorumları yerel olarak analiz eder</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: rgba(5, 150, 105, 0.1); width: 56px; height: 56px; border-radius: 14px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto; border: 1px solid rgba(5, 150, 105, 0.25);'>
                <span style='color: #059669; font-weight: 700; font-size: 1.4rem;'>3</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 8px; font-size: 1rem;'>İçgörüler</div>
            <div style='font-size: 0.9rem; color: #6B7280; line-height: 1.5;'>Grafikler, trendler ve raporları görün</div>
        </div>
        """, unsafe_allow_html=True)




def page_analyze():
    """Professional Analysis Page - LIGHT MODE"""
    
    st.markdown("""
    <div style='margin-bottom: 28px;'>
        <h2 style='font-size: 1.8rem; margin: 0; color: #1F2937;'>Analysis Console</h2>
        <p style='color: #6B7280; font-size: 0.9rem; margin-top: 4px;'>Extract insights from YouTube video comments</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure analysis_mode is in session_state
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "single"
    
    # --- COMPACT MODE SELECTION (Pills Style) ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='background: #4169E1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>1</span>
        <span style='font-size: 1rem; font-weight: 600; color: #1F2937;'>Analiz Modu</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Try st.pills (Streamlit 1.33+) or fallback to radio
    try:
        mode = st.pills(
            "Mod",
            options=["🎥 Tekil Video", "🚀 Toplu Arama"],
            default="🎥 Tekil Video" if st.session_state.analysis_mode == "single" else "🚀 Toplu Arama",
            label_visibility="collapsed"
        )
        is_single = mode == "🎥 Tekil Video"
    except AttributeError:
        # Fallback for older Streamlit versions - use segmented_control or radio
        try:
            mode = st.segmented_control(
                "Mod",
                options=["🎥 Tekil Video", "🚀 Toplu Arama"],
                default="🎥 Tekil Video" if st.session_state.analysis_mode == "single" else "🚀 Toplu Arama",
                label_visibility="collapsed"
            )
            is_single = mode == "🎥 Tekil Video"
        except AttributeError:
            # Final fallback to radio
            mode = st.radio(
                "Mod",
                ["🎥 Tekil Video", "🚀 Toplu Arama"],
                horizontal=True,
                label_visibility="collapsed",
                index=0 if st.session_state.analysis_mode == "single" else 1
            )
            is_single = mode == "🎥 Tekil Video"
    
    # Update session state
    st.session_state.analysis_mode = "single" if is_single else "multi"
    
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    
    if is_single:
        analyze_single_video()
    else:
        analyze_multi_video()


def analyze_single_video():
    # --- STEP 2: Input Configuration ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
        <span style='background: #6366F1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>2</span>
        <span style='font-size: 1rem; font-weight: 600; color: #1F2937;'>Video Bilgileri</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic help text for single video mode
    st.caption("📌 Analiz etmek istediğiniz YouTube videosunun linkini aşağıya yapıştırın.")
    
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
            comments = [c.get('metin_duygu') or c.get('metin', '') for c in results['yorumlar']]
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
    # --- STEP 2: Search Configuration ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
        <span style='background: #6366F1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>2</span>
        <span style='font-size: 1rem; font-weight: 600; color: #1F2937;'>Arama Ayarları</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic help text for multi-video mode
    st.caption("🔍 YouTube'da aranacak sorguyu girin, birden fazla videodan toplu yorum analizi yapılacaktır.")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search Query", placeholder="e.g., 'iPhone 15 inceleme'")
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
            
            comment_texts = [c.get('metin_duygu') or c.get('metin', '') for c in all_comments]
            
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
    
    # Video Header - Light Mode
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 32px;'>
        <h3 style='font-size: 1.5rem; color: #1F2937; margin-bottom: 8px;'>{data.get('baslik', 'Video Title')}</h3>
        <span style='color: #6B7280; font-size: 0.9rem; background: #F3F4F6; padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(0,0,0,0.08);'>
            {data.get('kanal', 'Unknown Channel')} &bull; {data.get('sure', '0')}s
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Video Description & AI Analysis (Simplified)
    if data.get('aciklama'):
        with st.expander("📝 AI Video Özeti", expanded=False):
            desc_key = f"desc_summary_{data.get('video_id')}"
            
            # Eğer özet henüz oluşturulmadıysa butonu göster
            if desc_key not in st.session_state:
                if st.button("✨ Video İçeriğini Özetle", key="btn_analyze_desc", use_container_width=True):
                    with st.spinner("AI video içeriğini analiz ediyor..."):
                        try:
                            ollama = OllamaLLM(model_name="gemma3:4b")
                            if ollama.check_connection():
                                summary = ollama.summarize_video_description(data['aciklama'])
                                st.session_state[desc_key] = summary
                                st.rerun()
                            else:
                                st.error("Ollama bağlantısı başarısız. Lütfen Ollama'nın çalıştığından emin olun.")
                        except Exception as e:
                            st.error(f"Analiz hatası: {e}")
            
            # Özet varsa temiz bir kutu içinde göster
            else:
                st.info(st.session_state[desc_key], icon="ℹ️")
                
                # İsteğe bağlı yeniden oluşturma
                if st.button("🔄 Yeniden Özetle", key="btn_reanalyze_desc", type="secondary", help="Özeti tekrar oluştur"):
                    del st.session_state[desc_key]
                    st.rerun()
    
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    
    # Custom Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    
    sentiment_score = 0
    if res:
        stats = SentimentAnalyzer().get_summary_stats(res)
        sentiment_score = stats['sentiment_score']
    
    # Determine sentiment status
    if sentiment_score > 0:
        s_color = "#059669"
        s_icon = "🟢"
        s_status = "Pozitif"
    elif sentiment_score < 0:
        s_color = "#DC2626"
        s_icon = "🔴"
        s_status = "Negatif"
    else:
        s_color = "#6B7280"
        s_icon = "🟡"
        s_status = "Nötr"
    
    # All cards have matching height for visual consistency
    CARD_HEIGHT = "88px"
    
    def metric_card(label, value, color="#4169E1"):
        return f"""
        <div style='background: #FFFFFF; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.06); height: {CARD_HEIGHT}; display: flex; flex-direction: column; justify-content: center;'>
            <div style='color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;'>{label}</div>
            <div style='color: #1F2937; font-size: 1.5rem; font-weight: 700; margin-top: 4px;'>{value}</div>
        </div>
        """
    
    def sentiment_card(score, color, icon, status):
        """Compact sentiment score card - same height as other metric cards"""
        arrow = "🔼" if score > 0 else "🔻" if score < 0 else "➖"
        return f"""
        <div style='background: #FFFFFF; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.06); height: {CARD_HEIGHT}; display: flex; flex-direction: column; justify-content: center;'>
            <div style='color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;'>Sentiment Score</div>
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 4px;'>
                <span style='color: {color}; font-size: 1.5rem; font-weight: 700;'>{arrow} {score:.2f}</span>
                <span style='font-size: 0.75rem; font-weight: 600; color: {color}; background: #F3F4F6; padding: 2px 8px; border-radius: 10px;'>{icon} {status}</span>
            </div>
        </div>
        """

    with c1: st.markdown(metric_card("Total Comments", len(comments), "#4169E1"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Total Likes", f"{sum(c.get('begeni', 0) for c in comments):,}", "#6366F1"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Video Views", f"{data.get('goruntulenme', 0):,}", "#059669"), unsafe_allow_html=True)
    
    # Sentiment Score with dedicated card
    with c4: st.markdown(sentiment_card(sentiment_score, s_color, s_icon, s_status), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1F2937; margin-bottom: 20px; border-left: 4px solid #D97706; padding-left: 12px;'>Visual Insights</h3>", unsafe_allow_html=True)
    
    # Prepare rich context for AI
    description_excerpt = data.get('aciklama', '')[:1000]
    if description_excerpt:
        context = f"Video Title: {data.get('baslik', '')}\n\nVideo Description Context: {description_excerpt}..."
    else:
        context = data.get('baslik', '')
        
    display_tabs(comments, res, context)


def display_multi_video_results():
    videos = st.session_state.multi_video_data
    all_comments = []
    for v in videos:
        all_comments.extend(v.get('yorumlar', []))
    
    st.divider()
    
    # Updated header with total comments
    total_comments = len(all_comments)
    st.markdown(f"""
    <div style='margin-bottom: 24px;'>
        <h2 style='font-size: 1.6rem; color: #1F2937; margin: 0;'>
            📊 Batch Results - {len(videos)} Video, {total_comments:,} Yorum
        </h2>
        <p style='color: #6B7280; font-size: 0.9rem; margin-top: 6px;'>Toplu analiz sonuçları</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats with custom styling
    c1, c2, c3, c4 = st.columns(4)
    
    def stat_card(label, value, color="#4169E1", emoji=""):
        emoji_html = f"<span style='margin-right: 4px;'>{emoji}</span>" if emoji else ""
        return f"""
        <div style='background: #FFFFFF; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #E5E7EB;'>
            <div style='color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;'>{label}</div>
            <div style='color: #1F2937; font-size: 1.4rem; font-weight: 700; margin-top: 4px;'>{emoji_html}{value}</div>
        </div>
        """
    
    with c1: st.markdown(stat_card("Total Comments", f"{total_comments:,}", "#4169E1"), unsafe_allow_html=True)
    with c2: st.markdown(stat_card("Total Likes", f"{sum(c.get('begeni', 0) for c in all_comments):,}", "#6366F1"), unsafe_allow_html=True)
    with c3: st.markdown(stat_card("Videos Scanned", len(videos), "#059669"), unsafe_allow_html=True)
    
    if st.session_state.multi_video_sentiment:
        stats = SentimentAnalyzer().get_summary_stats(st.session_state.multi_video_sentiment)
        sentiment_score = stats['sentiment_score']
        # Determine color and emoji based on sentiment
        if sentiment_score > 0:
            s_color = "#059669"
            s_emoji = "📈"
        elif sentiment_score < 0:
            s_color = "#DC2626"
            s_emoji = "📉"
        else:
            s_color = "#6B7280"
            s_emoji = "➖"
        with c4: st.markdown(stat_card("Overall Sentiment", f"{sentiment_score:.2f}", s_color, s_emoji), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # SOURCE LIST - Enhanced with video titles, URLs and comment counts
    with st.expander("📋 VIDEO KAYNAK LİSTESİ", expanded=True):
        st.markdown("""
        <div style='margin-bottom: 12px;'>
            <span style='color: #6B7280; font-size: 0.85rem;'>Analiz edilen videoların listesi ve yorum sayıları:</span>
        </div>
        """, unsafe_allow_html=True)
        
        for i, v in enumerate(videos, 1):
            video_title = v.get('baslik', 'Video')[:60] + ('...' if len(v.get('baslik', '')) > 60 else '')
            comment_count = len(v.get('yorumlar', []))
            video_url = v.get('url', v.get('video_url', '#'))
            
            st.markdown(f"""
            <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 4px rgba(0,0,0,0.04);'>
                <div style='flex: 1;'>
                    <div style='display: flex; align-items: center; gap: 8px;'>
                        <span style='background: #4169E1; color: white; font-size: 0.75rem; font-weight: 600; padding: 3px 8px; border-radius: 4px;'>{i}</span>
                        <a href='{video_url}' target='_blank' style='color: #1F2937; font-weight: 600; font-size: 0.95rem; text-decoration: none; hover: color: #4169E1;'>
                            {video_title}
                        </a>
                    </div>
                </div>
                <div style='background: #F0F9FF; color: #4169E1; padding: 6px 12px; border-radius: 16px; font-weight: 600; font-size: 0.85rem; white-space: nowrap;'>
                    💬 {comment_count} yorum
                </div>
            </div>
            """, unsafe_allow_html=True)

    display_tabs(all_comments, st.session_state.multi_video_sentiment, "Multi-Video Analysis")


def display_tabs(comments, sentiment_results, title_context):
    tabs = st.tabs(["DASHBOARD", "TIMELINE", "DATA FEED", "WORD CLOUD", "AI SUMMARY"])
    
    with tabs[0]:
        if sentiment_results:
            c1, c2 = st.columns(2)
            dist = SentimentAnalyzer().get_sentiment_distribution(sentiment_results)
            
            with c1:
                st.markdown("**Duygu Dağılımı**")
                st.plotly_chart(create_sentiment_pie_chart(dist['positive'], dist['negative'], dist['neutral']), use_container_width=True)
            
            with c2:
                stats = SentimentAnalyzer().get_summary_stats(sentiment_results)
                st.markdown("**Duygu Endeksi**")
                st.plotly_chart(create_engagement_gauge(stats['sentiment_score']), use_container_width=True)
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
                
                # Mini stacked bar - Light Mode colors
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Bar(
                    x=[dist['positive']], y=[''], orientation='h', 
                    marker_color='#059669', name='Pos', hoverinfo='x'
                ))
                fig_mini.add_trace(go.Bar(
                    x=[dist['negative']], y=[''], orientation='h', 
                    marker_color='#DC2626', name='Neg', hoverinfo='x'
                ))
                fig_mini.add_trace(go.Bar(
                    x=[dist['neutral']], y=[''], orientation='h', 
                    marker_color='#6B7280', name='Neu', hoverinfo='x'
                ))
                fig_mini.update_layout(
                    barmode='stack', 
                    height=40, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    paper_bgcolor='rgba(255,255,255,0)',
                    plot_bgcolor='rgba(255,255,255,0)',
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False, showticklabels=False)
                )
                st.plotly_chart(fig_mini, key="mini_chart", config={'displayModeBar': False})
                st.caption("Page Sentiment Distribution")

        st.markdown("---")

        # 3. Comments List (Light Mode Styled Cards)
        for c in current_batch:
            # Clean comment text from any HTML artifacts
            comment_text = c.get('metin', '')
            if isinstance(comment_text, str):
                # Remove any HTML artifacts that might leak through
                comment_text = comment_text.replace('</div>', '').replace('<div>', '')
                comment_text = comment_text.replace('</span>', '').replace('<span>', '')
                comment_text = comment_text.strip()
            
            # Try to find corresponding sentiment if available
            sentiment_color = "#6B7280"  # Default gray
            if sentiment_results:
                # For future: could attach sentiment to each comment
                pass

            st.markdown(f"""
            <div class="comment-card" style="background: #FFFFFF; padding: 18px 22px; margin-bottom: 14px; border-left: 4px solid #4169E1; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.06);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: #4169E1; font-weight: 600; font-size: 0.95rem;">👤 {c.get('yazar', 'User')}</span>
                    <span style="background: #F3F4F6; padding: 4px 10px; border-radius: 16px; font-size: 0.8rem; color: #6B7280; border: 1px solid rgba(0,0,0,0.06);">
                        ❤️ {c.get('begeni', 0)}
                    </span>
                </div>
                <div style="color: #1F2937; font-size: 0.95rem; line-height: 1.7;">
                    {comment_text}
                </div>
                {f'<div style="margin-top:12px; font-size:0.8rem; color:#6B7280; display: flex; align-items: center; gap: 6px;"><span>📺</span> {c["_video_title"]}</div>' if '_video_title' in c else ''}
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
            # Clean the summary text (remove any trailing </div> if present)
            summary_text = st.session_state[summary_key]
            if isinstance(summary_text, str):
                # Remove any trailing HTML artifacts
                summary_text = summary_text.replace('</div>', '').strip()
            
            # Render as a styled report card container
            st.markdown(f"""
            <div style="background: #F8F9FA; 
                        border-left: 5px solid #4A90E2; 
                        border-radius: 10px; 
                        padding: 20px 24px; 
                        margin: 16px 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                        border: 1px solid #E5E7EB;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #E5E7EB;">
                    <span style="font-size: 1.4rem;">🤖</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #1F2937;">AI Analiz Raporu</span>
                </div>
                <div style="color: #374151; font-size: 0.95rem; line-height: 1.9;">
                    {summary_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Özeti Temizle", key="btn_clr_summary"):
                st.session_state[summary_key] = None
                st.rerun()


def page_battle():
    # Clean header - Light Mode
    st.markdown("""
    <div style='margin-bottom: 32px;'>
        <h2 style='font-size: 1.8rem; margin: 0; color: #1F2937;'>Competitive Battle</h2>
        <p style='color: #6B7280; font-size: 0.9rem; margin-top: 4px;'>Compare two videos using AI-powered evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- STEP 1: Video URL Inputs ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
        <span style='background: #4169E1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>1</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #1F2937;'>Source Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    u1 = c1.text_input("Competitor A", key="battle_url1", placeholder="https://youtube.com/watch?v=...")
    u2 = c2.text_input("Competitor B", key="battle_url2", placeholder="https://youtube.com/watch?v=...")
    
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    
    # --- STEP 2: Evaluation Criteria ---
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
        <span style='background: #6366F1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>2</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #1F2937;'>Evaluation Criteria</span>
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
                
                v1_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v1_data.get('yorumlar', [])]
                v2_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v2_data.get('yorumlar', [])]
                
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
        st.markdown("## Comparison Results")
        
        if result.winner != "Berabere":
            st.success(f"**Leader: {result.winner}**")
        
        c1, c2 = st.columns(2)
        c1.metric(v1.get('baslik', 'Video 1')[:25], f"{result.video1_total_comments} comments")
        c2.metric(v2.get('baslik', 'Video 2')[:25], f"{result.video2_total_comments} comments")
        
        # ========== CLASSIFICATION SUMMARY TABLE ==========
        st.markdown("### Classification Summary")
        st.caption("Her kategori için kaç yorum 1 (uygun) veya 0 (uygun değil) olarak sınıflandırıldı")
        
        if result.v1_classifications and result.v2_classifications:
            # Get category names from first classification (excluding 'yorum' field)
            categories = [k for k in result.v1_classifications[0].keys() if k != 'yorum']
            
            if categories:
                # Build summary data
                summary_data = []
                for cat in categories:
                    v1_ones = sum(1 for c in result.v1_classifications if c.get(cat, 0) == 1)
                    v1_zeros = sum(1 for c in result.v1_classifications if c.get(cat, 0) == 0)
                    v2_ones = sum(1 for c in result.v2_classifications if c.get(cat, 0) == 1)
                    v2_zeros = sum(1 for c in result.v2_classifications if c.get(cat, 0) == 0)
                    
                    summary_data.append({
                        'Kategori': cat,
                        'V1 ✓ (1)': v1_ones,
                        'V1 ✗ (0)': v1_zeros,
                        'V1 %': f"{v1_ones/(v1_ones+v1_zeros)*100:.1f}%" if (v1_ones+v1_zeros) > 0 else "0%",
                        'V2 ✓ (1)': v2_ones,
                        'V2 ✗ (0)': v2_zeros,
                        'V2 %': f"{v2_ones/(v2_ones+v2_zeros)*100:.1f}%" if (v2_ones+v2_zeros) > 0 else "0%"
                    })
                
                df_summary = pd.DataFrame(summary_data)
                
                # Style the dataframe
                st.dataframe(
                    df_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Kategori': st.column_config.TextColumn('Kategori', width='medium'),
                        'V1 ✓ (1)': st.column_config.NumberColumn('V1 ✓', help=f'{v1.get("baslik", "Video 1")[:20]} - Uygun'),
                        'V1 ✗ (0)': st.column_config.NumberColumn('V1 ✗', help=f'{v1.get("baslik", "Video 1")[:20]} - Uygun Değil'),
                        'V1 %': st.column_config.TextColumn('V1 %', width='small'),
                        'V2 ✓ (1)': st.column_config.NumberColumn('V2 ✓', help=f'{v2.get("baslik", "Video 2")[:20]} - Uygun'),
                        'V2 ✗ (0)': st.column_config.NumberColumn('V2 ✗', help=f'{v2.get("baslik", "Video 2")[:20]} - Uygun Değil'),
                        'V2 %': st.column_config.TextColumn('V2 %', width='small'),
                    }
                )
                
                st.markdown(f"""
                <div style='display: flex; gap: 20px; margin-top: 8px;'>
                    <span style='color: #4A90E2; font-size: 0.85rem; font-weight: 500;'>Video A = {v1.get('baslik', 'Video 1')[:30]}</span>
                    <span style='color: #6C5CE7; font-size: 0.85rem; font-weight: 500;'>Video B = {v2.get('baslik', 'Video 2')[:30]}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sample size warning - no emojis (enterprise style)
        min_comments = min(result.video1_total_comments, result.video2_total_comments)
        if min_comments < 10:
            st.warning(f"**Low Sample Warning:** The video with fewest comments has only {min_comments} comments. Minimum 20+ comments recommended for reliable analysis.")
        elif min_comments < 20:
            st.info(f"**Note:** Sample size ({min_comments} comments) is moderate. Interpret results carefully.")
        
        st.markdown("### Category Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Side-by-Side", "Kategori Pasta", "Zaman Trendi", "Heatmap"])
        
        with tab1:
            fig = create_category_comparison_chart(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Her kategori için kaç yorum o kategoriye uyduğu (eşleşme oranı %). Daha yüksek = o konuda daha çok konuşulmuş.")
        
        with tab2:
            # Category Pie Charts - showing match distribution for each category
            fig = create_category_pie_grid(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Her kategori için eşleşen ve diğer yorum oranları.")
        
        with tab3:
            # Category Temporal Charts - sentiment over time for each category
            v1_comments_raw = v1.get('yorumlar', [])
            v2_comments_raw = v2.get('yorumlar', [])
            v1_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v1_comments_raw]
            v2_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v2_comments_raw]
            
            # Get sentiment for both videos
            from sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            with st.spinner("Duygu analizi yapılıyor..."):
                v1_sentiments = analyzer.analyze_batch(v1_comments[:100])
                v2_sentiments = analyzer.analyze_batch(v2_comments[:100])
            
            fig = create_category_temporal_chart(
                result.categories,
                v1_comments_raw[:100],
                v2_comments_raw[:100],
                v1_sentiments,
                v2_sentiments,
                v1.get('baslik', 'Video 1'),
                v2.get('baslik', 'Video 2')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Her kategori için zamana bağlı pozitif ve negatif yorum sayıları.")
        
        with tab4:
            fig = create_category_heatmap(result.categories, v1.get('baslik', 'Video 1'), v2.get('baslik', 'Video 2'))
            st.plotly_chart(fig, use_container_width=True)
        
        # ========== LLM COMPARISON SUMMARIES - SESSION STATE BASED ==========
        st.markdown("### AI Comparison Summary")
        st.caption("AI-generated comment analysis summary for each video")
        
        # Initialize summary keys in session state
        summary_key_v1 = "battle_summary_v1"
        summary_key_v2 = "battle_summary_v2"
        
        if summary_key_v1 not in st.session_state:
            st.session_state[summary_key_v1] = None
        if summary_key_v2 not in st.session_state:
            st.session_state[summary_key_v2] = None
        
        # Video info cards
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            v1_name = v1.get('baslik', 'Video 1')[:35]
            st.markdown(f"""
            <div style='background: #F9FAFB; border-left: 4px solid #4A90E2; border-radius: 8px; padding: 16px; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>
                <div style='font-weight: 600; color: #1F2937; font-size: 0.95rem; margin-bottom: 8px;'>Video A: {v1_name}</div>
                <div style='color: #6B7280; font-size: 0.8rem;'>{result.video1_total_comments} comments analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_v2:
            v2_name = v2.get('baslik', 'Video 2')[:35]
            st.markdown(f"""
            <div style='background: #F9FAFB; border-left: 4px solid #6C5CE7; border-radius: 8px; padding: 16px; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>
                <div style='font-weight: 600; color: #1F2937; font-size: 0.95rem; margin-bottom: 8px;'>Video B: {v2_name}</div>
                <div style='color: #6B7280; font-size: 0.8rem;'>{result.video2_total_comments} comments analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate summaries ONLY if not already in session state
        if st.session_state[summary_key_v1] is None or st.session_state[summary_key_v2] is None:
            with st.spinner("Generating AI summary..."):
                try:
                    from ollama_llm import OllamaLLM
                    summarizer = OllamaLLM()
                    
                    v1_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v1.get('yorumlar', [])[:30]]
                    v2_comments = [c.get('metin_duygu') or c.get('metin', '') for c in v2.get('yorumlar', [])[:30]]
                    
                    # Get individual summaries using BATTLE MODE specific summarizer
                    v1_raw = summarizer.summarize_for_battle(v1_comments, v1.get('baslik', 'Video 1'))
                    v2_raw = summarizer.summarize_for_battle(v2_comments, v2.get('baslik', 'Video 2'))
                    
                    # Robust cleaning function
                    def clean_ai_out(text):
                        if not text: return "Summary not available."
                        # Clean HTML tags and artifacts that might leak from Ollama
                        return text.replace("</div>", "").replace("</p>", "").replace("<p>", "").replace("```", "").strip()
                    
                    st.session_state[summary_key_v1] = clean_ai_out(v1_raw)
                    st.session_state[summary_key_v2] = clean_ai_out(v2_raw)
                    
                except Exception as e:
                    st.warning(f"AI summary generation failed: {e}")
                    st.info("Check Ollama connection: ollama serve")
                    st.session_state[summary_key_v1] = "Summary generation failed."
                    st.session_state[summary_key_v2] = "Summary generation failed."
        
        # Display summaries from session state (persists through reruns)
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            summary_v1 = st.session_state.get(summary_key_v1, "Summary not available.")
            st.markdown(f"""
            <div style='background: #FFFFFF; border-radius: 8px; padding: 16px; margin-top: 12px; border-left: 4px solid #4A90E2; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>
                <p style='color: #374151; font-size: 0.9rem; line-height: 1.7; margin: 0;'>
                    {summary_v1}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            summary_v2 = st.session_state.get(summary_key_v2, "Summary not available.")
            st.markdown(f"""
            <div style='background: #FFFFFF; border-radius: 8px; padding: 16px; margin-top: 12px; border-left: 4px solid #6C5CE7; border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>
                <p style='color: #374151; font-size: 0.9rem; line-height: 1.7; margin: 0;'>
                    {summary_v2}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall Comparison Result - High Contrast for Readability
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Comparison Verdict")
        comparison_text = result.summary if result.summary else "Comparison result not available."
        st.markdown(f"""
        <div style='background: #D1FAE5; border-left: 4px solid #059669; border-radius: 8px; padding: 16px 20px; 
                    border: 1px solid #A7F3D0; box-shadow: 0 1px 4px rgba(0,0,0,0.04);'>
            <p style='color: #155724; font-size: 0.95rem; line-height: 1.8; margin: 0; font-weight: 500;'>
                {comparison_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        # ========== EXPORT DATA SECTION ==========
        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
        st.markdown("### Export Data")
        
        tab_v1, tab_v2 = st.tabs([f"Video A: {v1.get('baslik', 'Video 1')[:25]}", f"Video B: {v2.get('baslik', 'Video 2')[:25]}"])
        
        with tab_v1:
            if result.v1_classifications:
                df_v1 = pd.DataFrame(result.v1_classifications)
                st.dataframe(df_v1, use_container_width=True, height=250)
                
                # Download buttons for Video 1
                col_dl1, col_dl2, col_spacer = st.columns([1, 1, 2])
                with col_dl1:
                    csv_v1 = df_v1.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_v1,
                        file_name=f"video_a_analysis.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True
                    )
                with col_dl2:
                    # Excel download
                    import io
                    buffer = io.BytesIO()
                    df_v1.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"video_a_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True
                    )
        
        with tab_v2:
            if result.v2_classifications:
                df_v2 = pd.DataFrame(result.v2_classifications)
                st.dataframe(df_v2, use_container_width=True, height=250)
                
                # Download buttons for Video 2
                col_dl1, col_dl2, col_spacer = st.columns([1, 1, 2])
                with col_dl1:
                    csv_v2 = df_v2.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_v2,
                        file_name=f"video_b_analysis.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True
                    )
                with col_dl2:
                    import io
                    buffer = io.BytesIO()
                    df_v2.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"video_b_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True
                    )
        
        # New Comparison Button - Centered and prominent
        st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("Start New Comparison", type="primary", use_container_width=True):
                # Clear all battle-related session state
                st.session_state.battle_result = None
                st.session_state.battle_video1 = None
                st.session_state.battle_video2 = None
                # Also clear AI summaries
                if 'battle_summary_v1' in st.session_state:
                    del st.session_state['battle_summary_v1']
                if 'battle_summary_v2' in st.session_state:
                    del st.session_state['battle_summary_v2']
                st.rerun()


def page_stats():
    st.title("İleri Düzey Metin Madenciliği")
    
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
        <h2 style='font-size: 1.6rem; margin: 0; color: #1F2937;'>{title_text}</h2>
        <p style='color: #6B7280; font-size: 0.85rem; margin-top: 4px;'>Toplam {len(all_comments)} yorum analiz ediliyor</p>
    </div>
    """, unsafe_allow_html=True)
    
    import numpy as np
    import re
    from collections import Counter
    
    texts = [c.get('metin', '') for c in all_comments]
    comment_lengths = [len(t) for t in texts]
    likes = [c.get('begeni', 0) for c in all_comments]
    
    # === GENEL METRİKLER - Light Mode Styled Cards ===
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 16px 0 12px 0;'>
        <span style='background: #4169E1; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;'>📊</span>
        <span style='font-size: 1.1rem; font-weight: 600; color: #1F2937;'>Genel Metrikler</span>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    def stat_box(label, value, icon, color):
        return f"""
        <div style='background: #FFFFFF; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #E5E7EB;'>
            <div style='color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500;'>{icon} {label}</div>
            <div style='color: #1F2937; font-size: 1.5rem; font-weight: 700; margin-top: 4px;'>{value}</div>
        </div>
        """
    
    with m1: st.markdown(stat_box("Toplam Yorum", f"{len(all_comments):,}", "💬", "#4169E1"), unsafe_allow_html=True)
    with m2: st.markdown(stat_box("Toplam Beğeni", f"{sum(likes):,}", "❤️", "#DC2626"), unsafe_allow_html=True)
    with m3: st.markdown(stat_box("Ort. Uzunluk", f"{np.mean(comment_lengths):.0f} char", "📏", "#059669"), unsafe_allow_html=True)
    if sentiment:
        stats = SentimentAnalyzer().get_summary_stats(sentiment)
        score = stats['sentiment_score']
        s_color = "#059669" if score > 0 else "#DC2626" if score < 0 else "#6B7280"
        s_emoji = "📈" if score > 0 else "📉" if score < 0 else "➖"
        with m4: st.markdown(stat_box("Duygu Skoru", f"{s_emoji} {score:.2f}", "🎭", s_color), unsafe_allow_html=True)
    else:
        with m4: st.markdown(stat_box("Duygu Skoru", "N/A", "🎭", "#6B7280"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 1. Bİ-GRAM ANALİZİ (Light Mode) ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #4169E1, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>1️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; color: #1F2937;'>Bi-Gram Analizi (İkili Kelime Öbekleri)</span>
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
                
                # Royal Blue gradient for Light Mode
                blue_shades = ['#4169E1', '#5A7FE8', '#7395EF', '#8CABF6', '#A5C1FD', '#BED7FF']
                colors = (blue_shades * 3)[:len(bigrams)]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=bigrams[::-1],
                    x=counts[::-1],
                    orientation='h',
                    marker=dict(
                        color=colors[::-1],
                        line=dict(width=1, color='rgba(255,255,255,0.8)')
                    ),
                    text=[f" {c}" for c in counts[::-1]],
                    textposition='outside',
                    textfont=dict(size=11, color='#333333'),
                    hovertemplate='<b>%{y}</b><br>Kullanım: %{x}<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis_title="Kullanım Sayısı",
                    yaxis_title="",
                    paper_bgcolor='rgba(255,255,255,0)',
                    plot_bgcolor='rgba(255,255,255,0)',
                    font=dict(color='#333333', size=12),
                    height=420,
                    margin=dict(l=20, r=60, t=10, b=40),
                    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#4169E1", font_color="#333333")
                )
                fig.update_xaxes(gridcolor='#EAEAEA', showgrid=True, zeroline=False, tickfont=dict(color='#666666'))
                fig.update_yaxes(showgrid=False, tickfont=dict(color='#333333'))
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Yeterli bi-gram verisi yok.")
        else:
            st.info("Yeterli metin verisi yok.")
    except Exception as e:
        st.warning(f"Bi-gram analizi yapılamadı: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 2. ETKİLEŞİM MATRİSİ (Bubble Chart) - Light Mode ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #4169E1, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>2️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; color: #1F2937;'>Etkileşim Matrisi (Duygu × Beğeni × Uzunluk)</span>
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
                paper_bgcolor='rgba(255,255,255,0)',
                plot_bgcolor='rgba(255,255,255,0)',
                font=dict(color='#333333', size=12),
                height=450,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, bgcolor='rgba(255,255,255,0)', font=dict(color='#333333')
                ),
                margin=dict(l=20, r=20, t=50, b=40),
                hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#4169E1", font_color="#333333")
            )
            fig.update_xaxes(gridcolor='#EAEAEA', showgrid=True, zeroline=True, zerolinecolor='#CCCCCC', tickfont=dict(color='#666666'))
            fig.update_yaxes(gridcolor='#EAEAEA', showgrid=True, zeroline=False, tickfont=dict(color='#666666'))
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Duygu analizi verisi gerekli. Önce bir video analiz edin.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 3. EN ETKİLİ YORUMCULAR - Light Mode ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #4169E1, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>3️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; color: #1F2937;'>En Etkili Yorumcular</span>
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
            st.markdown("**Top 10 by Total Likes**")
            authors = [a[0][:15] + '...' if len(a[0]) > 15 else a[0] for a in sorted_authors]
            total_likes = [a[1]['total_likes'] for a in sorted_authors]
            
            # Blue gradient for Light Mode
            blue_shades = ['#4169E1', '#5A7FE8', '#7395EF', '#8CABF6', '#A5C1FD', '#BED7FF', '#D6E8FF', '#EEF4FF', '#F8FBFF', '#FFFFFF']
            colors = blue_shades[:len(authors)]
            
            # --- KRİTİK DÜZELTME: DİNAMİK YÜKSEKLİK ---
            # Her çubuk için 40px + Eksenler için 30px pay bırakıyoruz.
            # Böylece grafik asla gereğinden fazla yer kaplamıyor ve yukarı yapışıyor.
            dynamic_height = (len(authors) * 40) + 30
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=authors[::-1],
                x=total_likes[::-1],
                orientation='h',
                marker=dict(
                    color=colors[::-1],
                    line=dict(width=1, color='rgba(255,255,255,0.8)')
                ),
                text=[f" {l:,}" for l in total_likes[::-1]],
                textposition='outside',
                textfont=dict(size=11, color='#333333'),
                hovertemplate='<b>%{y}</b><br>Total Likes: %{x:,}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Total Likes",
                yaxis_title="",
                paper_bgcolor='rgba(255,255,255,0)',
                plot_bgcolor='rgba(255,255,255,0)',
                font=dict(color='#333333', size=12),
                
                # Sabit 350 yerine hesapladığımız boyutu veriyoruz
                height=dynamic_height,
                
                # margin-top (t) değerini 0 yaptık, sol (l) boşluğu da kıstık
                margin=dict(l=0, r=50, t=0, b=30),
                
                hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#4169E1", font_color="#333333"),
                
                # Barların kalınlığını ve aralığını ayarlayarak daha sıkı durmasını sağlıyoruz
                bargap=0.2
            )
            fig.update_xaxes(gridcolor='#EAEAEA', showgrid=True, zeroline=False, tickfont=dict(color='#666666'))
            fig.update_yaxes(showgrid=False, tickfont=dict(color='#333333'))
            
            # use_container_width=True ile sütuna tam oturmasını sağlıyoruz
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col_table:
            st.markdown("**Most Popular Comments**")
            
            # Find the most liked comment for each top author
            top_authors_list = [a[0] for a in sorted_authors[:5]]
            
            # Get best comments for top authors
            for author_name in top_authors_list:
                # Find all comments by this author
                author_comments = [c for c in all_comments if c.get('yazar', '') == author_name]
                
                if author_comments:
                    # Get the most liked comment
                    best_comment = max(author_comments, key=lambda x: x.get('begeni', 0))
                    comment_text = best_comment.get('metin', '')
                    likes = best_comment.get('begeni', 0)
                    
                    # Clean and truncate comment text
                    if isinstance(comment_text, str):
                        comment_text = comment_text.replace('</div>', '').replace('<div>', '').strip()
                        if len(comment_text) > 150:
                            comment_text = comment_text[:147] + "..."
                    
                    # Truncate author name
                    display_name = author_name[:20] + ("..." if len(author_name) > 20 else "")
                    
                    st.markdown(f"""
                    <div style='background: #FFFFFF; border: 1px solid #E5E7EB; 
                                padding: 14px 16px; border-radius: 10px; margin-bottom: 10px;
                                box-shadow: 0 1px 4px rgba(0,0,0,0.04);'>
                        <div style='color: #4169E1; font-weight: 600; font-size: 0.9rem; margin-bottom: 8px;'>
                            @{display_name}
                        </div>
                        <div style='color: #4B5563; font-size: 0.85rem; font-style: italic; line-height: 1.6;'>
                            "{comment_text}"
                        </div>
                        <div style='text-align: right; margin-top: 8px;'>
                            <span style='background: #FEE2E2; color: #DC2626; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>
                                ❤️ {likes:,}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Yeterli yazar verisi yok.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    # ========== 4. SORU ANALİZİ (Donut Chart) - Light Mode ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #4169E1, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>4️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; color: #1F2937;'>Soru Analizi</span>
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
                colors=['#D97706', '#4169E1'],
                line=dict(width=2, color='#FFFFFF')
            ),
            textinfo='percent+label',
            textposition='outside',
            textfont=dict(size=13, color='#333333'),
            hovertemplate='<b>%{label}</b><br>Sayı: %{value}<br>Oran: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0)',
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(
                text=f"<b>{question_count}</b><br><span style='font-size:12px'>Soru</span>",
                x=0.5, y=0.5, font_size=24, font_color='#D97706', showarrow=False
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        question_pct = (question_count / len(all_comments) * 100) if len(all_comments) > 0 else 0
        
        st.markdown(f"""
        <div style='background: #FEF3C7; border: 1px solid rgba(217, 119, 6, 0.3); padding: 24px; border-radius: 12px; margin-top: 20px;'>
            <div style='color: #6B7280; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px;'>Soru Oranı</div>
            <div style='color: #D97706; font-size: 2.8rem; font-weight: 800; line-height: 1;'>{question_pct:.1f}%</div>
            <div style='color: #4B5563; font-size: 0.8rem; margin-top: 12px;'>
                {question_count} soru / {len(all_comments)} yorum
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Insight - Light Mode info box
        if question_pct > 30:
            insight = "⚠️ Kitle çok soru soruyor! Daha fazla açıklama veya FAQ gerekebilir."
            insight_color = "#D97706"
            box_bg = "#FEF3C7"
        elif question_pct > 15:
            insight = "💡 Orta düzeyde soru var. İçeriğiniz anlaşılır görünüyor."
            insight_color = "#4169E1"
            box_bg = "#E3F2FD"
        else:
            insight = "✅ Çok az soru var. İçeriğiniz açık ve net!"
            insight_color = "#059669"
            box_bg = "#D1FAE5"
        
        st.markdown(f"""
        <div style='background: {box_bg}; border-left: 4px solid {insight_color}; 
                    padding: 14px 18px; margin-top: 16px; border-radius: 8px;'>
            <span style='color: #1F2937; font-size: 0.9rem;'>{insight}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== 5. ZAMANA BAĞLI DUYGU DEĞİŞİMİ - Light Mode ==========
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px; margin: 24px 0 16px 0;'>
        <span style='background: linear-gradient(135deg, #4169E1, #6366F1); color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;'>5️⃣</span>
        <span style='font-size: 1.3rem; font-weight: 700; color: #1F2937;'>Zamana Bağlı Duygu Analizi</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("📅 Yorum tarihlerine göre duygu değişimi (scatter plot + trend çizgisi)")
    
    if sentiment and len(all_comments) > 0:
        try:
            # Check if comments have timestamps
            has_timestamps = any(c.get('timestamp', 0) not in [0, None, ''] for c in all_comments)
            
            if has_timestamps:
                fig = create_temporal_sentiment_chart(all_comments, sentiment)
                st.plotly_chart(fig, use_container_width=True)
                
                # Insight based on trend - Light Mode
                pos_count = sum(1 for s in sentiment if hasattr(s, 'label') and s.label == 'positive')
                neg_count = sum(1 for s in sentiment if hasattr(s, 'label') and s.label == 'negative')
                total = len(sentiment)
                
                if pos_count > neg_count * 1.5:
                    insight = f"📈 Genel trend POZİTİF: {pos_count}/{total} ({pos_count/total*100:.0f}%) yorum olumlu."
                    insight_color = "#059669"
                    box_bg = "#D1FAE5"
                elif neg_count > pos_count * 1.5:
                    insight = f"📉 Genel trend NEGATİF: {neg_count}/{total} ({neg_count/total*100:.0f}%) yorum olumsuz."
                    insight_color = "#DC2626"
                    box_bg = "#FEE2E2"
                else:
                    insight = f"↔️ Dengeli trend: Pozitif {pos_count} (%{pos_count/total*100:.0f}), Negatif {neg_count} (%{neg_count/total*100:.0f})"
                    insight_color = "#4169E1"
                    box_bg = "#E3F2FD"
                
                st.markdown(f"""
                <div style='background: {box_bg}; border-left: 4px solid {insight_color}; 
                            padding: 14px 18px; margin-top: 16px; border-radius: 8px;'>
                    <span style='color: #1F2937; font-size: 0.9rem;'>{insight}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⚠️ Yorumlarda tarih bilgisi bulunamadı. Yeni bir video analizi yaparak tarihli yorumları çekebilirsiniz.")
                
        except Exception as e:
            st.warning(f"Temporal grafik oluşturulamadı: {e}")
    else:
        st.warning("Temporal analiz için önce duygu analizi gerekli. Bir video analiz edin.")


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
