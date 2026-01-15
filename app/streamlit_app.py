#!/usr/bin/env python3
"""
Big Five Personality Analyzer - Streamlit Demo
A comprehensive web mining capstone project demo for detecting 
Big Five personality traits from social media posts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    TRAIT_NAMES,
    TRAIT_COLS,
    TOP_K_EVIDENCE,
    TOP_K_RECS,
)
from src.utils.text import preprocess_tweets
from src.models.tfidf_ridge import TfidfRidgeModel
from src.ir.bm25 import BM25Index
from src.ir.evidence import TRAIT_QUERIES
from src.recsys.hashtag_recsys import HashtagRecommender

# Page config
st.set_page_config(
    page_title="Big Five Personality Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #a855f7;
        --background-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.8);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: gradient 3s ease infinite;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Trait cards */
    .trait-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        padding: 1.2rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .trait-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
    }
    
    .trait-name {
        font-weight: 700;
        font-size: 1.1rem;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    
    .trait-score {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Evidence cards */
    .evidence-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .evidence-card.open { border-left-color: #f59e0b; }
    .evidence-card.conscientious { border-left-color: #10b981; }
    .evidence-card.extroverted { border-left-color: #f43f5e; }
    .evidence-card.agreeable { border-left-color: #3b82f6; }
    .evidence-card.stable { border-left-color: #8b5cf6; }
    
    .evidence-text {
        color: #cbd5e1;
        font-style: italic;
        line-height: 1.6;
    }
    
    .evidence-score {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Hashtag pills */
    .hashtag-pill {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: transform 0.2s ease;
    }
    
    .hashtag-pill:hover {
        transform: scale(1.05);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(145deg, #1e3a5f, #0f172a);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 1rem 0;
    }
    
    /* Custom progress bar */
    .progress-container {
        background: #1e293b;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Card container */
    .card-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animation */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Trait colors for visualization
TRAIT_COLORS = {
    "open": "#f59e0b",
    "conscientious": "#10b981",
    "extroverted": "#f43f5e",
    "agreeable": "#3b82f6",
    "stable": "#8b5cf6",
}

TRAIT_DESCRIPTIONS = {
    "open": "Creativity, curiosity, openness to new experiences",
    "conscientious": "Organization, responsibility, goal-oriented behavior", 
    "extroverted": "Sociability, energy, positive emotions",
    "agreeable": "Cooperation, empathy, trust in others",
    "stable": "Emotional stability, calmness under stress",
}

TRAIT_ICONS = {
    "open": "🎨",
    "conscientious": "📋",
    "extroverted": "🎉",
    "agreeable": "🤝",
    "stable": "🧘",
}


@st.cache_resource
def load_tfidf_model():
    """Load TF-IDF + Ridge model."""
    # Try multiple possible paths
    model_paths = [
        MODELS_DIR / "baseline.joblib",
        MODELS_DIR / "baseline_en.joblib",
        MODELS_DIR / "tfidf_ridge.pkl",
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                return TfidfRidgeModel.load(path)
            except Exception:
                continue
    return None


@st.cache_resource
def load_recommender():
    """Load hashtag recommender."""
    try:
        # Try loading from pickle first
        data_path = PROCESSED_DIR / "pan15_en.pkl"
        if data_path.exists():
            df = pd.read_pickle(data_path)
        else:
            data_path = PROCESSED_DIR / "pan15_en.parquet"
            if data_path.exists():
                df = pd.read_parquet(data_path)
            else:
                return None
        
        recommender = HashtagRecommender()
        recommender.fit(df)
        return recommender
    except Exception as e:
        st.warning(f"Could not load recommender: {e}")
        return None


def create_radar_chart(traits: dict):
    """Create an interactive radar chart using Plotly."""
    categories = [TRAIT_ICONS[t] + " " + t.capitalize() for t in traits.keys()]
    values = list(traits.values())
    colors = [TRAIT_COLORS[t] for t in traits.keys()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=10, color=colors + [colors[0]]),
        name='Your Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#e2e8f0'),
                gridcolor='rgba(148, 163, 184, 0.2)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=40, b=40),
        height=400,
    )
    
    return fig


def create_bar_chart(traits: dict):
    """Create a horizontal bar chart for trait scores."""
    df = pd.DataFrame([
        {"trait": TRAIT_ICONS[t] + " " + t.capitalize(), "score": s, "color": TRAIT_COLORS[t]}
        for t, s in traits.items()
    ])
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['trait']],
            x=[row['score']],
            orientation='h',
            marker=dict(
                color=row['color'],
                line=dict(color='rgba(255,255,255,0.3)', width=1),
            ),
            text=[f"{row['score']:.2f}"],
            textposition='inside',
            textfont=dict(color='white', size=14, family='Arial Black'),
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8'),
        ),
        yaxis=dict(
            tickfont=dict(color='#e2e8f0', size=12),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=20, t=20, b=40),
        height=250,
        bargap=0.3,
    )
    
    return fig


def build_temp_index(tweets: list) -> BM25Index:
    """Build temporary BM25 index for evidence retrieval."""
    documents = [
        {"doc_id": f"temp_{i}", "user_id": "temp_user", "text": t, "tweet_idx": i}
        for i, t in enumerate(tweets)
    ]
    index = BM25Index()
    index.build(documents, text_key="text")
    return index


def retrieve_temp_evidence(index: BM25Index, top_k: int = 5) -> dict:
    """Retrieve evidence for each trait."""
    evidence = {}
    for trait in TRAIT_NAMES:
        query = TRAIT_QUERIES.get(trait, trait)
        results = index.search(query, top_k=top_k, user_id="temp_user")
        evidence[trait] = [
            {"tweet": doc["text"], "score": score}
            for doc, score in results
        ]
    return evidence


def display_trait_card(trait: str, score: float):
    """Display a trait card with visual elements."""
    icon = TRAIT_ICONS[trait]
    color = TRAIT_COLORS[trait]
    description = TRAIT_DESCRIPTIONS[trait]
    
    # Determine level
    if score >= 0.7:
        level = "High"
        level_color = "#10b981"
    elif score >= 0.4:
        level = "Moderate"
        level_color = "#f59e0b"
    else:
        level = "Low"
        level_color = "#ef4444"
    
    st.markdown(f"""
    <div class="trait-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.5rem;">{icon}</span>
                <span class="trait-name" style="margin-left: 0.5rem;">{trait.capitalize()}</span>
            </div>
            <div class="trait-score">{score:.2f}</div>
        </div>
        <div style="margin-top: 0.5rem;">
            <div class="progress-container">
                <div class="progress-fill" style="width: {score*100}%; background: linear-gradient(90deg, {color}, {color}dd);"></div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
            <span style="color: #64748b; font-size: 0.85rem;">{description}</span>
            <span style="color: {level_color}; font-weight: 600; font-size: 0.85rem;">{level}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Big Five Personality Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze personality traits from social media posts using Machine Learning + Information Retrieval</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        top_k_evidence = st.slider(
            "📊 Evidence per trait",
            min_value=3,
            max_value=10,
            value=TOP_K_EVIDENCE,
            help="Number of example posts to show for each trait"
        )
        
        top_k_recs = st.slider(
            "🏷️ Hashtag recommendations",
            min_value=5,
            max_value=20,
            value=TOP_K_RECS,
            help="Number of hashtags to recommend"
        )
        
        st.divider()
        
        st.markdown("### 📖 About Big Five (OCEAN)")
        
        for trait in TRAIT_NAMES:
            st.markdown(f"""
            **{TRAIT_ICONS[trait]} {trait.capitalize()}**  
            <small style="color: #94a3b8;">{TRAIT_DESCRIPTIONS[trait]}</small>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1e293b, #0f172a); padding: 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
            ⚠️ <strong>Disclaimer:</strong> This is an ML prediction for educational purposes. 
            Not a clinical psychological assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Analyze Text", "📁 Upload File", "ℹ️ How It Works"])
    
    input_text = ""
    
    with tab1:
        st.markdown("### Enter social media posts")
        st.markdown("*Paste your posts below, one per line. More posts = better accuracy!*")
        
        input_text = st.text_area(
            "Posts",
            height=200,
            placeholder="Enter tweets or posts here, one per line...\n\nExample:\nJust finished reading an amazing book about philosophy! Mind = blown 🤯\nHad the best time at the party with friends tonight! 🎉\nNeed to organize my schedule for next week, feeling productive 📋\nReally appreciate everyone who helped me today, you're all amazing!\nKeeping calm and focused despite the chaos around me 🧘",
            label_visibility="collapsed",
        )
        
        # Example posts button
        if st.button("📋 Load Example Posts"):
            input_text = """Just finished reading an amazing book about philosophy! Mind = blown 🤯
Had the best time at the party with friends tonight! 🎉
Need to organize my schedule for next week, feeling productive 📋
Really appreciate everyone who helped me today, you're all amazing!
Keeping calm and focused despite the chaos around me 🧘
Trying out this new creative project, so excited to experiment!
Met some new people at the conference, great networking opportunities
Setting goals for the month and tracking my progress daily
Always here to listen if anyone needs to talk 
Meditation helped me stay centered through a stressful day"""
            st.rerun()
    
    with tab2:
        st.markdown("### Upload a file")
        st.markdown("*Supported formats: .txt (one post per line) or .csv (with 'text' column)*")
        
        uploaded_file = st.file_uploader(
            "Upload your file",
            type=["txt", "csv"],
            label_visibility="collapsed",
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                file_df = pd.read_csv(uploaded_file)
                if "text" in file_df.columns:
                    input_text = "\n".join(file_df["text"].dropna().tolist())
                else:
                    input_text = "\n".join(file_df.iloc[:, 0].dropna().tolist())
            else:
                input_text = uploaded_file.read().decode("utf-8")
            
            st.success(f"✅ Loaded {len(input_text.splitlines())} posts from file!")
    
    with tab3:
        st.markdown("""
        ### 🔬 How It Works
        
        This system uses a combination of techniques to analyze personality:
        
        **1. Text Feature Extraction (TF-IDF)**
        - Character n-grams (3-5 chars) capture writing style
        - Word n-grams (1-2 words) capture vocabulary patterns
        
        **2. Machine Learning (Ridge Regression)**
        - Trained on PAN15 Author Profiling dataset
        - Predicts 5 continuous trait scores (0-1 scale)
        
        **3. Evidence Retrieval (BM25)**
        - Finds posts most relevant to each trait
        - Uses trait-specific keyword queries
        
        **4. Hashtag Recommendation**
        - Content-based filtering using embeddings
        - Personality-aware re-ranking
        
        ### 📊 Evaluation Metrics
        | Metric | Description |
        |--------|-------------|
        | RMSE | Root Mean Squared Error per trait |
        | MAE | Mean Absolute Error per trait |
        | Pearson r | Correlation with ground truth |
        """)
    
    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button(
            "🔍 Analyze Personality",
            type="primary",
            use_container_width=True,
        )
    
    if analyze_clicked:
        if not input_text or not input_text.strip():
            st.error("⚠️ Please enter some text or upload a file first!")
            return
        
        tweets = [t.strip() for t in input_text.strip().split("\n") if t.strip()]
        
        if len(tweets) < 1:
            st.error("⚠️ Please provide at least one post.")
            return
        
        # Process tweets
        tweets = preprocess_tweets(tweets)
        
        with st.spinner("🔄 Analyzing personality traits..."):
            # Load model
            model = load_tfidf_model()
            if model is None:
                st.error("❌ Model not found. Please run training scripts first.")
                st.info("Run: `python scripts/train_eval_baseline_tfidf.py`")
                return
            
            # Predict
            text_concat = " ".join(tweets)
            predictions = model.predict(pd.Series([text_concat]))[0]
            predicted_traits = {
                trait: float(np.clip(predictions[i], 0, 1))
                for i, trait in enumerate(TRAIT_NAMES)
            }
            
            # Get evidence
            temp_index = build_temp_index(tweets)
            evidence = retrieve_temp_evidence(temp_index, top_k=top_k_evidence)
            
            # Get recommendations
            recommender = load_recommender()
            recommendations = []
            if recommender:
                try:
                    recommendations = recommender.recommend_personality_aware(
                        text_concat,
                        predicted_traits,
                        top_k=top_k_recs,
                    )
                except Exception:
                    try:
                        recommendations = recommender.recommend_popularity(top_k=top_k_recs)
                    except Exception:
                        pass
        
        # Success message
        st.success("✅ Analysis Complete!")
        
        # Stats cards
        st.markdown("---")
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(tweets)}</div>
                <div class="stats-label">Posts Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[1]:
            dominant_trait = max(predicted_traits, key=predicted_traits.get)
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{TRAIT_ICONS[dominant_trait]}</div>
                <div class="stats-label">Dominant: {dominant_trait.capitalize()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[2]:
            avg_score = np.mean(list(predicted_traits.values()))
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{avg_score:.2f}</div>
                <div class="stats-label">Average Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[3]:
            word_count = sum(len(t.split()) for t in tweets)
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{word_count}</div>
                <div class="stats-label">Words Processed</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Results in two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Personality Profile")
            
            # Radar chart
            fig_radar = create_radar_chart(predicted_traits)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Bar chart
            st.markdown("### 📈 Trait Scores")
            fig_bar = create_bar_chart(predicted_traits)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Detailed Trait Analysis")
            
            for trait in TRAIT_NAMES:
                display_trait_card(trait, predicted_traits[trait])
        
        st.markdown("---")
        
        # Evidence and Recommendations
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("### 📑 Evidence by Trait")
            st.markdown("*Posts most relevant to each personality trait:*")
            
            for trait in TRAIT_NAMES:
                with st.expander(f"{TRAIT_ICONS[trait]} **{trait.upper()}** (score: {predicted_traits[trait]:.2f})"):
                    trait_evidence = evidence.get(trait, [])
                    if trait_evidence:
                        for i, item in enumerate(trait_evidence[:3]):
                            st.markdown(f"""
                            <div class="evidence-card {trait}">
                                <div class="evidence-text">"{item['tweet']}"</div>
                                <div class="evidence-score">Relevance: {item['score']:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No strong evidence found for this trait.")
        
        with col4:
            st.markdown("### 🏷️ Recommended Hashtags")
            st.markdown("*Personalized hashtag suggestions based on your personality:*")
            
            if recommendations:
                # Display as pills
                hashtag_html = ""
                for hashtag, score in recommendations[:top_k_recs]:
                    hashtag_html += f'<span class="hashtag-pill">#{hashtag}</span> '
                
                st.markdown(f"""
                <div style="margin: 1rem 0; line-height: 2.5;">
                    {hashtag_html}
                </div>
                """, unsafe_allow_html=True)
                
                # Details expander
                with st.expander("📊 Recommendation Details"):
                    rec_df = pd.DataFrame(
                        [(f"#{h}", f"{s:.3f}") for h, s in recommendations[:top_k_recs]],
                        columns=["Hashtag", "Score"]
                    )
                    st.dataframe(rec_df, hide_index=True, use_container_width=True)
            else:
                st.info("🔧 Recommender not available. Train with: `python scripts/recsys_eval.py`")
        
        # Raw output expander
        with st.expander("🔧 Raw Data (JSON)"):
            output_data = {
                "predicted_traits": predicted_traits,
                "dominant_trait": dominant_trait,
                "posts_analyzed": len(tweets),
                "recommendations": [(h, float(s)) for h, s in recommendations[:10]] if recommendations else [],
            }
            st.json(output_data)


if __name__ == "__main__":
    main()
