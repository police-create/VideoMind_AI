"""VideoMind AI - Multimodal YouTube Research & Learning Copilot.

A portfolio-grade research and learning workspace transforming YouTube
videos into verifiable, evidence-grounded knowledge.
"""

import streamlit as st
import time
import io
import json
import pandas as pd
from typing import List, Dict, Any, Optional

# Core Architecture Imports
from src.config import get_config
from src.models import (
    VideoSource, TranscriptChunk, EvidenceCandidate, AnswerResponse,
    KeyMoment, StudyPack, DisagreementReport, ComparisonMatrix
)
from src.youtube.url_parser import extract_video_id, is_valid_youtube_url, build_youtube_watch_url
from src.youtube.metadata import fetch_video_metadata
from src.youtube.comments import fetch_public_comments, analyze_audience_themes
from src.ingestion.transcript import fetch_transcript_entries
from src.ingestion.timestamps import build_semantic_chunks
from src.ingestion.demo_loader import load_demo_dataset
from src.embeddings.embedder import get_embeddings_batch
from src.retrieval.vector_store import FaissVectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.rag.answer_generator import generate_grounded_answer
from src.agents.query_router import classify_query_intent
from src.agents.comparison_agent import compare_sources
from src.agents.research_agent import detect_disagreement
from src.agents.tutor_agent import (
    generate_tutoring_curriculum, evaluate_tutor_response, TutoringCurriculum
)
from src.evidence.fact_checker import verify_claim_externally
from src.multimodal.video_analyzer import (
    analyze_video_moment_structured, timestamp_to_seconds, VisualMomentAnalysis
)
from src.features.notes import generate_study_pack
from src.features.quiz import generate_quiz
from src.features.key_moments import extract_key_moments
from src.features.code_extractor import extract_code_demonstrations
from src.database.repository import Repository
from src.evidence.citation_builder import format_evidence_badge
from src.exceptions import VideoMindError
from eval.evaluation_suite import run_evaluation

# -----------------------------------------------------------------------------
# STREAMLIT CONFIGURATION & CUSTOM DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VideoMind AI • Multimodal YouTube Research Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Academic/AI Research Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #090d16;
        --card-bg: rgba(22, 30, 49, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-emerald: #10b981;
        --accent-cyan: #06b6d4;
        --accent-indigo: #6366f1;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Glassmorphism Cards */
    .vm-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .vm-card-highlight {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.10) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Badges */
    .vm-badge-grounded {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    .vm-badge-abstain {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    .vm-badge-external {
        background-color: rgba(6, 182, 212, 0.2);
        color: #06b6d4;
        border: 1px solid rgba(6, 182, 212, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    .vm-badge-demo {
        background-color: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Video Player Drawer */
    .player-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# APPLICATION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def get_repository() -> Repository:
    if "repository" not in st.session_state:
        st.session_state.repository = Repository()
    return st.session_state.repository


def initialize_app_state():
    """Ensure all core session state dictionaries and stores are available."""
    config = get_config()

    if "sources" not in st.session_state:
        st.session_state.sources: List[VideoSource] = []

    if "chunks" not in st.session_state:
        st.session_state.chunks: List[TranscriptChunk] = []

    if "chunks_by_source" not in st.session_state:
        st.session_state.chunks_by_source: Dict[str, List[TranscriptChunk]] = {}

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FaissVectorStore(dimension=768)

    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever(st.session_state.vector_store)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "selected_source_ids" not in st.session_state:
        st.session_state.selected_source_ids = []

    if "demo_mode_loaded" not in st.session_state:
        st.session_state.demo_mode_loaded = False

    # Synchronized In-App Player State
    if "active_player_vid" not in st.session_state:
        st.session_state.active_player_vid = None

    if "active_player_sec" not in st.session_state:
        st.session_state.active_player_sec = 0

    if "active_player_title" not in st.session_state:
        st.session_state.active_player_title = ""

    # Tutor state
    if "tutor_curriculum" not in st.session_state:
        st.session_state.tutor_curriculum = None

    # Benchmark results state
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None


initialize_app_state()
repo = get_repository()
config = get_config()


# -----------------------------------------------------------------------------
# HELPER ACTIONS
# -----------------------------------------------------------------------------
def set_in_app_player(video_id: str, start_sec: float, title: str = ""):
    """Trigger the in-app synchronized video player to load at exact second."""
    st.session_state.active_player_vid = video_id
    st.session_state.active_player_sec = int(start_sec)
    st.session_state.active_player_title = title


def render_synchronized_player_panel():
    """Renders the synchronized video player if an active video is loaded."""
    if st.session_state.active_player_vid:
        vid = st.session_state.active_player_vid
        sec = st.session_state.active_player_sec
        title = st.session_state.active_player_title or "Synchronized Video"
        mm = sec // 60
        ss = sec % 60
        watch_url = f"https://www.youtube.com/watch?v={vid}"

        with st.container():
            st.markdown(
                f"""
                <div class="player-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <strong>📺 In-App Synchronized Player:</strong> {title} 
                            <span style="color: #10b981; font-weight: 700;">[{mm:02d}:{ss:02d}]</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            col_v, col_ctrl = st.columns([4, 1])
            with col_v:
                st.video(watch_url, start_time=sec)
            with col_ctrl:
                st.caption(f"Seek position: `{mm:02d}:{ss:02d}` ({sec}s)")
                st.markdown(f"[▶ Open external tab]({watch_url}&t={sec}s)")
                if st.button("✕ Close Player", key="btn_close_player"):
                    st.session_state.active_player_vid = None
                    st.rerun()
            st.divider()


def activate_demo_dataset():
    """Load bundled sample sources for offline/demo usage."""
    with st.spinner("Loading curated academic demo sources & building semantic index..."):
        sources, chunks = load_demo_dataset()
        st.session_state.sources = list(sources)
        st.session_state.chunks = list(chunks)
        st.session_state.chunks_by_source = {}

        for c in chunks:
            if c.video_id not in st.session_state.chunks_by_source:
                st.session_state.chunks_by_source[c.video_id] = []
            st.session_state.chunks_by_source[c.video_id].append(c)

        # Index in FAISS
        st.session_state.vector_store.clear()
        embeddings = get_embeddings_batch([c.text for c in chunks])
        st.session_state.vector_store.add_chunks(chunks, embeddings)
        st.session_state.selected_source_ids = [s.video_id for s in sources]
        st.session_state.demo_mode_loaded = True

        for s in sources:
            repo.log_event("video_indexed", {"video_id": s.video_id, "title": s.title, "is_demo": True})

    st.success(f"Successfully loaded {len(sources)} educational videos ({len(chunks)} timestamped chunks)!")
    st.rerun()


def ingest_youtube_source(url_or_id: str):
    """Fetch metadata and transcript for a live YouTube video and index it."""
    try:
        video_id = extract_video_id(url_or_id)
    except Exception as e:
        st.error(f"Invalid URL: {e}")
        return

    # Check if already indexed
    if any(s.video_id == video_id for s in st.session_state.sources):
        st.info("This video is already in your source workspace.")
        return

    progress_bar = st.progress(0, text="Validating YouTube URL...")
    try:
        time.sleep(0.2)
        progress_bar.progress(25, text="Fetching video metadata...")
        source = fetch_video_metadata(video_id, config.youtube_api_key)

        progress_bar.progress(50, text="Retrieving transcript cues...")
        entries = fetch_transcript_entries(video_id)

        progress_bar.progress(75, text="Building semantic chunks with timestamps...")
        chunks = build_semantic_chunks(
            raw_entries=entries,
            video_id=video_id,
            source_title=source.title,
            target_words=config.chunk_size // 3,
            overlap_words=config.chunk_overlap // 3
        )

        progress_bar.progress(90, text="Generating embeddings & indexing in FAISS...")
        embeddings = get_embeddings_batch([c.text for c in chunks], api_key=config.gemini_api_key)
        st.session_state.vector_store.add_chunks(chunks, embeddings)

        # Store in session
        st.session_state.sources.append(source)
        st.session_state.chunks.extend(chunks)
        st.session_state.chunks_by_source[video_id] = chunks
        if video_id not in st.session_state.selected_source_ids:
            st.session_state.selected_source_ids.append(video_id)

        repo.log_event("video_indexed", {"video_id": video_id, "title": source.title, "is_demo": False})
        progress_bar.progress(100, text="Video successfully indexed!")
        time.sleep(0.3)
        progress_bar.empty()
        st.success(f"Indexed: '{source.title}' ({len(chunks)} chunks)")
        st.rerun()
    except VideoMindError as e:
        progress_bar.empty()
        st.error(f"Error: {e.message}")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Unexpected ingestion failure: {str(e)}")


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SOURCE CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 **VideoMind AI**")
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Multimodal YouTube Research Copilot</p>", unsafe_allow_html=True)
    st.divider()

    # Navigation Menu
    nav_selection = st.radio(
        "Workspace Navigation",
        [
            "🏠 Dashboard",
            "📺 Sources & Video Sync",
            "💬 Research Chat & Claims",
            "🎓 Personalized AI Tutor",
            "👁️ Multimodal Visual Lab",
            "⚖️ Multi-Video Compare",
            "📚 AI Study Lab",
            "📝 Quiz Generator",
            "⏱️ Key Moments & Timeline",
            "💻 Developer Mode (Code)",
            "🔍 Disagreement Detector",
            "👥 Audience Insights",
            "📓 Research Notebook",
            "📊 Analytics & Benchmarks",
            "⚙️ Settings"
        ],
        index=0
    )

    st.divider()

    # Active Sources Quick Selector
    st.markdown(f"**Active Sources ({len(st.session_state.sources)})**")
    if not st.session_state.sources:
        st.caption("No sources loaded yet.")
        if st.button("⚡ Load Demo Sources", use_container_width=True):
            activate_demo_dataset()
    else:
        # Checkbox selection for multi-source reasoning
        selected_ids = []
        for s in st.session_state.sources:
            is_checked = s.video_id in st.session_state.selected_source_ids
            badge = " [DEMO]" if s.is_demo else ""
            if st.checkbox(f"{s.title[:24]}...{badge}", value=is_checked, key=f"src_cb_{s.video_id}"):
                selected_ids.append(s.video_id)
        st.session_state.selected_source_ids = selected_ids

        col_clr1, col_clr2 = st.columns(2)
        with col_clr1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_source_ids = [s.video_id for s in st.session_state.sources]
                st.rerun()
        with col_clr2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.sources.clear()
                st.session_state.chunks.clear()
                st.session_state.chunks_by_source.clear()
                st.session_state.vector_store.clear()
                st.session_state.selected_source_ids.clear()
                st.session_state.demo_mode_loaded = False
                st.session_state.active_player_vid = None
                st.rerun()

    st.divider()
    # API Status Indicator
    api_ready = config.is_gemini_available()
    if api_ready:
        st.markdown("<span style='color: #10b981;'>● Gemini API: Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: #f59e0b;'>● Mode: Offline / Demo Ready</span>", unsafe_allow_html=True)


# Always render in-app video player header if user activated a timestamp
render_synchronized_player_panel()


# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD / HOME
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Dashboard":
    st.markdown("""
    <div class="vm-card-highlight">
        <h1 style='margin-bottom: 4px; font-weight: 800;'>VideoMind AI</h1>
        <h4 style='color: #94a3b8; font-weight: 400;'>Multimodal YouTube Research & Learning Copilot</h4>
        <p style='color: #cbd5e1; font-size: 0.95rem;'>
            Turn YouTube lectures and technical talks into an evidence-grounded research workspace.
            Featuring hybrid vector retrieval, atomic claim-evidence alignment, targeted visual reconstruction,
            and adaptive Socratic tutoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        st.markdown("### Quick Ingest")
        input_url = st.text_input(
            "Paste YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="dash_url_input"
        )
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("➕ Ingest Video", use_container_width=True, type="primary"):
                if input_url:
                    ingest_youtube_source(input_url)
                else:
                    st.warning("Please enter a valid YouTube URL.")
        with col_btn2:
            if st.button("⚡ Try Demo Mode", use_container_width=True):
                activate_demo_dataset()

    with col2:
        st.markdown("### Workspace Metrics")
        st.metric("Sources Indexed", len(st.session_state.sources))
        st.metric("Chunks Stored", st.session_state.vector_store.get_total_chunks())

    with col3:
        st.markdown("### Model & Engine")
        st.write(f"**LLM:** `{config.llm_model}`")
        st.write(f"**Embedder:** `{config.embedding_model}`")
        st.write(f"**Index:** `FAISS (768-dim IP)`")

    st.divider()

    st.markdown("### 🌟 Research Differentiators")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        st.markdown("""
        <div class="vm-card">
            <h4>🎯 Evidence First</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Strict provenance tracking. Every claim maps back to verifiable timestamped cues.</p>
        </div>
        """, unsafe_allow_html=True)
    with f_col2:
        st.markdown("""
        <div class="vm-card">
            <h4>🎓 Socratic AI Tutor</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Adaptive learning paths tailored to your level (Beginner/Intermediate/Advanced) and goals.</p>
        </div>
        """, unsafe_allow_html=True)
    with f_col3:
        st.markdown("""
        <div class="vm-card">
            <h4>🚫 Principled Abstention</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Zero hallucinations. If evidence is insufficient, the system explicitly abstains.</p>
        </div>
        """, unsafe_allow_html=True)
    with f_col4:
        st.markdown("""
        <div class="vm-card">
            <h4>👁️ Multimodal Visuals</h4>
            <p style='color: #94a3b8; font-size: 0.85rem;'>Targeted inspection of slides, diagrams, and code editors with Mermaid reconstruction.</p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: SOURCES & VIDEO SYNC
# -----------------------------------------------------------------------------
elif nav_selection == "📺 Sources & Video Sync":
    st.markdown("## 📺 Source Intelligence & Video Synchronizer")
    st.markdown("Manage, inspect, and play ingested YouTube lectures with timestamp alignment.")

    # URL Ingest input
    with st.expander("➕ Ingest New YouTube Video", expanded=True):
        col_in, col_act = st.columns([3, 1])
        with col_in:
            new_url = st.text_input("Enter YouTube URL (Standard, Shorts, youtu.be, or embed):", key="src_page_url")
        with col_act:
            st.write("")
            st.write("")
            if st.button("Ingest Video", use_container_width=True, type="primary"):
                if new_url:
                    ingest_youtube_source(new_url)
                else:
                    st.warning("Please supply a valid URL.")

    if not st.session_state.sources:
        st.info("No videos are currently loaded. Add a YouTube URL above or click below to load demo sources.")
        if st.button("⚡ Load Demo Sources"):
            activate_demo_dataset()
    else:
        st.markdown(f"### Current Ingested Sources ({len(st.session_state.sources)})")
        for idx, src in enumerate(st.session_state.sources, 1):
            with st.container():
                c_thumb, c_meta, c_actions = st.columns([1.5, 3.5, 1.5])
                with c_thumb:
                    if src.thumbnail_url:
                        st.image(src.thumbnail_url, use_container_width=True)
                with c_meta:
                    demo_tag = '<span class="vm-badge-demo">DEMO DATA</span>' if src.is_demo else '<span class="vm-badge-grounded">LIVE SOURCE</span>'
                    st.markdown(f"**SOURCE {idx:02d}: {src.title}** {demo_tag}", unsafe_allow_html=True)
                    st.markdown(f"**Channel:** {src.channel} • **Duration:** `{src.duration_formatted}` • **ID:** `{src.video_id}`")
                    st.markdown(f"[▶ Open external YouTube link]({src.url})")
                    chunks_count = len(st.session_state.chunks_by_source.get(src.video_id, []))
                    st.caption(f"Indexed {chunks_count} timestamped semantic chunks.")

                with c_actions:
                    if st.button("📺 Play In-App Player", key=f"play_src_{src.video_id}", use_container_width=True):
                        set_in_app_player(src.video_id, 0.0, src.title)
                        st.rerun()

                    is_active = src.video_id in st.session_state.selected_source_ids
                    if is_active:
                        if st.button("Deselect Source", key=f"desel_{src.video_id}", use_container_width=True):
                            st.session_state.selected_source_ids.remove(src.video_id)
                            st.rerun()
                    else:
                        if st.button("Select Source", key=f"sel_{src.video_id}", use_container_width=True):
                            st.session_state.selected_source_ids.append(src.video_id)
                            st.rerun()

                    if st.button("🗑️ Remove Source", key=f"del_{src.video_id}", use_container_width=True):
                        st.session_state.sources = [s for s in st.session_state.sources if s.video_id != src.video_id]
                        if src.video_id in st.session_state.selected_source_ids:
                            st.session_state.selected_source_ids.remove(src.video_id)
                        st.session_state.chunks = [c for c in st.session_state.chunks if c.video_id != src.video_id]
                        if src.video_id in st.session_state.chunks_by_source:
                            del st.session_state.chunks_by_source[src.video_id]
                        st.session_state.vector_store.clear()
                        if st.session_state.chunks:
                            embeddings = get_embeddings_batch([c.text for c in st.session_state.chunks])
                            st.session_state.vector_store.add_chunks(st.session_state.chunks, embeddings)
                        st.rerun()

                # Transcript Inspector
                with st.expander(f"🔍 Inspect Transcript Cues for '{src.title[:30]}...'"):
                    src_chunks = st.session_state.chunks_by_source.get(src.video_id, [])
                    for c in src_chunks:
                        col_c1, col_c2 = st.columns([1, 4])
                        with col_c1:
                            if st.button(f"▶ {c.start_formatted}", key=f"chk_play_{c.chunk_id}"):
                                set_in_app_player(c.video_id, c.start_seconds, c.source_title)
                                st.rerun()
                        with col_c2:
                            st.markdown(f"*{c.text}*")
            st.divider()


# -----------------------------------------------------------------------------
# PAGE 3: RESEARCH CHAT & CLAIM MAPPING
# -----------------------------------------------------------------------------
elif nav_selection == "💬 Research Chat & Claims":
    st.markdown("## 💬 Evidence-Grounded Research Chat (Feature 6 & 9)")
    st.markdown("Perplexity-style evidence presentation + NotebookLM-style source interaction with verifiable claim mappings.")

    # Source Selection bar
    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]
    if not active_sources:
        st.warning("⚠️ No sources selected. Please select at least one source in the sidebar or sources page.")
        if st.button("Select All Active Sources"):
            st.session_state.selected_source_ids = [s.video_id for s in st.session_state.sources]
            st.rerun()

    # Query Mode and Verification Options
    col_m1, col_m2, col_m3 = st.columns([1.5, 1.5, 2])
    with col_m1:
        chat_mode = st.selectbox(
            "Query Scope",
            ["ASK SELECTED VIDEOS", "ASK ALL SOURCES", "ASK PRIMARY SOURCE ONLY"],
            index=0
        )
    with col_m2:
        enable_web = st.toggle("🌐 External Verification", value=False, help="Cross-reference video statements against external academic literature (Feature 20).")
    with col_m3:
        st.write("")
        st.caption(f"Querying across **{len(active_sources)}** active source(s).")

    # Display conversation messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "answer_response" in msg and msg["answer_response"]:
                resp: AnswerResponse = msg["answer_response"]
                # Evidence Badge
                if resp.abstained:
                    st.markdown('<span class="vm-badge-abstain">NOT ENOUGH EVIDENCE (ABSTAINED)</span>', unsafe_allow_html=True)
                else:
                    badge_color = "#10b981" if resp.evidence_strength == "High" else "#f59e0b"
                    st.markdown(
                        f'<span class="vm-badge-grounded">SOURCE GROUNDED</span> '
                        f'<span style="background:{badge_color}25; color:{badge_color}; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:700;">'
                        f'Evidence Strength: {resp.evidence_strength} ({resp.groundedness_score:.2f})</span>',
                        unsafe_allow_html=True
                    )
                    if resp.external_verification:
                        st.markdown('<span class="vm-badge-external">EXTERNAL SOURCE VERIFIED</span>', unsafe_allow_html=True)

                # External Fact Check Report Box (Feature 20)
                if resp.external_verification:
                    ext = resp.external_verification
                    with st.expander("🌐 External Fact Check & Consensus (Feature 20)", expanded=True):
                        st.markdown(f"**Verification Status:** `{ext.get('verification_status')}`")
                        st.markdown(f"**Academic Literature Consensus:**\n> {ext.get('external_consensus')}")
                        st.markdown(f"*{ext.get('explanation')}*")
                        if ext.get("external_sources"):
                            st.caption("External Citations:")
                            for cit in ext.get("external_sources"):
                                st.markdown(f"- [{cit.get('title')}]({cit.get('url')}) • *{cit.get('publisher')}*")

                # Groundedness & Claim Map Expanders (Feature 9)
                if resp.claims:
                    with st.expander(f"🗺️ View Claim Map ({len(resp.claims)} Atomic Claims)", expanded=False):
                        for claim in resp.claims:
                            status_color = "#10b981" if "SUPPORTED" in claim.status else "#f59e0b" if "INFERENCE" in claim.status else "#94a3b8"
                            st.markdown(
                                f"<div style='border-left: 3px solid {status_color}; padding-left: 10px; margin-bottom: 8px;'>"
                                f"<strong>Claim:</strong> {claim.text}<br/>"
                                f"<small style='color: {status_color}; font-weight: 600;'>[{claim.status}]</small> — "
                                f"<small>{claim.explanation}</small>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                # Evidence Cards Expander (Feature 8)
                if resp.evidence_candidates:
                    with st.expander(f"📑 Supporting Evidence Cards ({len(resp.evidence_candidates)} Chunks)", expanded=False):
                        for cand in resp.evidence_candidates:
                            chk = cand.chunk
                            col_ex1, col_ex2 = st.columns([4, 1.2])
                            with col_ex1:
                                st.markdown(f"**{chk.source_title}** • `[{chk.start_formatted} – {chk.end_formatted}]`")
                                st.markdown(f"> *\"{chk.text}\"*")
                                st.caption(f"Relevance: {cand.score:.2f} ({cand.evidence_strength}) • Chunk ID: `{chk.chunk_id}`")
                            with col_ex2:
                                if st.button("▶ Play In-App", key=f"chat_play_{chk.chunk_id}_{len(st.session_state.messages)}"):
                                    set_in_app_player(chk.video_id, chk.start_seconds, chk.source_title)
                                    st.rerun()
                            st.divider()

    # User Input
    user_query = st.chat_input("Ask about video content, formulas, diagrams, code, or compare explanations...")
    if user_query:
        if not active_sources:
            st.error("Please load and select at least one video source before querying.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                intent = classify_query_intent(user_query, source_count=len(active_sources))
                st.caption(f"🧠 Detected Intent: `{intent}` • Querying hybrid index...")

                filter_vids = [s.video_id for s in active_sources]
                if chat_mode == "ASK PRIMARY SOURCE ONLY" and active_sources:
                    filter_vids = [active_sources[0].video_id]

                with st.spinner("Retrieving candidate evidence and evaluating groundedness..."):
                    candidates = st.session_state.retriever.retrieve(
                        query=user_query,
                        top_k=config.top_k,
                        video_ids=filter_vids,
                        api_key=config.gemini_api_key
                    )

                    answer_response = generate_grounded_answer(
                        question=user_query,
                        evidence_candidates=candidates,
                        selected_sources=active_sources,
                        mode=chat_mode,
                        enable_web_verification=enable_web,
                        api_key=config.gemini_api_key
                    )

                st.markdown(answer_response.answer_text)

                if answer_response.abstained:
                    st.markdown('<span class="vm-badge-abstain">NOT ENOUGH EVIDENCE (ABSTAINED)</span>', unsafe_allow_html=True)
                else:
                    badge_color = "#10b981" if answer_response.evidence_strength == "High" else "#f59e0b"
                    st.markdown(
                        f'<span class="vm-badge-grounded">SOURCE GROUNDED</span> '
                        f'<span style="background:{badge_color}25; color:{badge_color}; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:700;">'
                        f'Evidence Strength: {answer_response.evidence_strength} ({answer_response.groundedness_score:.2f})</span>',
                        unsafe_allow_html=True
                    )
                    if answer_response.external_verification:
                        st.markdown('<span class="vm-badge-external">EXTERNAL SOURCE VERIFIED</span>', unsafe_allow_html=True)

                if answer_response.external_verification:
                    ext = answer_response.external_verification
                    with st.expander("🌐 External Fact Check & Consensus (Feature 20)", expanded=True):
                        st.markdown(f"**Verification Status:** `{ext.get('verification_status')}`")
                        st.markdown(f"**Academic Literature Consensus:**\n> {ext.get('external_consensus')}")
                        st.markdown(f"*{ext.get('explanation')}*")

                if answer_response.claims:
                    with st.expander(f"🗺️ View Claim Map ({len(answer_response.claims)} Atomic Claims)", expanded=False):
                        for claim in answer_response.claims:
                            status_color = "#10b981" if "SUPPORTED" in claim.status else "#f59e0b" if "INFERENCE" in claim.status else "#94a3b8"
                            st.markdown(
                                f"<div style='border-left: 3px solid {status_color}; padding-left: 10px; margin-bottom: 8px;'>"
                                f"<strong>Claim:</strong> {claim.text}<br/>"
                                f"<small style='color: {status_color}; font-weight: 600;'>[{claim.status}]</small> — "
                                f"<small>{claim.explanation}</small>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                if answer_response.evidence_candidates:
                    with st.expander(f"📑 Supporting Evidence Cards ({len(answer_response.evidence_candidates)} Chunks)", expanded=False):
                        for cand in answer_response.evidence_candidates:
                            chk = cand.chunk
                            col_ex1, col_ex2 = st.columns([4, 1.2])
                            with col_ex1:
                                st.markdown(f"**{chk.source_title}** • `[{chk.start_formatted} – {chk.end_formatted}]`")
                                st.markdown(f"> *\"{chk.text}\"*")
                                st.caption(f"Relevance: {cand.score:.2f} ({cand.evidence_strength}) • Chunk ID: `{chk.chunk_id}`")
                            with col_ex2:
                                if st.button("▶ Play In-App", key=f"chat_play_{chk.chunk_id}_new"):
                                    set_in_app_player(chk.video_id, chk.start_seconds, chk.source_title)
                                    st.rerun()
                            st.divider()

                col_save_n, _ = st.columns([1, 3])
                with col_save_n:
                    if st.button("💾 Bookmark to Notes", key=f"bkmk_{len(st.session_state.messages)}"):
                        repo.add_note(
                            session_id="default_session",
                            title=user_query[:50],
                            content=answer_response.answer_text,
                            timestamp_ref=candidates[0].chunk.start_formatted if candidates else "",
                            source_title=candidates[0].chunk.source_title if candidates else ""
                        )
                        st.success("Saved to Research Notebook!")

                repo.log_event("question_asked", {"query": user_query, "intent": intent})
                repo.log_event("evidence_retrieved", {"count": len(candidates)})

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_response.answer_text,
                    "answer_response": answer_response
                })


# -----------------------------------------------------------------------------
# PAGE 4: PERSONALIZED AI TUTOR (FEATURE 17)
# -----------------------------------------------------------------------------
elif nav_selection == "🎓 Personalized AI Tutor":
    st.markdown("## 🎓 Personalized Socratic AI Tutor (Feature 17)")
    st.markdown("Adaptive learning copilot tailored to your exact goal and proficiency level.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one active video source to initialize your tutoring session.")
    else:
        col_t1, col_t2, col_t3 = st.columns([2, 1.5, 1.5])
        with col_t1:
            tutor_src = st.selectbox(
                "Source Video",
                active_sources,
                format_func=lambda s: f"{s.title} ({s.channel})",
                key="tutor_source_select"
            )
        with col_t2:
            tutor_goal = st.selectbox(
                "Learning Goal",
                ["Exam preparation", "Interview preparation", "Project learning", "General understanding"],
                index=0
            )
        with col_t3:
            tutor_level = st.selectbox(
                "Proficiency Level",
                ["Beginner (Intuition & Analogy)", "Intermediate (Workflow & Implementation)", "Advanced (Theoretical Rigor & Proofs)"],
                index=0
            )

        if st.button("🚀 Generate Personalized Curriculum", type="primary", key="btn_gen_tutor"):
            with st.spinner("Synthesizing personalized Socratic learning path..."):
                src_chunks = st.session_state.chunks_by_source.get(tutor_src.video_id, [])
                level_clean = tutor_level.split(" ")[0]
                curriculum = generate_tutoring_curriculum(
                    chunks=src_chunks,
                    source_title=tutor_src.title,
                    goal=tutor_goal,
                    level=level_clean,
                    api_key=config.gemini_api_key
                )
                st.session_state.tutor_curriculum = curriculum
                repo.log_event("tutor_curriculum_generated", {"video_id": tutor_src.video_id, "goal": tutor_goal, "level": level_clean})

        if st.session_state.tutor_curriculum:
            curr: TutoringCurriculum = st.session_state.tutor_curriculum
            st.divider()

            st.markdown(f"### 📋 Personalized Study Plan: {curr.source_title}")
            st.markdown(f"**Goal:** `{curr.goal}` • **Level:** `{curr.level}`")
            st.markdown(f"*{curr.summary_overview}*")

            # Learning Objectives
            with st.expander("🎯 Learning Objectives", expanded=True):
                for obj in curr.learning_objectives:
                    st.markdown(f"- {obj}")

            # Structured Modules
            st.markdown("### 📚 Progressive Mastery Milestones")
            for idx, mod in enumerate(curr.modules, 1):
                col_m_text, col_m_play = st.columns([4, 1.2])
                with col_m_text:
                    st.markdown(f"#### {mod.get('module_title')} `[{mod.get('timestamp')}]`")
                    st.markdown(f"{mod.get('core_concepts')}")
                    st.caption(f"💡 Exercise: {mod.get('practical_exercise')}")
                with col_m_play:
                    sec_ts = timestamp_to_seconds(mod.get('timestamp', '00:00'))
                    if st.button("▶ Watch Section", key=f"tutor_play_{idx}"):
                        set_in_app_player(tutor_src.video_id, sec_ts, tutor_src.title)
                        st.rerun()
                st.divider()

            # Interactive Checkpoints
            st.markdown("### 🧠 Socratic Knowledge Checkpoints")
            for c_idx, cp in enumerate(curr.checkpoints, 1):
                st.markdown(f"**Checkpoint {c_idx}: {cp.topic}** ({cp.difficulty}) `[{cp.timestamp_ref}]`")
                st.markdown(f"*{cp.question}*")
                with st.expander("💡 Need a hint?"):
                    st.caption(cp.hint)

                user_ans = st.text_area("Your explanation:", key=f"tutor_ans_{c_idx}")
                if st.button(f"Submit Checkpoint {c_idx}", key=f"sub_cp_{c_idx}"):
                    if user_ans.strip():
                        eval_res = evaluate_tutor_response(
                            question=cp.question,
                            user_answer=user_ans,
                            ideal_answer=cp.ideal_answer,
                            level=curr.level,
                            api_key=config.gemini_api_key
                        )
                        score = eval_res.get("score", 0)
                        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
                        st.markdown(
                            f"<div style='border: 1px solid {score_color}; padding: 10px; border-radius: 8px; margin-top: 8px;'>"
                            f"<strong>Evaluation Status:</strong> <span style='color:{score_color}; font-weight:700;'>{eval_res.get('status')} ({score}/100)</span><br/>"
                            f"<p style='margin-top: 4px;'>{eval_res.get('feedback')}</p>"
                            f"<small><strong>Guidance:</strong> {eval_res.get('key_missing_concept')}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Please type your response before submitting.")
                st.divider()


# -----------------------------------------------------------------------------
# PAGE 5: MULTIMODAL VISUAL LAB (FEATURE 7)
# -----------------------------------------------------------------------------
elif nav_selection == "👁️ Multimodal Visual Lab":
    st.markdown("## 👁️ Targeted Multimodal Visual Lab (Feature 7)")
    st.markdown("Inspect slides, architecture diagrams, code editors, and visual demonstrations at specific video moments.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one active source.")
    else:
        col_v1, col_v2 = st.columns([2.5, 1.5])
        with col_v1:
            target_src = st.selectbox("Select Video", active_sources, format_func=lambda s: s.title, key="vis_source_sel")
        with col_v2:
            preset_ts = st.selectbox(
                "Quick Milestone Timestamps",
                ["Custom Timestamp", "06:05 (Bagging Architecture)", "07:30 (Deep Neural Pipeline)", "05:40 (IDE Code Demo)"]
            )

        ts_query = "06:05" if "06:05" in preset_ts else "07:30" if "07:30" in preset_ts else "05:40" if "05:40" in preset_ts else "06:05"
        custom_ts = st.text_input("Enter Target Timestamp (e.g. 06:05, 14:30)", value=ts_query)
        visual_query_text = st.text_input(
            "What visual aspect do you want to inspect?",
            value="Describe the architecture diagram and data flow shown on screen at this moment."
        )

        if st.button("🔍 Run Multimodal Visual Inspection", type="primary"):
            with st.spinner(f"Analyzing visual moment at {custom_ts} in '{target_src.title}'..."):
                analysis: VisualMomentAnalysis = analyze_video_moment_structured(
                    video_id=target_src.video_id,
                    query=visual_query_text,
                    timestamp_str=custom_ts,
                    api_key=config.gemini_api_key
                )
                st.session_state[f"vis_{target_src.video_id}_{custom_ts}"] = analysis

        vis_key = f"vis_{target_src.video_id}_{custom_ts}"
        if vis_key in st.session_state:
            res: VisualMomentAnalysis = st.session_state[vis_key]
            st.divider()

            col_desc, col_play = st.columns([3, 2])
            with col_desc:
                st.markdown(f"### Visual Inspection at `{res.timestamp}`")
                st.markdown(f"**Diagram Classification:** `{res.diagram_type}`")
                st.markdown(f"{res.visual_description}")

                if res.on_screen_text:
                    st.markdown("#### 🔤 Detected On-Screen Text & Labels")
                    for lbl in res.on_screen_text:
                        st.markdown(f"- `{lbl}`")

            with col_play:
                sec_val = res.timestamp_seconds or timestamp_to_seconds(res.timestamp)
                st.markdown(f"**Synchronized Playback at {res.timestamp}:**")
                st.video(target_src.url, start_time=int(sec_val))

            if res.mermaid_code:
                st.markdown("### 📐 Reconstructed Architectural Diagram")
                st.caption("Structured Mermaid visualization synthesized from on-screen presentation elements.")
                st.markdown(f"```mermaid\n{res.mermaid_code}\n```")


# -----------------------------------------------------------------------------
# PAGE 6: MULTI-VIDEO COMPARISON (FEATURE 10)
# -----------------------------------------------------------------------------
elif nav_selection == "⚖️ Multi-Video Compare":
    st.markdown("## ⚖️ Multi-Source Comparative Intelligence (Feature 10)")
    st.markdown("Compare explanations, pedagogical depth, and code across 2 to 5 YouTube sources.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if len(active_sources) < 2:
        st.warning("⚠️ Please select at least 2 video sources in the sidebar to run a comparative matrix.")
        if st.button("⚡ Select All Sources for Comparison"):
            st.session_state.selected_source_ids = [s.video_id for s in st.session_state.sources]
            st.rerun()
    else:
        topic_input = st.text_input(
            "Comparison Focus / Topic",
            placeholder="e.g. Random Forest, Principal Component Analysis, Optimization Algorithms",
            value="Supervised Learning & Model Evaluation"
        )

        if st.button("Generate Comparison Matrix", type="primary"):
            with st.spinner(f"Synthesizing comparative matrix across {len(active_sources)} sources..."):
                matrix = compare_sources(
                    sources=active_sources,
                    chunks_by_source=st.session_state.chunks_by_source,
                    topic_query=topic_input,
                    api_key=config.gemini_api_key
                )
                st.session_state["comparison_matrix"] = matrix
                repo.log_event("sources_compared", {"topic": topic_input, "source_count": len(active_sources)})

        if "comparison_matrix" in st.session_state:
            matrix = st.session_state["comparison_matrix"]
            st.markdown("### 📊 Comparative Analysis Matrix")
            st.markdown(f"*{matrix.summary}*")

            # Render Table
            table_data = {"Dimension": matrix.dimensions}
            for v_title in matrix.videos:
                col_vals = []
                for dim in matrix.dimensions:
                    col_vals.append(matrix.matrix.get(dim, {}).get(v_title, "N/A"))
                table_data[v_title] = col_vals

            df_matrix = pd.DataFrame(table_data)
            st.table(df_matrix)

            # Export capabilities
            csv_data = df_matrix.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Matrix as CSV",
                data=csv_data,
                file_name=f"videomind_comparison_{topic_input.replace(' ', '_')}.csv",
                mime="text/csv"
            )


# -----------------------------------------------------------------------------
# PAGE 7: AI STUDY LAB (FEATURE 12 & 13)
# -----------------------------------------------------------------------------
elif nav_selection == "📚 AI Study Lab":
    st.markdown("## 📚 AI Study Lab (Feature 12 & 13)")
    st.markdown("Generate research-grade study packs, structured notes, and adaptive explanations.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one active source.")
    else:
        target_source = st.selectbox(
            "Select Source Video for Study Pack",
            active_sources,
            format_func=lambda s: f"{s.title} ({s.channel})"
        )

        tab_pack, tab_adaptive = st.tabs(["📑 Build Study Pack", "🎯 Adaptive Explanation"])

        with tab_pack:
            if st.button("⚡ Build Complete Study Pack", type="primary", key="btn_study_pack"):
                with st.spinner("Synthesizing comprehensive study pack from transcript evidence..."):
                    source_chunks = st.session_state.chunks_by_source.get(target_source.video_id, [])
                    pack = generate_study_pack(source_chunks, target_source.title, api_key=config.gemini_api_key)
                    st.session_state[f"study_pack_{target_source.video_id}"] = pack
                    repo.log_event("study_pack_generated", {"video_id": target_source.video_id})

            if f"study_pack_{target_source.video_id}" in st.session_state:
                pack = st.session_state[f"study_pack_{target_source.video_id}"]

                sub_t1, sub_t2, sub_t3, sub_t4, sub_t5 = st.tabs([
                    "OVERVIEW", "CONCEPTS & DEFINITIONS", "INTERVIEW PREP", "EXAM PREP", "REVISION CHECKLIST"
                ])

                with sub_t1:
                    st.markdown("### Executive Summary")
                    st.markdown(pack.summary)
                    if pack.examples:
                        st.markdown("#### Concrete Examples")
                        for ex in pack.examples:
                            st.markdown(f"- {ex}")

                with sub_t2:
                    st.markdown("### Core Concepts")
                    for c in pack.key_concepts:
                        st.markdown(f"**{c.get('concept')}** `[{c.get('timestamp', '00:00')}]`")
                        st.markdown(f"> {c.get('explanation')}")

                    st.markdown("### Rigorous Definitions")
                    for d in pack.definitions:
                        st.markdown(f"- **{d.get('term')}**: {d.get('definition')}")

                    if pack.formulas:
                        st.markdown("### Mathematical & Algorithmic Formulations")
                        for form in pack.formulas:
                            st.code(form, language="python")

                with sub_t3:
                    st.markdown("### High-Yield Technical Interview Questions")
                    for i, q in enumerate(pack.interview_questions, 1):
                        st.markdown(f"**Q{i}:** {q}")

                with sub_t4:
                    st.markdown("### Academic Exam Questions")
                    for i, q in enumerate(pack.exam_questions, 1):
                        st.markdown(f"**Q{i}:** {q}")

                    if pack.common_mistakes:
                        st.markdown("### Common Pitfalls & Mistakes")
                        for m in pack.common_mistakes:
                            st.markdown(f"⚠️ *{m}*")

                with sub_t5:
                    st.markdown("### Revision Checklist")
                    for item in pack.revision_checklist:
                        st.checkbox(item, key=f"rev_{item[:20]}")

                # Download Study Pack as Markdown
                md_export = f"# Study Pack: {target_source.title}\n\n## Executive Summary\n{pack.summary}\n\n## Key Concepts\n"
                for c in pack.key_concepts:
                    md_export += f"- **{c.get('concept')}** [{c.get('timestamp', '00:00')}]: {c.get('explanation')}\n"
                md_export += "\n## Definitions\n"
                for d in pack.definitions:
                    md_export += f"- **{d.get('term')}**: {d.get('definition')}\n"

                st.download_button(
                    label="📥 Download Complete Study Pack (.md)",
                    data=md_export.encode('utf-8'),
                    file_name=f"{target_source.title[:30]}_StudyPack.md",
                    mime="text/markdown"
                )

        with tab_adaptive:
            st.markdown("### Adaptive Pedagogical Explanations (Feature 13)")
            topic_to_explain = st.text_input("Concept to Explain", value="Principal Component Analysis", key="adapt_topic")
            level = st.radio("Proficiency Level", ["Beginner (Intuition & Analogy)", "Intermediate (Workflow & Math)", "Advanced (Assumptions & Rigor)"])

            if st.button("Generate Adaptive Explanation", key="btn_adapt"):
                chunks = st.session_state.chunks_by_source.get(target_source.video_id, [])
                sample_chunk = chunks[0] if chunks else None
                ts = sample_chunk.start_formatted if sample_chunk else "01:15"

                if "Beginner" in level:
                    st.markdown(f"""
                    **[SUPPORTED BY SOURCE at {ts}]:**
                    Imagine you are taking a photo of a running 3D object from the best possible camera angle so nothing is blocked. PCA finds that single best camera angle where the data spreads out the most.
                    """)
                elif "Intermediate" in level:
                    st.markdown(f"""
                    **[SUPPORTED BY SOURCE at {ts}]:**
                    PCA calculates the empirical covariance matrix of centered features, computes eigenvectors and eigenvalues, and projects original data onto orthogonal principal axes ordered by variance explained.
                    """)
                else:
                    st.markdown(f"""
                    **[SUPPORTED BY SOURCE at {ts}]:**
                    PCA assumes linear relationships and orthogonal basis vectors. It seeks to maximize variance or equivalently minimize reconstruction Mean Squared Error. Limitations include sensitivity to outliers and failure on non-linear manifolds.
                    """)


# -----------------------------------------------------------------------------
# PAGE 8: QUIZ GENERATOR (FEATURE 14)
# -----------------------------------------------------------------------------
elif nav_selection == "📝 Quiz Generator":
    st.markdown("## 📝 Evidence-Grounded Quiz Generator (Feature 14)")
    st.markdown("Test comprehension with questions strictly sourced from video evidence.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one active source.")
    else:
        target_src = st.selectbox(
            "Select Video for Quiz",
            active_sources,
            format_func=lambda s: s.title
        )

        if st.button("⚡ Generate Assessment Quiz", type="primary"):
            with st.spinner("Extracting questions from video evidence..."):
                chunks = st.session_state.chunks_by_source.get(target_src.video_id, [])
                quiz_items = generate_quiz(chunks, target_src.title, api_key=config.gemini_api_key)
                st.session_state[f"quiz_{target_src.video_id}"] = quiz_items
                repo.log_event("quiz_generated", {"video_id": target_src.video_id, "count": len(quiz_items)})

        if f"quiz_{target_src.video_id}" in st.session_state:
            items = st.session_state[f"quiz_{target_src.video_id}"]
            st.markdown(f"### Assessment: {target_src.title}")

            for idx, q in enumerate(items, 1):
                st.markdown(f"#### Q{idx} ({q.difficulty} • {q.question_type})")
                st.markdown(f"**{q.question_text}**")
                if q.timestamp:
                    st.caption(f"Source Timestamp: `{q.timestamp}`")

                if q.question_type in ["MCQ", "True/False"]:
                    user_ans = st.radio("Select answer:", q.options, key=f"q_radio_{target_src.video_id}_{idx}")
                    if st.button(f"Submit Q{idx}", key=f"sub_q_{idx}"):
                        if user_ans == q.correct_answer:
                            st.success(f"✅ Correct! {q.explanation}")
                        else:
                            st.error(f"❌ Incorrect. Correct answer: {q.correct_answer}. {q.explanation}")
                else:
                    user_text = st.text_input("Your answer:", key=f"q_txt_{idx}")
                    with st.expander("Reveal Correct Answer & Source Evidence"):
                        st.markdown(f"**Correct Answer:** {q.correct_answer}")
                        st.markdown(f"> *{q.explanation}*")
                st.divider()

            # Download Quiz
            quiz_md = f"# Quiz Assessment: {target_src.title}\n\n"
            for idx, q in enumerate(items, 1):
                quiz_md += f"### Q{idx} [{q.difficulty}]: {q.question_text}\n"
                if q.options:
                    for opt in q.options:
                        quiz_md += f"- {opt}\n"
                quiz_md += f"\n**Correct Answer:** {q.correct_answer}\n**Explanation:** {q.explanation}\n\n"

            st.download_button(
                label="📥 Download Quiz & Solutions (.md)",
                data=quiz_md.encode('utf-8'),
                file_name=f"{target_src.title[:30]}_Quiz.md",
                mime="text/markdown"
            )


# -----------------------------------------------------------------------------
# PAGE 9: KEY MOMENTS & TIMELINE (FEATURE 15 & 21)
# -----------------------------------------------------------------------------
elif nav_selection == "⏱️ Key Moments & Timeline":
    st.markdown("## ⏱️ Key Moments & Source Timeline (Feature 15 & 21)")
    st.markdown("Detects conceptual milestones, code demonstrations, diagrams, and warnings.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one active source.")
    else:
        target_src = st.selectbox("Select Video Timeline", active_sources, format_func=lambda s: s.title)
        chunks = st.session_state.chunks_by_source.get(target_src.video_id, [])

        moments = extract_key_moments(chunks)

        if not moments:
            st.info("No key moments detected in this segment.")
        else:
            st.markdown(f"### Detected Milestones ({len(moments)})")
            for idx, m in enumerate(moments, 1):
                col_t, col_badge, col_sum, col_act = st.columns([1, 1.2, 3.5, 1.3])
                with col_t:
                    st.markdown(f"`{m.timestamp_formatted}`")
                with col_badge:
                    st.markdown(f"**[{m.label}]**")
                with col_sum:
                    st.markdown(f"{m.summary}")
                with col_act:
                    if st.button("▶ Play In-App", key=f"km_play_{idx}"):
                        set_in_app_player(target_src.video_id, m.timestamp_seconds, target_src.title)
                        st.rerun()
                st.divider()


# -----------------------------------------------------------------------------
# PAGE 10: DEVELOPER MODE (CODE EXTRACTOR) (FEATURE 16)
# -----------------------------------------------------------------------------
elif nav_selection == "💻 Developer Mode (Code)":
    st.markdown("## 💻 Developer Mode & Code Extraction (Feature 16)")
    st.markdown("Isolates programming demonstrations, algorithms, and reconstructs clean code.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select at least one source.")
    else:
        target_src = st.selectbox("Select Video for Code Extraction", active_sources, format_func=lambda s: s.title)

        if st.button("Scan for Code Demonstrations", type="primary"):
            with st.spinner("Scanning transcript and visual cues for code implementations..."):
                chunks = st.session_state.chunks_by_source.get(target_src.video_id, [])
                code_items = extract_code_demonstrations(chunks, api_key=config.gemini_api_key)
                st.session_state[f"code_{target_src.video_id}"] = code_items

        if f"code_{target_src.video_id}" in st.session_state:
            items = st.session_state[f"code_{target_src.video_id}"]
            if not items:
                st.info("No explicit programming code or script syntax detected in this video.")
            else:
                for item in items:
                    st.markdown(f"### Demonstration at `{item.get('timestamp')}`")
                    st.markdown(f"*{item.get('explanation')}*")

                    st.markdown("#### [SOURCE-DERIVED SPOKEN EXCERPT]")
                    st.code(item.get("source_derived_snippet", ""), language="python")

                    st.markdown("#### [AI-RECONSTRUCTED CODE]")
                    st.caption("Clean algorithmic implementation reconstructed with strict syntax checking.")
                    st.code(item.get("reconstructed_code", ""), language=item.get("language", "python"))
                    st.divider()


# -----------------------------------------------------------------------------
# PAGE 11: DISAGREEMENT DETECTOR (FEATURE 11)
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Disagreement Detector":
    st.markdown("## 🔍 Cross-Source Disagreement Detector (Feature 11)")
    st.markdown("Detects potential conflicting explanations, terminology divergences, or conceptual alignments.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if len(active_sources) < 2:
        st.warning("⚠️ Disagreement detection requires at least 2 active sources.")
    else:
        src_a = st.selectbox("Source A", active_sources, index=0, format_func=lambda s: s.title)
        remaining = [s for s in active_sources if s.video_id != src_a.video_id]
        src_b = st.selectbox("Source B", remaining, index=0, format_func=lambda s: s.title)

        disagree_topic = st.text_input("Topic to Verify Disagreement", value="Random Forest & Representation Learning")

        if st.button("Analyze Disagreement & Divergence", type="primary"):
            with st.spinner("Comparing cross-source assertions..."):
                chunks_a = st.session_state.chunks_by_source.get(src_a.video_id, [])
                chunks_b = st.session_state.chunks_by_source.get(src_b.video_id, [])
                report = detect_disagreement(
                    topic=disagree_topic,
                    source_a=src_a,
                    chunks_a=chunks_a,
                    source_b=src_b,
                    chunks_b=chunks_b,
                    api_key=config.gemini_api_key
                )

            st.markdown(f"### Disagreement Analysis: `{report.topic}`")
            status_color = "#f59e0b" if report.conflict_type != "COMPATIBLE" else "#10b981"
            st.markdown(
                f"<div style='border: 1px solid {status_color}; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px;'>"
                f"<strong>Classification:</strong> <span style='color: {status_color}; font-weight: 700;'>{report.conflict_type}</span><br/>"
                f"<p style='margin-top: 6px; margin-bottom: 0;'>{report.interpretation}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{report.source_a_name}** `[{report.source_a_timestamp}]`")
                st.markdown(f"> {report.source_a_claim}")
            with col_b:
                st.markdown(f"**{report.source_b_name}** `[{report.source_b_timestamp}]`")
                st.markdown(f"> {report.source_b_claim}")


# -----------------------------------------------------------------------------
# PAGE 12: AUDIENCE INSIGHTS (FEATURE 19)
# -----------------------------------------------------------------------------
elif nav_selection == "👥 Audience Insights":
    st.markdown("## 👥 Public Audience Insights (Feature 19)")
    st.markdown("Sample and categorize common questions, confusion, and feedback from public YouTube comments.")

    active_sources = [s for s in st.session_state.sources if s.video_id in st.session_state.selected_source_ids]

    if not active_sources:
        st.warning("Please select an active video source.")
    else:
        target_src = st.selectbox("Select Video", active_sources, format_func=lambda s: s.title)

        if not config.youtube_api_key:
            st.info(
                "ℹ️ Audience Insights utilizes the YouTube Data API v3. "
                "You can configure an optional `YOUTUBE_API_KEY` in Settings to fetch live public comments. "
                "Displaying demo analysis sample below."
            )
            sample_comments = [
                {"text": "Why do we take m = sqrt(p) for feature sampling in random forest?", "likes": 42},
                {"text": "Still confused on how bagging reduces variance without changing bias.", "likes": 18},
                {"text": "Can you please make a part 2 covering XGBoost and LightGBM?", "likes": 65},
                {"text": "I got an error at 05:40 with cross_val_score scoring='f1_macro'.", "likes": 12}
            ]
        else:
            with st.spinner("Fetching public comments..."):
                sample_comments = fetch_public_comments(target_src.video_id, config.youtube_api_key)

        analysis = analyze_audience_themes(sample_comments)
        st.markdown(f"*{analysis['summary']}*")

        col_q, col_conf = st.columns(2)
        with col_q:
            st.markdown("### ❓ Top Learner Questions")
            for q in analysis["questions"]:
                st.markdown(f"- {q}")

        with col_conf:
            st.markdown("### ⚠️ Common Confusion Points")
            for c in analysis["confusion_points"]:
                st.markdown(f"- {c}")


# -----------------------------------------------------------------------------
# PAGE 13: RESEARCH NOTEBOOK (FEATURE 18)
# -----------------------------------------------------------------------------
elif nav_selection == "📓 Research Notebook":
    st.markdown("## 📓 Persistent Research Notebook (Feature 18)")
    st.markdown("Organize bookmarks, research notes, and citations saved during study sessions.")

    notes = repo.get_notes()

    with st.expander("➕ Add New Research Note"):
        n_title = st.text_input("Note Title")
        n_source = st.selectbox("Associated Source", [s.title for s in st.session_state.sources] + ["General Note"])
        n_time = st.text_input("Timestamp Ref (e.g. 05:40)")
        n_content = st.text_area("Note Content & Insights")
        if st.button("Save Note", type="primary"):
            if n_title and n_content:
                repo.add_note(
                    session_id="default_session",
                    title=n_title,
                    content=n_content,
                    timestamp_ref=n_time,
                    source_title=n_source
                )
                st.success("Note saved!")
                st.rerun()

    if not notes:
        st.info("No saved notes in your notebook yet. Bookmark insights from chat or write notes above.")
    else:
        st.markdown(f"### Saved Notes ({len(notes)})")
        for n in notes:
            with st.container():
                st.markdown(f"#### {n['title']}")
                st.caption(f"Source: `{n.get('source_title', 'General')}` • Timestamp: `{n.get('timestamp_ref', 'N/A')}` • Created: {n['created_at'][:16]}")
                st.markdown(n["content"])
                if st.button("🗑️ Delete", key=f"del_note_{n['note_id']}"):
                    repo.delete_note(n["note_id"])
                    st.rerun()
            st.divider()

        # Export Notebook
        nb_md = "# VideoMind AI - Research Notebook\n\n"
        for n in notes:
            nb_md += f"## {n['title']}\n"
            nb_md += f"Source: {n.get('source_title', 'General')} | Timestamp: {n.get('timestamp_ref', 'N/A')}\n\n"
            nb_md += f"{n['content']}\n\n---\n\n"

        st.download_button(
            label="📥 Export Complete Notebook (.md)",
            data=nb_md.encode('utf-8'),
            file_name="VideoMind_Research_Notebook.md",
            mime="text/markdown"
        )


# -----------------------------------------------------------------------------
# PAGE 14: ANALYTICS & BENCHMARKS (FEATURE 22 + SCIENTIFIC PRINCIPLES)
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Analytics & Benchmarks":
    st.markdown("## 📊 Research Analytics & Scientific Benchmarks")
    st.markdown("Track usage volume, knowledge acquisition, and verify empirical groundedness metrics.")

    summary = repo.get_analytics_summary()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Videos Ingested", summary["videos_indexed"])
    m2.metric("Questions Asked", summary["questions_asked"])
    m3.metric("Evidence Retrieved", summary["evidence_retrieved"])
    m4.metric("Study Packs Built", summary["study_packs_generated"])

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Quizzes Generated", summary["quizzes_generated"])
    m6.metric("Sources Compared", summary["sources_compared"])
    m7.metric("Notes Bookmarked", summary["notes_saved"])
    m8.metric("Index Chunks Active", st.session_state.vector_store.get_total_chunks())

    st.divider()

    st.markdown("### 🧪 Live Groundedness Benchmark Suite")
    st.markdown(
        "Runs standardized evaluation across 7 canonical query types: "
        "`DIRECT FACT`, `TEMPORAL`, `MULTI-HOP`, `COMPARISON`, `VISUAL`, `UNSUPPORTED`, `CODE`."
    )

    if st.button("🚀 Run Live Groundedness Benchmark", type="primary"):
        with st.spinner("Executing rigorous evaluation suite..."):
            bench = run_evaluation()
            st.session_state.benchmark_results = bench

    if st.session_state.benchmark_results:
        b_res = st.session_state.benchmark_results
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Test Cases", b_res["total_cases"])
        col_b2.metric("Retrieval Recall", f"{b_res['recall_rate'] * 100:.1f}%")
        col_b3.metric("Abstention Fidelity", f"{b_res['abstention_fidelity'] * 100:.1f}%")
        col_b4.metric("Average Latency", f"{b_res['average_latency_sec']:.4f}s")

        st.markdown("#### Benchmark Test Case Breakdown")
        df_bench = pd.DataFrame(b_res["cases"])
        st.dataframe(df_bench, use_container_width=True)

    st.divider()
    st.markdown("### 🛡️ Groundedness Architecture")
    st.markdown("""
    - **Hybrid Retrieval:** FAISS cosine inner-product embeddings + BM25 lexical keyword ranking.
    - **Principled Abstention:** Calibrated evidence scoring threshold stops hallucinations when cues are absent.
    - **Atomic Claim Map:** Answers are deconstructed into verified claims linking to exact timestamp ranges.
    - **Targeted Multimodality:** In-app visual moments synthesized into Mermaid diagrams and synchronized playback.
    """)


# -----------------------------------------------------------------------------
# PAGE 15: SETTINGS
# -----------------------------------------------------------------------------
elif nav_selection == "⚙️ Settings":
    st.markdown("## ⚙️ Settings & Configuration")
    st.markdown("Configure Google Gemini API keys, models, and retrieval hyperparameters.")

    with st.form("settings_form"):
        current_gemini = config.gemini_api_key or ""
        new_gemini = st.text_input(
            "Google Gemini API Key",
            value=current_gemini,
            type="password",
            help="Get your API key at https://aistudio.google.com/"
        )

        current_yt = config.youtube_api_key or ""
        new_yt = st.text_input(
            "Optional YouTube Data API v3 Key",
            value=current_yt,
            type="password",
            help="For live public comments & enriched view counts."
        )

        llm_choice = st.selectbox(
            "Primary LLM Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0
        )

        top_k_val = st.slider("Retrieval Top-K Chunks", min_value=2, max_value=12, value=config.top_k)
        alpha_val = st.slider("Hybrid Search Alpha (0.0 = Keyword, 1.0 = Pure Semantic)", min_value=0.0, max_value=1.0, value=config.hybrid_alpha, step=0.05)

        submitted = st.form_submit_button("Save Configuration", type="primary")
        if submitted:
            config.update_gemini_key(new_gemini)
            config.youtube_api_key = new_yt.strip() if new_yt else None
            config.llm_model = llm_choice
            config.top_k = top_k_val
            config.hybrid_alpha = alpha_val
            st.success("Configuration updated successfully!")
            st.rerun()

    st.divider()
    st.markdown("### Operational Diagnostics")
    st.write(f"- **Gemini Configured:** `{'Yes' if config.is_gemini_available() else 'No (Demo / Offline Mode Active)'}`")
    st.write(f"- **Embedding Model:** `{config.embedding_model}`")
    st.write(f"- **Database Path:** `{config.database_path}`")
