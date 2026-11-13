"""A vintage academic dashboard for the DuckDB + DeepSeek sentiment ledger."""

from __future__ import annotations

import html
import logging
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

from backend import PipelineError, run_pipeline

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="The Sentiment Ledger",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
    --parchment: #f4f1ea;
    --paper: #fbf8f0;
    --ink: #2b2523;
    --brown: #5b3a29;
    --brown-soft: #8c6a52;
    --faded: #6f6154;
    --rule: #c9b99a;
    --rule-soft: #ded2bd;
    --positive: #3f5a3a;
    --negative: #7a2e25;
}

html, body, .stApp {
    background-color: var(--parchment);
    color: var(--ink);
}

.stApp {
    background-image:
        radial-gradient(circle at 12% 8%, rgba(91, 58, 41, 0.045) 0, transparent 32%),
        radial-gradient(circle at 88% 78%, rgba(91, 58, 41, 0.035) 0, transparent 38%);
    font-family: 'Lora', Georgia, serif;
}

.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp table, .stApp td, .stApp th {
    font-family: 'Lora', Georgia, serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.0rem;
    padding-bottom: 2.8rem;
}

[data-testid="stSidebar"] {
    background: #ebe4d5;
    border-right: 1px solid var(--rule);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.8rem;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: var(--faded);
    font-size: 0.86rem;
    line-height: 1.65;
}

[data-testid="stSidebar"] h3 {
    font-size: 1.05rem;
    margin: 1.1rem 0 0.35rem;
}

[data-testid="stTextInput"] label {
    font-size: 0.66rem !important;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--faded) !important;
    font-weight: 500 !important;
}

[data-testid="stTextInput"] input {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1px solid #bfae91 !important;
    border-radius: 1px !important;
    padding: 0.68rem 0.75rem !important;
    font-size: 0.9rem !important;
    box-shadow: inset 0 1px 2px rgba(43, 37, 35, 0.04) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--brown) !important;
    box-shadow: 0 0 0 1px var(--brown) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #a2927f !important;
}

[data-testid="stTextInputRootElement"] > div {
    background: transparent !important;
}

[data-testid="stTextInput"] button {
    background: var(--paper) !important;
    border: none !important;
    border-left: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--brown) !important;
}

[data-testid="stTextInput"] button:hover {
    background: #f1eadb !important;
}

[data-testid="stTextInput"] button svg {
    color: var(--brown) !important;
    fill: var(--brown) !important;
}

.stButton > button {
    width: 100%;
    background: var(--ink);
    color: var(--parchment);
    border: 1px solid var(--ink);
    border-radius: 1px;
    padding: 0.78rem 1.2rem;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.82rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    transition: background 180ms ease, color 180ms ease, transform 180ms ease;
    box-shadow: 0 8px 22px rgba(43, 37, 35, 0.12);
}

.stButton > button:hover {
    background: var(--brown);
    border-color: var(--brown);
    color: var(--paper);
}

.stButton > button:active {
    transform: translateY(1px);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--paper);
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    box-shadow: 0 12px 32px rgba(43, 37, 35, 0.055);
}

[data-testid="stPlotlyChart"] {
    background: transparent;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero {
    text-align: center;
    padding: 1.0rem 0 1.55rem;
    animation: fadeUp 0.9s cubic-bezier(0.2, 0.7, 0.2, 1) both;
}

.hero-kicker {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.85rem;
    font-size: 0.64rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--brown);
}

.hero-kicker::before, .hero-kicker::after {
    content: "";
    width: 42px;
    height: 1px;
    background: var(--rule);
}

.hero h1 {
    font-size: clamp(2.8rem, 6vw, 5.1rem) !important;
    line-height: 0.98 !important;
    letter-spacing: -0.025em;
    margin: 0.55rem 0 0.6rem 0 !important;
}

.hero h1 em {
    color: var(--brown);
    font-style: italic;
}

.hero-sub {
    max-width: 650px;
    margin: 0 auto !important;
    color: #5f5348 !important;
    font-size: 1.02rem !important;
    line-height: 1.75 !important;
}

.hero-meta {
    margin-top: 1.0rem;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7a6a59;
}

.hero-ornament {
    margin-top: 1.05rem;
    color: var(--brown);
    font-size: 1.1rem;
    letter-spacing: 0.5em;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1.35rem 0 1.25rem;
    animation: fadeUp 0.7s 0.08s both;
}

.stat {
    position: relative;
    background: var(--paper);
    border: 1px solid var(--rule);
    padding: 1.05rem 1.2rem 0.95rem;
    overflow: hidden;
}

.stat::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: var(--brown);
    opacity: 0.82;
}

.stat-label {
    display: block;
    color: var(--faded);
    font-size: 0.62rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
}

.stat-value {
    display: block;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.25rem;
    line-height: 1.1;
    margin: 0.28rem 0 0.1rem;
    color: var(--ink);
}

.stat-note {
    display: block;
    color: var(--faded);
    font-size: 0.77rem;
}

.panel-kicker {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--brown);
    margin-bottom: 0.28rem;
}

.panel-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.5rem;
    line-height: 1.15;
    margin: 0 0 0.18rem;
    color: var(--ink);
}

.panel-sub {
    font-size: 0.83rem;
    line-height: 1.55;
    color: var(--faded);
    margin: 0;
}

.empty-card {
    background: var(--paper);
    border: 1px solid var(--rule);
    box-shadow: 0 18px 48px rgba(43, 37, 35, 0.07);
    text-align: center;
    padding: 2.35rem 1.5rem 2.1rem;
    margin-top: 1.15rem;
    position: relative;
    animation: fadeUp 0.8s 0.12s both;
}

.empty-card::after {
    content: "";
    position: absolute;
    inset: 7px;
    border: 1px solid rgba(201, 185, 154, 0.55);
    pointer-events: none;
}

.empty-ornament {
    color: var(--brown);
    font-size: 1.8rem;
    line-height: 1;
}

.empty-card h2 {
    font-size: 2.05rem !important;
    margin: 0.5rem 0 0.35rem !important;
}

.empty-card > p {
    max-width: 560px;
    margin: 0 auto !important;
    color: var(--faded) !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

.steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.2rem;
    max-width: 850px;
    margin: 1.8rem auto 0;
    text-align: left;
}

.step {
    border-top: 1px solid var(--rule);
    padding-top: 0.75rem;
}

.step-num {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.25rem;
    color: var(--brown);
}

.step h4 {
    font-size: 0.98rem !important;
    margin: 0.18rem 0 0.25rem !important;
}

.step p {
    color: var(--faded) !important;
    font-size: 0.78rem !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}

.notice {
    background: #f8ece6;
    border-left: 3px solid var(--negative);
    padding: 0.95rem 1.1rem;
    margin: 0.9rem 0 0.7rem;
    color: #5d2b23;
    font-size: 0.88rem;
    line-height: 1.6;
    animation: fadeUp 0.5s both;
}

.notice strong {
    font-family: 'Playfair Display', Georgia, serif;
    letter-spacing: 0.02em;
}

.ledger-shell {
    background: var(--paper);
    border: 1px solid var(--rule);
    box-shadow: 0 18px 48px rgba(43, 37, 35, 0.07);
    margin-top: 1.2rem;
    animation: fadeUp 0.8s 0.16s both;
}

.ledger-top {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.2rem 1.4rem 0.95rem;
    border-bottom: 1px solid var(--rule);
}

.ledger-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.6rem;
    line-height: 1.1;
    margin: 0.12rem 0 0.18rem;
}

.ledger-sub {
    color: var(--faded);
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0;
}

.ledger-count {
    color: var(--faded);
    font-size: 0.64rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    white-space: nowrap;
    padding-bottom: 0.15rem;
}

.table-scroll {
    max-height: 575px;
    overflow: auto;
}

.ledger-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
}

.ledger-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #f1eadb;
    color: var(--faded);
    font-family: 'Lora', Georgia, serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-align: left;
    padding: 0.75rem 0.9rem;
    border-bottom: 1px solid var(--rule);
    white-space: nowrap;
}

.ledger-table tbody td {
    padding: 0.72rem 0.9rem;
    border-bottom: 1px solid var(--rule-soft);
    vertical-align: top;
    color: #3a322d;
}

.ledger-table tbody tr:last-child td {
    border-bottom: none;
}

.ledger-table tbody tr:hover td {
    background: #f7f1e5;
}

.id-cell {
    color: #9b8b79 !important;
    font-size: 0.72rem !important;
    font-variant-numeric: tabular-nums;
    width: 42px;
}

.product-cell {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 0.94rem !important;
    min-width: 150px;
}

.rating-cell {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.rating-cell strong {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.02rem;
    font-weight: 600;
}

.rating-cell span {
    color: #9b8b79;
    font-size: 0.72rem;
}

.avg-cell {
    color: var(--faded) !important;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.badge {
    display: inline-block;
    padding: 0.18rem 0.5rem;
    border: 1px solid;
    font-size: 0.6rem;
    letter-spacing: 0.11em;
    line-height: 1.4;
    text-transform: uppercase;
    white-space: nowrap;
}

.badge.pos {
    color: var(--positive);
    border-color: #9caf8f;
    background: #eef2e6;
}

.badge.neg {
    color: var(--negative);
    border-color: #c69a8f;
    background: #f7e9e4;
}

.badge.neutral {
    color: var(--faded);
    border-color: var(--rule);
    background: #f3ede1;
}

.review-cell {
    min-width: 290px;
    max-width: 460px;
    line-height: 1.55 !important;
    color: #3a322d !important;
}

.marginalia {
    background: var(--paper);
    border: 1px solid var(--rule);
    box-shadow: 0 12px 32px rgba(43, 37, 35, 0.055);
    padding: 1.15rem 1.25rem 1.0rem;
    height: 100%;
}

.marginalia h3 {
    font-size: 1.55rem !important;
    line-height: 1.12;
    margin: 0.12rem 0 0.85rem !important;
}

.margin-item {
    border-top: 1px solid var(--rule-soft);
    padding: 0.68rem 0 0.55rem;
}

.margin-label {
    display: block;
    color: var(--brown);
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.margin-product {
    display: block;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.06rem;
    line-height: 1.25;
    margin: 0.16rem 0 0.08rem;
}

.margin-note {
    display: block;
    color: var(--faded);
    font-size: 0.76rem;
    line-height: 1.45;
}

.margin-caveat {
    border-top: 1px solid var(--rule);
    margin: 0.75rem 0 0 !important;
    padding-top: 0.7rem;
    color: #7a6a59 !important;
    font-size: 0.74rem !important;
    font-style: italic;
    line-height: 1.55 !important;
}

.sidebar-brand {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.55rem;
    line-height: 1.08;
    color: var(--ink);
    margin-bottom: 0.2rem;
}

.sidebar-eyebrow {
    font-size: 0.6rem;
    letter-spacing: 0.19em;
    text-transform: uppercase;
    color: var(--brown);
}

.sidebar-rule {
    height: 1px;
    background: var(--rule);
    margin: 1.05rem 0;
}

.sidebar-note {
    color: var(--faded);
    font-size: 0.76rem;
    line-height: 1.6;
}

.sidebar-steps {
    margin: 0.55rem 0 0;
    padding-left: 1.05rem;
}

.sidebar-steps li {
    margin-bottom: 0.42rem;
    padding-left: 0.12rem;
    font-size: 0.78rem !important;
}

.sidebar-steps li::marker {
    color: var(--brown);
    font-family: 'Playfair Display', Georgia, serif;
}

.footer {
    border-top: 1px solid var(--rule);
    margin-top: 2.0rem;
    padding-top: 0.9rem;
    text-align: center;
    color: #7a6a59;
    font-size: 0.7rem;
    letter-spacing: 0.035em;
    line-height: 1.7;
}

.footer a {
    color: var(--brown);
    text-decoration: none;
    border-bottom: 1px solid rgba(91, 58, 41, 0.35);
}

@media (max-width: 860px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .stat-grid { grid-template-columns: 1fr; }
    .steps { grid-template-columns: 1fr; gap: 0.8rem; }
    .hero h1 { font-size: 2.9rem !important; }
    .ledger-top { display: block; }
    .ledger-count { display: block; margin-top: 0.5rem; }
    .ledger-table { min-width: 780px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _escape(value: object) -> str:
    return html.escape(str(value)).replace("$", "&#36;")


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("run_at", None)


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-eyebrow">Field Notes · No. 04</div>
            <div class="sidebar-brand">The Reading Room</div>
            <div class="sidebar-rule"></div>
            """,
            unsafe_allow_html=True,
        )

        api_key = st.text_input(
            "AI API key",
            type="password",
            placeholder="sk-…",
        )

        st.markdown(
            """
            <p class="sidebar-note">The key is held in this session, used for
            <strong>LLM inference</strong>, and never logged or saved.</p>
            <div class="sidebar-rule"></div>
            <div class="sidebar-eyebrow">Method</div>
            <ol class="sidebar-steps">
                <li><strong>DuckDB</strong> reads the CSV and computes each
                product’s mean rating.</li>
                <li><strong>The Language Model</strong> reads every review and returns
                <em>Positive</em> or <em>Negative</em>.</li>
                <li>The two results are joined back at review grain.</li>
            </ol>
            <div class="sidebar-rule"></div>
            <p class="sidebar-note"><strong>No RAG.</strong> No vector store.
            Just relational SQL, a careful prompt, and close reading.</p>
            """,
            unsafe_allow_html=True,
        )

    return api_key


def _run_button() -> bool:
    _, center, _ = st.columns([1, 1.25, 1])
    with center:
        return st.button("Run Analysis", width="stretch")


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">A Study in Applied Semantics</div>
            <h1>The Sentiment <em>Ledger</em></h1>
            <p class="hero-sub">Product reviews, read closely by a language model
            and tallied in DuckDB — a small, honest bridge between relational
            analytics and semantic judgment.</p>
            <div class="hero-meta">CPU-only · Windows · LLM API · No RAG</div>
            <div class="hero-ornament">❦</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_notice(message: str | None) -> None:
    if message:
        st.markdown(
            f'<div class="notice"><strong>Analysis interrupted.</strong> '
            f"{_escape(message)}</div>",
            unsafe_allow_html=True,
        )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-ornament">❦</div>
            <h2>The ledger is waiting.</h2>
            <p>Add your AI key in the margin, then ask the model to read
            the reviews. DuckDB will do the counting; the model will do the
            close reading.</p>
            <div class="steps">
                <div class="step">
                    <div class="step-num">I.</div>
                    <h4>Read</h4>
                    <p>DuckDB scans <em>data/reviews_real.csv</em> and preserves one
                    row per review.</p>
                </div>
                <div class="step">
                    <div class="step-num">II.</div>
                    <h4>Read closely</h4>
                    <p>The model classifies each distinct review as Positive or
                    Negative, one careful prompt at a time.</p>
                </div>
                <div class="step">
                    <div class="step-num">III.</div>
                    <h4>Reconcile</h4>
                    <p>The verdicts are merged with product averages and set
                    into the ledger below.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # ADDED: Show real data immediately so the page isn't empty
    st.markdown(
        """
        <div class="panel-kicker" style="margin-top: 2rem;">Data Preview</div>
        <div class="panel-title">The raw ledger</div>
        <p class="panel-sub">Here are the raw customer reviews. Enter your API key in the sidebar and click "Run Analysis" to apply LLM semantic tagging.</p>
        """,
        unsafe_allow_html=True
    )
    try:
        raw_df = pd.read_csv("data/reviews_real.csv")
        st.dataframe(raw_df.head(20), use_container_width=True, height=300)
        st.caption("Showing the first 20 rows of the dataset.")
    except Exception as e:
        st.error(f"Could not load data: {e}. Make sure data/reviews_real.csv exists.")


def _render_metrics(frame: pd.DataFrame) -> None:
    total = len(frame)
    positive = int((frame["sentiment"] == "Positive").sum())
    share = positive / total if total else 0.0
    mean_rating = float(frame["rating"].mean())
    products = int(frame["product"].nunique())

    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat">
                <span class="stat-label">Reviews read</span>
                <span class="stat-value">{total}</span>
                <span class="stat-note">across {products} products</span>
            </div>
            <div class="stat">
                <span class="stat-label">Positive share</span>
                <span class="stat-value">{share:.0%}</span>
                <span class="stat-note">{positive} of {total} reviews</span>
            </div>
            <div class="stat">
                <span class="stat-label">Mean rating</span>
                <span class="stat-value">{mean_rating:.1f}<span style="font-size:1rem;color:#9b8b79;"> / 5</span></span>
                <span class="stat-note">the rating the model did not see</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sentiment_figure(frame: pd.DataFrame):
    counts = frame["sentiment"].value_counts()
    labels = ["Positive", "Negative"]
    values = [int(counts.get(label, 0)) for label in labels]

    figure = px.bar(
        x=labels,
        y=values,
        color=labels,
        color_discrete_map={"Positive": "#5b3a29", "Negative": "#9c5a45"},
        text=values,
    )
    figure.update_traces(
        marker_line_color="#2b2523",
        marker_line_width=0.6,
        width=0.56,
        textposition="outside",
        textfont=dict(family="Lora, Georgia, serif", color="#2b2523", size=14),
        cliponaxis=False,
    )
    figure.update_layout(
        paper_bgcolor="#f4f1ea",
        plot_bgcolor="#f4f1ea",
        font=dict(family="Lora, Georgia, serif", color="#2b2523", size=12),
        margin=dict(l=12, r=16, t=18, b=8),
        showlegend=False,
        bargap=0.45,
        height=320,
    )
    figure.update_xaxes(
        title=None,
        showgrid=False,
        zeroline=False,
        linecolor="#8c6a52",
        tickcolor="#8c6a52",
        tickfont=dict(
            family="Playfair Display, Georgia, serif",
            size=13,
            color="#2b2523",
        ),
    )
    figure.update_yaxes(
        title=None,
        showgrid=True,
        gridcolor="rgba(43, 37, 35, 0.08)",
        zeroline=False,
        rangemode="tozero",
        linecolor="#8c6a52",
        tickcolor="#8c6a52",
        tickfont=dict(family="Lora, Georgia, serif", size=11, color="#2b2523"),
        dtick=max(1, (max(values) + 3) // 4),
    )
    return figure


def _render_chart(frame: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-kicker">Figure I</div>
            <div class="panel-title">The tally</div>
            <p class="panel-sub">Model-assigned sentiment across the reviewed
            catalogue.</p>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _sentiment_figure(frame),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )


def _render_marginalia(frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby("product", as_index=False)
        .agg(
            reviews=("review_id", "count"),
            mean_rating=("rating", "mean"),
            positive=("sentiment", lambda values: int((values == "Positive").sum())),
        )
        .assign(positive_share=lambda data: data["positive"] / data["reviews"])
    )

    most_praised = summary.sort_values(
        ["positive_share", "mean_rating", "product"],
        ascending=[False, False, True],
    ).iloc[0]
    most_critical = summary.sort_values(
        ["positive_share", "mean_rating", "product"],
        ascending=[True, True, True],
    ).iloc[0]

    def item(label: str, row: pd.Series) -> str:
        return (
            '<div class="margin-item">'
            f'<span class="margin-label">{label}</span>'
            f'<span class="margin-product">{_escape(row["product"])}</span>'
            f'<span class="margin-note">{row["positive_share"]:.0%} positive · '
            f'{row["mean_rating"]:.2f}/5 mean rating</span>'
            "</div>"
        )

    st.markdown(
        '<div class="marginalia">'
        '<div class="panel-kicker">Marginalia</div>'
        "<h3>What the ledger suggests</h3>"
        + item("Most praised", most_praised)
        + item("Least positive", most_critical)
        + '<p class="margin-caveat">Sentiment is model-assigned, not ground '
        "truth. Read the reviews, not only the tally.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_ledger(frame: pd.DataFrame, run_at: str | None) -> None:
    rows: list[str] = []
    for _, row in frame.iterrows():
        sentiment = str(row["sentiment"])
        badge_class = (
            "pos"
            if sentiment == "Positive"
            else "neg"
            if sentiment == "Negative"
            else "neutral"
        )
        rows.append(
            "<tr>"
            f'<td class="id-cell">{_escape(row["review_id"])}</td>'
            f'<td class="product-cell">{_escape(row["product"])}</td>'
            f'<td class="rating-cell"><strong>{row["rating"]:.0f}</strong>'
            "<span>/5</span></td>"
            f'<td class="avg-cell">{row["avg_product_rating"]:.2f}</td>'
            f'<td><span class="badge {badge_class}">{_escape(sentiment)}</span></td>'
            f'<td class="review-cell">{_escape(row["review_text"])}</td>'
            "</tr>"
        )

    stamp = _escape(run_at) if run_at else "just now"
    st.markdown(
        '<div class="ledger-shell">'
        '<div class="ledger-top">'
        "<div>"
        '<div class="panel-kicker">Exhibit II</div>'
        '<div class="ledger-title">The ledger</div>'
        '<p class="ledger-sub">Every review, its rating, its product average, '
        "and the model’s reading.</p>"
        "</div>"
        f'<div class="ledger-count">{len(frame)} rows · {stamp}</div>'
        "</div>"
        '<div class="table-scroll">'
        '<table class="ledger-table">'
        "<thead><tr>"
        "<th>No.</th><th>Product</th><th>Rating</th><th>Product avg.</th>"
        "<th>Sentiment</th><th>Review</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _init_state()
    api_key = _render_sidebar()
    _render_hero()

    clicked = _run_button()
    if clicked:
        if not api_key or not api_key.strip():
            st.session_state["error"] = (
                "Enter your AI API key in the sidebar before running the analysis."
            )
            st.session_state["result"] = None
        else:
            with st.spinner("Reading the reviews, one careful pass at a time…"):
                try:
                    result = run_pipeline(api_key)
                except PipelineError as exc:
                    st.session_state["error"] = str(exc)
                    st.session_state["result"] = None
                except Exception:
                    logger.exception("Unexpected pipeline failure")
                    st.session_state["error"] = (
                        "Something unexpected interrupted the analysis. "
                        "Check the server log and try again."
                    )
                    st.session_state["result"] = None
                else:
                    st.session_state["result"] = result
                    st.session_state["error"] = None
                    st.session_state["run_at"] = datetime.now().strftime(
                        "%d %b %Y · %H:%M"
                    )

    _render_notice(st.session_state.get("error"))

    result = st.session_state.get("result")
    if isinstance(result, pd.DataFrame) and not result.empty:
        _render_metrics(result)
        chart_column, margin_column = st.columns([1.55, 1], gap="large")
        with chart_column:
            _render_chart(result)
        with margin_column:
            _render_marginalia(result)
        _render_ledger(result, st.session_state.get("run_at"))
    elif not st.session_state.get("error"):
        _render_empty_state()

    st.markdown(
        """
        <div class="footer">
            An adaptation of <a href="https://arxiv.org/abs/2504.01157"
            target="_blank" rel="noopener">Beyond Quacking</a> (arXiv:2504.01157)
            · Built with DuckDB, Streamlit, and a regard for good
            typography.<br>
            LLM sentiment is a reading, not a verdict. Validate before you act.
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()