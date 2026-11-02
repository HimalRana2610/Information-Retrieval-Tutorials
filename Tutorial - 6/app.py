"""
Streamlit UI for Tutorial 6 — Fake News Detection & Intelligent News Retrieval.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    ARTICLES,
    Embedder,
    LSAEmbedder,
    full_text,
    load_glove,
    ltr_rank,
    predict_misleading,
    semantic_search,
    style_features,
    train_classifiers,
    train_ltr,
    TEST_QUERIES,
)


@st.cache_resource(show_spinner="Loading models and embeddings...")
def load_system():
    ids = sorted(ARTICLES.keys())
    bundle = train_classifiers(ids, ids)
    misleading_proba = {
        aid: float(bundle["results"]["Ensemble"]["proba"][i])
        for i, aid in enumerate(bundle["test_ids"])
    }
    glove = load_glove()
    emb = Embedder(ids, glove) if glove else LSAEmbedder(ids)

    train_pairs = []
    for _, (qtext, rel) in TEST_QUERIES.items():
        for aid in ids:
            train_pairs.append((qtext, aid, 1 if aid in rel else 0))
    ltr_model = train_ltr(emb, train_pairs, misleading_proba)
    return ids, bundle, misleading_proba, emb, ltr_model


def main():
    st.set_page_config(
        page_title="News Credibility & Retrieval",
        page_icon="📰",
        layout="wide",
    )
    st.title("AI-Powered Fake News Detection & Intelligent News Retrieval")
    st.caption(
        "Tutorial 6 · U23AI053 · Predictions flag articles as "
        "*potentially misleading* — they do not prove a story is false."
    )

    ids, bundle, misleading_proba, emb, ltr_model = load_system()

    tab_search, tab_analyse, tab_browse, tab_metrics = st.tabs(
        ["Semantic Search", "Analyse Article", "Browse Collection", "Metrics"]
    )

    with tab_search:
        st.subheader("Search news by meaning")
        q = st.text_input(
            "Query",
            value="electric vehicle battery fires and safety",
            help="Example: vaccine side effects, ransomware phishing, climate warming",
        )
        method = st.radio(
            "Ranking method",
            ["Learning to Rank", "Semantic (GloVe)", "Both"],
            horizontal=True,
        )
        if st.button("Search", type="primary") and q.strip():
            cols = st.columns(2 if method == "Both" else 1)
            if method in ("Semantic (GloVe)", "Both"):
                with cols[0]:
                    st.markdown("#### Semantic ranking")
                    for i, (aid, s) in enumerate(semantic_search(q, emb, ids), 1):
                        _render_hit(aid, s, misleading_proba, score_name="sim")
            if method in ("Learning to Rank", "Both"):
                with cols[-1]:
                    st.markdown("#### Learning-to-Rank")
                    for i, (aid, s) in enumerate(
                        ltr_rank(q, ids, emb, ltr_model, misleading_proba), 1
                    ):
                        _render_hit(aid, s, misleading_proba, score_name="LTR")

    with tab_analyse:
        st.subheader("Analyse an article or paste text")
        sample = ARTICLES["A09"]
        text = st.text_area(
            "Article text",
            value=f"{sample['title']}. {sample['body']}",
            height=180,
        )
        c1, c2 = st.columns(2)
        source = c1.text_input("Source domain", value=sample["source"])
        credibility = c2.slider("Prior source credibility", 0.0, 1.0, float(sample["credibility"]), 0.01)
        if st.button("Run AI Analysis", type="primary"):
            lab, conf, parts = predict_misleading(text, source, credibility, bundle)
            feats = style_features(text, source, credibility)

            left, right = st.columns([1, 1.2])
            with left:
                if lab:
                    st.error(f"Verdict: POTENTIALLY MISLEADING\n\nConfidence {conf:.0%}")
                else:
                    st.success(f"Verdict: likely credible\n\nConfidence of misleading {conf:.0%}")
                st.progress(conf)
                st.caption("0% → likely credible · 100% → potentially misleading")
            with right:
                st.markdown("#### Model votes")
                for m, p in parts.items():
                    st.write(f"**{m}**: {p:.0%}")
                st.markdown("#### Engineered features")
                st.json({k: round(v, 3) if isinstance(v, float) else v
                         for k, v in feats.items()})
            st.info(
                "Important: Model predictions alone cannot establish whether a "
                "news story is true. Treat this as a triage signal for review."
            )

    with tab_browse:
        st.subheader("Collection")
        filt = st.multiselect(
            "Category",
            sorted({a["category"] for a in ARTICLES.values()}),
            default=None,
        )
        show_m = st.checkbox("Only potentially misleading", value=False)
        rows = []
        for aid in ids:
            a = ARTICLES[aid]
            if filt and a["category"] not in filt:
                continue
            if show_m and misleading_proba[aid] < 0.5:
                continue
            rows.append({
                "ID": aid,
                "Category": a["category"],
                "Source": a["source"],
                "P(misleading)": round(misleading_proba[aid], 3),
                "Title": a["title"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_metrics:
        st.subheader("Held-out classification (refit view on full set)")
        st.write(
            "For the official train/test split metrics, run `python main.py` "
            "and see Step 2. Plots:"
        )
        for name in ("classifier_comparison.png", "pca_articles.png",
                     "tsne_articles.png"):
            path = os.path.join(os.path.dirname(__file__), name)
            if os.path.exists(path):
                st.image(path, caption=name)


def _render_hit(aid, score, misleading_proba, score_name="score"):
    a = ARTICLES[aid]
    p = misleading_proba[aid]
    label = "⚠ potentially misleading" if p >= 0.5 else "✓ likely credible"
    st.markdown(
        f"**{aid}** · {score_name}={score:.3f} · {label} ({p:.0%})  \n"
        f"{a['title']}  \n"
        f"<small>{a['source']} · {a['category']} · {a['published']}</small>",
        unsafe_allow_html=True,
    )
    with st.expander("Show text"):
        st.write(full_text(aid))


if __name__ == "__main__":
    main()
