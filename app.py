"""
DATS 6401 - Homework 7: Text Data
What is AI-agent research about?  ~15,000 arXiv paper summaries (2023-2026).

Run:  streamlit run app.py
Data: data/papers.parquet + data/topics.json, built by prepare_data.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

st.set_page_config(page_title="AI-Agent Research Text Explorer", page_icon="🤖", layout="wide")

HERE = Path(__file__).parent
# Data files may sit in data/ or right next to app.py (easier GitHub upload)
DATA = HERE / "data" if (HERE / "data" / "papers.parquet").exists() else HERE
BLUE, ORANGE, GREY, LIGHT = "#0072B2", "#D55E00", "#9AA5B1", "#D9DEE3"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Words the source list's summary template repeats in nearly every entry.
TEMPLATE_WORDS = {
    "introduces", "introduce", "framework", "utilizes", "utilize", "uses", "use",
    "using", "demonstrates", "demonstrate", "approach", "method", "methods",
    "proposes", "propose", "paper", "novel", "results", "based", "enables",
    "achieves", "performance", "new", "significantly", "existing", "et", "al",
}


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    if not (DATA / "papers.parquet").exists() or not (DATA / "topics.json").exists():
        st.error("Data files not found. Upload papers.parquet and topics.json to the repo, "
                 "in the same place as app.py (or inside a data/ folder).")
        st.stop()
    df = pd.read_parquet(DATA / "papers.parquet")
    df["text"] = df["title"] + ". " + df["summary"]
    meta = json.load(open(DATA / "topics.json"))
    return df, meta


@st.cache_data
def word_counts(texts: tuple, ngram: int, drop_template: bool):
    stop = list(ENGLISH_STOP_WORDS | TEMPLATE_WORDS) if drop_template else "english"
    cv = CountVectorizer(stop_words=stop, ngram_range=(ngram, ngram), min_df=3)
    X = cv.fit_transform(texts)
    return pd.Series(np.asarray(X.sum(axis=0)).ravel(), index=cv.get_feature_names_out())


def kwic(texts, word, width=60, n=12):
    """Keyword in context: the sentences behind a bar."""
    pat = re.compile(rf"\b{re.escape(word)}\b", re.I)
    lines = []
    for t in texts:
        for m in pat.finditer(t):
            left = t[max(0, m.start() - width):m.start()].rjust(width)
            right = t[m.end():m.end() + width]
            lines.append(f"{left} [{t[m.start():m.end()].upper()}] {right}")
            break
        if len(lines) >= n:
            break
    return lines


def note(reveals, misleads):
    c1, c2 = st.columns(2)
    c1.success(f"**What it reveals.** {reveals}")
    c2.warning(f"**How it can mislead.** {misleads}")


df, meta = load()
topics = {t["topic"]: t for t in meta["topics"]}
label = {t: v["label"] for t, v in topics.items()}
label[-1] = "-1: outliers (no topic)"

# ------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
years = st.sidebar.multiselect("Years", [2023, 2024, 2025, 2026], default=[2023, 2024, 2025, 2026],
                               help="arXiv submission year, from the paper ID")
order = sorted(topics, key=lambda t: -topics[t]["size"])
sel_topic = st.sidebar.selectbox("Topic", order + [-1], format_func=lambda t: label[t],
                                 help="Used in the Topics and Embedding map tabs")
st.sidebar.caption(
    "Corpus: summaries of AI-agent papers on arXiv from the open "
    "[Autonomous-Agents](https://github.com/tmgthb/Autonomous-Agents) list "
    "(Teemu Maatta, MIT licence). Text = title + 1-3 bullet summary."
)
d = df[df["year"].isin(years)] if years else df

# ---------------------------------------------------------------- header
st.title("🤖 What is AI-agent research about?")
st.markdown(
    "Three views of the same corpus, each built on a different **transform** of the text: "
    "word **counts**, **topics** (TF-IDF → UMAP → HDBSCAN → c-TF-IDF), and an **embedding projection**."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers (filtered)", f"{len(d):,}")
m2.metric("Topics found", meta["n_topics"])
m3.metric("Outliers, topic −1", f"{meta['outlier_share']:.0%}")
m4.metric("Median words / paper", int(d["text"].str.split().str.len().median()) if len(d) else 0)

if d.empty:
    st.info("Pick at least one year in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["① Frequency", "② Topics", "③ Embedding map", "④ Notes"])

# ===================================================== 1. FREQUENCY VIEW
with tab1:
    st.subheader("Which words are most frequent?")
    c1, c2, c3 = st.columns(3)
    ngram = c1.radio("Unit", [1, 2], format_func=lambda n: "single words" if n == 1 else "two-word phrases",
                     horizontal=True)
    drop_tpl = c2.toggle("Also drop template words", value=True,
                         help="'introduces', 'framework', 'utilizes'... come from how the list is written")
    top_n = c3.slider("Top N", 10, 40, 20)

    counts = word_counts(tuple(d["text"]), ngram, drop_tpl).nlargest(top_n).sort_values()
    fig = px.bar(counts, orientation="h", labels={"value": "occurrences", "index": ""},
                 color_discrete_sequence=[BLUE])
    fig.update_layout(showlegend=False, height=28 * top_n + 80, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Sorted bars, {len(d):,} papers, sklearn English stop words"
               + (" + template words" if drop_tpl else "") + ", no stemming, terms in ≥ 3 papers.")

    st.markdown("#### Keyness: what is distinctive of 2026 vs 2025?")
    early, late = df[df.year == 2025]["text"], df[df.year == 2026]["text"]
    ca = word_counts(tuple(late), ngram, True)
    cb = word_counts(tuple(early), ngram, True)
    vocab = ca.index.union(cb.index)
    ca, cb = ca.reindex(vocab, fill_value=0), cb.reindex(vocab, fill_value=0)
    lr = np.log2(((ca + 0.5) / ca.sum()) / ((cb + 0.5) / cb.sum()))
    min_total = 300 if ngram == 1 else 60   # keep common terms only
    lr = lr[(ca + cb) >= min_total]
    key = pd.concat([lr.nlargest(12), lr.nsmallest(12)]).sort_values()
    fig = go.Figure(go.Bar(x=key.values, y=key.index, orientation="h",
                           marker_color=[ORANGE if v > 0 else BLUE for v in key.values]))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="log₂ ratio  (← more 2025 · more 2026 →)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"log-ratio = log₂[(c₂₀₂₆+0.5)/N₂₀₂₆ ÷ (c₂₀₂₅+0.5)/N₂₀₂₅], terms with ≥ {min_total} uses. "
               "Ignores the year filter so the two groups stay fixed.")

    st.markdown("#### Concordance: read the lines behind a bar")
    word = st.text_input("Word or phrase", value="memory")
    lines = kwic(d["text"].tolist(), word.strip()) if word.strip() else []
    st.code("\n".join(lines) if lines else "No matches.", language=None)

    note("A handful of words carry the corpus: agent, LLM, multi-agent, reasoning, memory. "
         "Keyness shows 2026 papers talking about harnesses, skills and frontier models, "
         "and 2025 papers about surveys and 'LLM-powered' agents.",
         "Turn off the template-word toggle: 'framework', 'introduces' and 'utilizes' jump into the top 10. "
         "They describe how the list is written, not the research. The same trap hits keyness: 'aims', "
         "'stores', 'enhancing' mark 2025 because the curator's wording changed. Counts also split one idea "
         "across spellings (agent / agents / agentic) because nothing is stemmed.")

# ======================================================== 2. TOPIC VIEW
with tab2:
    st.subheader("Topics found by the pipeline")
    sizes = d["topic"].value_counts()
    sz = pd.DataFrame({"topic": sizes.index, "papers": sizes.values})
    sz["label"] = sz["topic"].map(label)
    sz["kind"] = np.where(sz["topic"] == -1, "outliers", np.where(sz["topic"] == sel_topic, "selected", "topic"))
    sz = sz.sort_values("papers")
    fig = px.bar(sz, x="papers", y="label", orientation="h", color="kind",
                 color_discrete_map={"topic": GREY, "selected": ORANGE, "outliers": LIGHT})
    fig.update_layout(height=max(500, 17 * len(sz)), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="", legend_title="", yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each paper gets one topic or −1. Labels are the top 3 c-TF-IDF terms. "
               f"Topic −1 holds {meta['outlier_share']:.0%} of papers: HDBSCAN refused to place them.")

    st.markdown(f"#### Topic {label[sel_topic]}")
    if sel_topic == -1:
        st.info("Topic −1 has no words: it is the papers that did not fit any dense cluster.")
    else:
        t = topics[sel_topic]
        c1, c2 = st.columns([3, 2])
        w = pd.DataFrame(t["words"]).iloc[::-1]
        fig = px.bar(w, x="score", y="term", orientation="h", color_discrete_sequence=[ORANGE])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="",
                          xaxis_title="c-TF-IDF score", title=f"Top words · {t['size']:,} papers")
        c1.plotly_chart(fig, width="stretch")

        # share of each year's papers that fall in this topic
        per_year = df.groupby("year").size()
        share = (pd.Series(t["by_year"]).reindex(per_year.index, fill_value=0) / per_year * 100)
        fig = px.bar(x=share.index.astype(str), y=share.values, color_discrete_sequence=[BLUE],
                     labels={"x": "", "y": "% of that year's papers"})
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Share by year")
        c2.plotly_chart(fig, width="stretch")
        c2.caption("Share, not count: the list grows a lot each year.")
        c2.markdown("**Example titles**\n" + "\n".join(f"- {x}" for x in t["examples"]))

    st.markdown("#### Rising and falling topics, 2025 → 2026")
    yr = df[df.topic >= 0].groupby(["topic", "year"]).size().unstack(fill_value=0)
    tot = df.groupby("year").size()
    chg = (yr[2026] / tot[2026] - yr[2025] / tot[2025]) * 100
    chg = pd.concat([chg.nlargest(8), chg.nsmallest(8)]).sort_values()
    fig = go.Figure(go.Bar(x=chg.values, y=[label[i] for i in chg.index], orientation="h",
                           marker_color=[ORANGE if v > 0 else BLUE for v in chg.values]))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="change in share of papers (percentage points)")
    st.plotly_chart(fig, width="stretch")

    note("Clear, nameable research areas: reinforcement learning, security, RAG, benchmarks, GUI and web agents, "
         "memory, MCP, medical and scientific agents. From 2025 to 2026, agent skills, RL and coding agents "
         "grow while RAG drops the most.",
         "The number of topics is a property of the pipeline, not the field: change min_cluster_size and you "
         "get a different count, and two topics can describe the same area (look for two RL topics). "
         f"{meta['outlier_share']:.0%} of papers are in topic −1 and have no words here. "
         "Embeddings are TF-IDF + SVD, so topics group shared vocabulary, not deeper meaning.")

# ====================================================== 3. EMBEDDING VIEW
with tab3:
    st.subheader("Embedding map: papers near each other share vocabulary")
    c1, c2 = st.columns([2, 1])
    proj = c1.radio("Projection", ["UMAP (n_neighbors=15)", "UMAP (n_neighbors=50, other seed)", "PCA"],
                    horizontal=True)
    colour = c2.radio("Colour", ["Selected topic", "8 largest topics"], horizontal=True)
    xcol, ycol = {"UMAP (n_neighbors=15)": ("umap_x", "umap_y"),
                  "UMAP (n_neighbors=50, other seed)": ("umap50_x", "umap50_y"),
                  "PCA": ("pca_x", "pca_y")}[proj]

    p = d.copy()
    if colour == "Selected topic":
        p["group"] = np.where(p.topic == sel_topic, label[sel_topic],
                              np.where(p.topic == -1, "outliers", "other topics"))
        cmap = {label[sel_topic]: ORANGE, "other topics": GREY, "outliers": LIGHT}
        order_g = ["outliers", "other topics", label[sel_topic]]
    else:
        big = order[:8]
        p["group"] = [label[t] if t in big else ("outliers" if t == -1 else "other topics") for t in p.topic]
        cmap = {label[t]: c for t, c in zip(big, OKABE)} | {"other topics": GREY, "outliers": LIGHT}
        order_g = ["outliers", "other topics"] + [label[t] for t in big]
    p["group"] = pd.Categorical(p["group"], order_g)
    p = p.sort_values("group")   # coloured points drawn last, on top
    fig = px.scatter(p, x=xcol, y=ycol, color="group", color_discrete_map=cmap,
                     hover_name="title", hover_data={"year": True, xcol: False, ycol: False, "group": False},
                     render_mode="webgl", opacity=0.6)
    fig.update_traces(marker_size=4)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), legend_title="",
                      legend=dict(itemsizing="constant"),
                      xaxis=dict(showticklabels=False, title=""), yaxis=dict(showticklabels=False, title=""))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Embedding: TF-IDF (1-2 grams) → TruncatedSVD 100-d, cosine. Projection: {proj}, "
               "min_dist=0.1, seed fixed. Axes have no units; only local neighbourhoods are meaningful.")

    note("Topics form compact islands (GUI agents, medical, autonomous driving, MCP), and related themes sit "
         "next to each other: the memory topics (long-term memory, long-horizon tasks, self-evolving agents) "
         "cluster together, and coding agents sit beside software engineering.",
         "Switch the projection: island positions, gaps and sizes change while the data do not. Distances "
         "between islands are not distances in meaning, and PCA shows almost no structure because 100 SVD "
         "dimensions keep only ~10% of TF-IDF variance. Colour comes from the clusterer, so the islands partly "
         "confirm themselves.")

# ============================================================== 4. NOTES
with tab4:
    st.subheader("Summary for the homework")
    st.markdown(f"""
| View | Transform | What it reveals | One way it misleads |
|---|---|---|---|
| ① Sorted bars + keyness + KWIC | word counts, log-ratio | the core vocabulary and what changed from 2025 to 2026 | template words and unstemmed variants inflate or split counts |
| ② Topic bars, c-TF-IDF words, trend | TF-IDF → SVD → UMAP → HDBSCAN → c-TF-IDF | {meta['n_topics']} named research areas and which are rising | the topic count depends on parameters; {meta['outlier_share']:.0%} of papers are hidden in topic −1 |
| ③ Embedding scatter | same embedding, UMAP / PCA to 2-D | neighbourhoods of related work | positions, gaps and sizes change with projection and seed |

**Pipeline and parameters**
- TF-IDF: {meta['params']['tfidf']}; SVD: {meta['params']['svd']} dimensions ({meta['svd_variance']:.0%} variance)
- UMAP for clustering: {meta['params']['umap_cluster']}
- HDBSCAN: {meta['params']['hdbscan']}
- c-TF-IDF: tf(t, c) × log(1 + A / f(t)), computed on all papers of a topic joined together

**Why not sentence-transformer embeddings?** LSA (TF-IDF + SVD) keeps the app light and reproducible, and every
step can be shown. Its topics group shared words; a neural embedding would also group paraphrases.

**Data caveat.** The text is a curator's 1-3 bullet summary of each paper, not the abstract, and the list
grew mostly in 2025-2026 (2023: {(df.year == 2023).sum()} papers, 2026: {(df.year == 2026).sum():,}).
""")
