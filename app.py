"""
DATS 6401 - Homework 7: Text Data
What is AI-agent research about?  ~15,000 arXiv paper summaries (2023-2026).

Run:   streamlit run app.py
Data:  papers.parquet + ai_agents_bertopic.json, both made by prepare_data.py
       (BERTopic runs offline; this app does not install bertopic or torch).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

st.set_page_config(page_title="AI-Agent Research Text Explorer", page_icon="🤖", layout="wide")

INK, ORANGE, GREY, LIGHT = "#2E6E8E", "#D55E00", "#9AA5B1", "#D9DEE3"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#8C6D31", "#000000", "#7F3C8D", "#11A579"]

# Data files may sit next to app.py or inside a data/ folder
HERE = Path(__file__).parent
DATA = HERE / "data" if (HERE / "data" / "papers.parquet").exists() else HERE


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    files = [DATA / "papers.parquet", DATA / "ai_agents_bertopic.json"]
    if not all(f.exists() for f in files):
        return None, None
    df = pd.read_parquet(files[0])
    df["text"] = df["title"] + ". " + df["summary"]
    return df, json.loads(files[1].read_text())


def stop_list(drop_template):
    return list(ENGLISH_STOP_WORDS | set(ART["template_words"])) if drop_template else "english"


@st.cache_data
def term_counts(texts: tuple, ngram: int, drop_template: bool):
    cv = CountVectorizer(stop_words=stop_list(drop_template), ngram_range=(ngram, ngram), min_df=3)
    X = cv.fit_transform(texts)
    return pd.Series(np.asarray(X.sum(axis=0)).ravel(), index=cv.get_feature_names_out())


@st.cache_resource
def paper_tfidf(ngram: int, drop_template: bool):
    """TF-IDF of every paper (sklearn smoothed idf, L2 rows)."""
    tv = TfidfVectorizer(stop_words=stop_list(drop_template), ngram_range=(ngram, ngram), min_df=5)
    return tv.fit_transform(DF["text"]), tv.get_feature_names_out()


def tfidf_keyness(topic: int, ngram: int, drop_template: bool, n=12):
    """Mean TF-IDF inside the topic minus mean TF-IDF in all other papers."""
    M, terms = paper_tfidf(ngram, drop_template)
    mask = (DF["topic"] == topic).to_numpy()
    diff = np.asarray(M[mask].mean(axis=0)).ravel() - np.asarray(M[~mask].mean(axis=0)).ravel()
    return pd.Series(diff, index=terms).nlargest(n)


def hbar(series, colour, xlabel, height=None, highlight=None):
    s = series.sort_values()
    colours = [GREY if highlight is not None and t in highlight else colour for t in s.index]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colours))
    fig.update_layout(height=height or 26 * len(s) + 70, margin=dict(l=0, r=10, t=10, b=10),
                      xaxis_title=xlabel)
    return fig


def kwic(texts, word, width=60, n=12):
    """Keyword in context: the sentences behind a bar."""
    pat = re.compile(rf"\b{re.escape(word)}\b", re.I)
    lines = []
    for t in texts:
        m = pat.search(t)
        if m:
            left = t[max(0, m.start() - width):m.start()].rjust(width)
            lines.append(f"{left} [{m.group(0).upper()}] {t[m.end():m.end() + width]}")
        if len(lines) >= n:
            break
    return lines


def note(reveals, misleads):
    c1, c2 = st.columns(2)
    c1.success(f"**What it reveals.** {reveals}")
    c2.warning(f"**How it can mislead.** {misleads}")


DF, ART = load()
if DF is None:
    st.error("Data files not found. Upload **papers.parquet** and **ai_agents_bertopic.json** "
             "to the repo, in the same place as app.py.")
    st.stop()

P = ART["params"]
TOPICS = {t["topic"]: t for t in ART["topics"]}
LABEL = {t: (f"{t}: {v['label']}" if t != -1 else "−1: unassigned") for t, v in TOPICS.items()}
BY_SIZE = sorted([t for t in TOPICS if t != -1], key=lambda t: -TOPICS[t]["count"])
share_unassigned = P["n_unassigned"] / ART["corpus"]["n_docs"]

# ------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
years = st.sidebar.multiselect("Years", [2023, 2024, 2025, 2026], default=[2023, 2024, 2025, 2026],
                               help="arXiv submission year, read from the paper ID. Empty = all years.")
sel = st.sidebar.selectbox("Topic", BY_SIZE + [-1], format_func=lambda t: f"{LABEL[t]} ({TOPICS[t]['count']:,})",
                           help="Used in every tab")
st.sidebar.caption(
    "Corpus: summaries of AI-agent papers on arXiv from the open "
    "[Autonomous-Agents](https://github.com/tmgthb/Autonomous-Agents) list "
    "(Teemu Maatta, MIT licence). Each document = title + 1-3 bullet summary."
)
D = DF[DF["year"].isin(years)] if years else DF

# ---------------------------------------------------------------- header
st.title("🤖 What is AI-agent research about?")
st.caption("Text has no x and no y. Each tab is a picture of a different transform of the same "
           "words: counts, TF-IDF, BERTopic topics, and an embedding projection.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers (filtered)", f"{len(D):,}")
m2.metric("BERTopic topics", P["n_topics"])
m3.metric("Unassigned (topic −1)", f"{share_unassigned:.0%}")
m4.metric("Median words / paper", int(D["text"].str.split().str.len().median()))

tab1, tab2, tab3, tab4 = st.tabs(["① Words & counts", "② Topics (BERTopic)", "③ Embedding map", "④ Notes"])

# ===================================================== 1. FREQUENCY VIEWS
with tab1:
    c1, c2, c3 = st.columns(3)
    ngram = c1.radio("Unit", [1, 2], horizontal=True,
                     format_func=lambda n: "single words" if n == 1 else "two-word phrases")
    drop_tpl = c2.toggle("Also drop template words", value=True,
                         help="'introduces', 'framework', 'utilizes'... come from how the list is written")
    top_n = c3.slider("How many terms", 10, 40, 20, 5)

    counts = term_counts(tuple(D["text"]), ngram, drop_tpl)
    top = counts.nlargest(top_n)

    st.subheader("The same counts, drawn two ways")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(hbar(top, INK, "occurrences", height=470), width="stretch")
        st.caption("**Sorted bars.** Frequency on position and length, the most accurate channels. "
                   "You can read whether the 3rd term beats the 5th.")
    with right:
        wc = WordCloud(width=900, height=520, background_color="white", prefer_horizontal=0.9,
                       random_state=42, color_func=lambda *a, **k: INK).generate_from_frequencies(top.to_dict())
        st.image(wc.to_array(), width="stretch")
        st.caption("**Word cloud.** Frequency on area (ranked near the bottom), position chosen by the "
                   "packing algorithm, and long words look bigger at equal counts. One colour, because "
                   "colour would encode nothing. Good as a cover image, not for comparison.")
    st.caption(f"{len(D):,} papers · sklearn English stop words"
               + (" + template words" if drop_tpl else "") + " · no stemming · terms in ≥ 3 papers.")

    st.divider()
    st.subheader("Raw counts vs TF-IDF keyness for one topic")
    if sel == -1:
        st.info("Pick a real topic in the sidebar: topic −1 has no shared theme to describe.")
    else:
        in_topic = DF[DF.topic == sel]["text"]
        raw = term_counts(tuple(in_topic), ngram, drop_tpl).nlargest(12)
        key = tfidf_keyness(sel, ngram, drop_tpl)
        shared = set(raw.index) & set(key.index)
        a, b = st.columns(2)
        a.markdown(f"**Raw counts inside topic {LABEL[sel]}**")
        a.plotly_chart(hbar(raw, INK, "occurrences", highlight=shared), width="stretch")
        b.markdown("**TF-IDF keyness: frequent here, rare in other topics**")
        b.plotly_chart(hbar(key, ORANGE, "mean TF-IDF: topic minus other papers", highlight=shared),
                       width="stretch")
        dropped = [t for t in raw.index if t not in shared]
        st.caption("Grey bars appear in both lists. "
                   + (f"Counts-only terms such as **{', '.join(dropped[:4])}** are common across the whole "
                      "corpus, so TF-IDF pushes them down. " if dropped else "")
                   + "TF-IDF is computed per paper (sklearn smoothed idf, L2 rows); keyness = mean TF-IDF "
                     "in this topic minus mean TF-IDF in all other papers.")

    st.divider()
    st.subheader("Keyness between years: what is distinctive of 2026 vs 2025?")
    ca = term_counts(tuple(DF[DF.year == 2026]["text"]), ngram, True)
    cb = term_counts(tuple(DF[DF.year == 2025]["text"]), ngram, True)
    vocab = ca.index.union(cb.index)
    ca, cb = ca.reindex(vocab, fill_value=0), cb.reindex(vocab, fill_value=0)
    lr = np.log2(((ca + 0.5) / ca.sum()) / ((cb + 0.5) / cb.sum()))
    min_total = 300 if ngram == 1 else 60
    lr = lr[(ca + cb) >= min_total]
    ky = pd.concat([lr.nlargest(12), lr.nsmallest(12)]).sort_values()
    fig = go.Figure(go.Bar(x=ky.values, y=ky.index, orientation="h",
                           marker_color=[ORANGE if v > 0 else INK for v in ky.values]))
    fig.update_layout(height=600, margin=dict(l=0, r=10, t=10, b=10),
                      xaxis_title="log₂ ratio   (← more 2025 · more 2026 →)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"log-ratio = log₂[(c₂₀₂₆+0.5)/N₂₀₂₆ ÷ (c₂₀₂₅+0.5)/N₂₀₂₅], terms used ≥ {min_total} times. "
               "Fixed groups, so the year filter does not apply here.")

    st.subheader("Concordance: read the lines behind a bar")
    word = st.text_input("Word or phrase", value="harness")
    lines = kwic(D["text"].tolist(), word.strip()) if word.strip() else []
    st.code("\n".join(lines) if lines else "No matches.", language=None)

    note("A few words carry the corpus: agent, LLM, multi-agent, reasoning, memory. TF-IDF keyness turns "
         "a topic's generic counts into its specific vocabulary, and year keyness shows 2026 papers moving "
         "to harnesses, skills and frontier models.",
         "The word cloud makes long words look important and hides ranks. Turn off the template toggle and "
         "'framework', 'introduces', 'utilizes' climb the bars: they reflect the curator's wording, as do "
         "'aims' and 'stores' in the 2025 keyness. Nothing is stemmed, so agent / agents / agentic split one idea.")

# ======================================================== 2. TOPIC VIEWS
with tab2:
    st.markdown(
        f"**Pipeline (run offline):** {P['embedding']} → UMAP (n_neighbors={P['umap']['n_neighbors']}, "
        f"{P['umap']['n_components']}-d, cosine) → HDBSCAN (min_cluster_size={P['hdbscan']['min_cluster_size']}) "
        f"→ c-TF-IDF. **Result:** {P['n_topics']} topics and {P['n_unassigned']:,} papers "
        f"({share_unassigned:.0%}) left in topic −1."
    )

    st.subheader(f"1 · What topic {LABEL[sel]} is made of")
    t = TOPICS[sel]
    if sel == -1:
        st.info("Topic −1 is the papers HDBSCAN did not place in any dense cluster. "
                f"It is the largest group: {t['count']:,} papers.")
    c1, c2 = st.columns([3, 2])
    w = pd.Series({a: b for a, b in t["words"]})
    c1.plotly_chart(hbar(w, INK if sel != -1 else GREY, "c-TF-IDF score"), width="stretch")
    c1.caption("c-TF-IDF joins all papers of a topic into one document and scores terms that are "
               "distinctive to it. These are scores, not word counts.")
    per_year = DF.groupby("year").size()
    share = pd.Series(t["by_year"]).rename(int).reindex(per_year.index, fill_value=0) / per_year * 100
    fig = px.bar(x=share.index.astype(str), y=share.values, color_discrete_sequence=[ORANGE],
                 labels={"x": "", "y": "% of that year's papers"}, height=250)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    c2.plotly_chart(fig, width="stretch")
    c2.caption("Share, not count: the list grows a lot each year (2023 has only "
               f"{ART['corpus']['by_year']['2023']} papers).")
    c2.markdown("**Example titles**\n" + "\n".join(f"- {x}" for x in t["examples"]))

    st.subheader("2 · Topic sizes")
    sz = pd.DataFrame([{"label": LABEL[k], "count": v["count"], "topic": k} for k, v in TOPICS.items()])
    sz["kind"] = np.where(sz.topic == -1, "unassigned", np.where(sz.topic == sel, "selected", "topic"))
    fig = px.bar(sz.sort_values("count"), x="count", y="label", orientation="h", color="kind",
                 color_discrete_map={"topic": GREY, "selected": ORANGE, "unassigned": LIGHT},
                 height=18 * len(sz) + 80, labels={"label": "", "count": "papers", "kind": ""})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each paper gets exactly one topic or −1. The unassigned bar is kept on purpose.")

    st.subheader("3 · Where topics sit relative to each other")
    tc = pd.DataFrame(ART["topic_coords"])
    tc["kind"] = np.where(tc.topic == sel, "selected", "topic")
    fig = px.scatter(tc, x="x", y="y", size="count", hover_name="label", text="topic", size_max=55,
                     color="kind", color_discrete_map={"topic": INK, "selected": ORANGE}, height=520)
    fig.update_traces(textposition="middle center", textfont=dict(color="white", size=10))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(showticklabels=False, title=None), yaxis=dict(showticklabels=False, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption("Circle = topic at the mean 2-D UMAP position of its papers, sized by paper count. "
               "Near and far is a hint from a projection, not a measurement.")

    st.subheader("4 · Topic hierarchy (BERTopic `visualize_hierarchy`)")
    fig_h = pio.from_json(json.dumps(ART["hierarchy_fig"]))
    fig_h.update_layout(width=None, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_h, width="stretch")
    st.caption("Ward linkage on the topics' c-TF-IDF vectors. Topics that merge early (short branches) "
               "could be one topic; cutting the tree higher gives fewer, broader topics.")

    st.subheader("5 · Rising and falling topics, 2025 → 2026")
    yr = pd.DataFrame({k: pd.Series(v["by_year"]).rename(int) for k, v in TOPICS.items() if k != -1}).T.fillna(0)
    chg = (yr[2026] / per_year[2026] - yr[2025] / per_year[2025]) * 100
    chg = pd.concat([chg.nlargest(8), chg.nsmallest(8)]).sort_values()
    fig = go.Figure(go.Bar(x=chg.values, y=[LABEL[i] for i in chg.index], orientation="h",
                           marker_color=[ORANGE if v > 0 else INK for v in chg.values]))
    fig.update_layout(height=500, margin=dict(l=0, r=10, t=10, b=10),
                      xaxis_title="change in share of papers (percentage points)")
    st.plotly_chart(fig, width="stretch")

    note("Nameable research areas: reinforcement learning, security, game theory, RAG, benchmarks, memory, "
         "GUI agents, MCP, medical and scientific agents. From 2025 to 2026, game theory, agent skills, RL "
         "and coding agents grow, while RAG shrinks the most. The hierarchy shows which topics are close cousins.",
         f"The topic count is a property of the pipeline (min_cluster_size, seed, embedding), not of the "
         f"field. {share_unassigned:.0%} of papers sit in topic −1 and have no words here. The embedding is "
         "TF-IDF + SVD, so topics group shared vocabulary rather than meaning; a sentence-transformer would "
         "also group paraphrases.")

# ====================================================== 3. EMBEDDING VIEW
with tab3:
    st.subheader("Every paper, placed by its embedding")
    c1, c2, c3 = st.columns([2, 1.3, 1])
    proj = c1.radio("Projection", ["UMAP (n_neighbors=15)", "UMAP (n_neighbors=50, other seed)", "PCA"],
                    horizontal=True)
    colour = c2.radio("Colour", ["Selected topic", "10 largest topics"], horizontal=True)
    show_noise = c3.checkbox("Show unassigned (−1)", value=True)
    xcol, ycol = {"UMAP (n_neighbors=15)": ("x", "y"),
                  "UMAP (n_neighbors=50, other seed)": ("x50", "y50"), "PCA": ("pca_x", "pca_y")}[proj]

    p = D if show_noise else D[D.topic != -1]
    p = p.copy()
    if colour == "Selected topic":
        p["group"] = np.where(p.topic == sel, LABEL[sel], np.where(p.topic == -1, "unassigned", "other topics"))
        cmap = {LABEL[sel]: ORANGE, "other topics": GREY, "unassigned": LIGHT}
        order = ["unassigned", "other topics", LABEL[sel]]
    else:
        big = BY_SIZE[:10]
        p["group"] = [LABEL[t] if t in big else ("unassigned" if t == -1 else "other topics") for t in p.topic]
        cmap = {LABEL[t]: c for t, c in zip(big, PALETTE)} | {"other topics": GREY, "unassigned": LIGHT}
        order = ["unassigned", "other topics"] + [LABEL[t] for t in big]
    p["group"] = pd.Categorical(p["group"], [o for o in dict.fromkeys(order)])
    p = p.sort_values("group")          # coloured points drawn last, on top
    fig = px.scatter(p, x=xcol, y=ycol, color="group", color_discrete_map=cmap, hover_name="title",
                     hover_data={"year": True, xcol: False, ycol: False, "group": False},
                     render_mode="webgl", opacity=0.6, height=640)
    fig.update_traces(marker_size=4)
    fig.update_layout(legend=dict(title="", itemsizing="constant"), margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(showticklabels=False, title=None), yaxis=dict(showticklabels=False, title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(p):,} papers · embedding: {P['embedding']}, cosine · {proj}, fixed seed. "
               "Axes have no units; only local neighbourhoods are meaningful.")

    note("Topics form islands (GUI agents, medical, autonomous driving, MCP) and related themes sit next to "
         "each other: coding agents beside software engineering, long-term memory beside self-evolving agents "
         "and KV-cache work, autonomous driving beside world models.",
         "Switch the projection: island positions, gaps and sizes change while the data do not. PCA shows "
         "almost no structure because the SVD keeps only ~10% of TF-IDF variance. Untick the unassigned "
         "papers and the map looks far tidier than the data are. The colours come from the same clusterer, "
         "so the islands partly confirm themselves.")

# ============================================================== 4. NOTES
with tab4:
    st.subheader("What each view reveals, and one way it can mislead")
    st.markdown(f"""
| View | Transform | What it reveals | One way it misleads |
|---|---|---|---|
| Sorted bars vs word cloud | word counts | the core vocabulary of the field | the cloud encodes frequency as area and word length; template words inflate counts |
| Raw counts vs TF-IDF keyness | counts vs TF-IDF (topic minus rest) | the specific vocabulary of a topic | TF-IDF depends on what the topic is compared against |
| Year keyness + concordance | log-ratio, KWIC | what changed from 2025 to 2026, and the sentences behind a word | style changes of the curator look like research trends |
| c-TF-IDF bars, sizes, hierarchy, trends | BERTopic | {P['n_topics']} named research areas, how they nest, which are rising | topic count depends on parameters; {share_unassigned:.0%} hidden in topic −1 |
| Embedding scatter | UMAP / PCA of embeddings | neighbourhoods of related work | positions, gaps and sizes change with projection and seed |

**Pipeline and parameters**
- Embedding: {P['embedding']}
- UMAP: n_neighbors={P['umap']['n_neighbors']}, n_components={P['umap']['n_components']}, min_dist={P['umap']['min_dist']}, cosine, random_state={P['umap']['random_state']}
- HDBSCAN: min_cluster_size={P['hdbscan']['min_cluster_size']}, min_samples={P['hdbscan']['min_samples']}
- c-TF-IDF vectorizer: {P['vectorizer']}
- BERTopic was run offline with `prepare_data.py`; the app only reads the saved results.

**Data caveat.** Each document is a curator's 1-3 bullet summary, not the abstract, and the list grows
fast: {", ".join(f"{k}: {v:,}" for k, v in ART['corpus']['by_year'].items())} papers.
""")
