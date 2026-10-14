"""
Homework 7 - run BERTopic once, offline, and export small files for the app.

The deployed Streamlit app must not install bertopic / sentence-transformers /
torch (too heavy for Streamlit Community Cloud), so this script does the
expensive half and the app only reads the results:

    papers.parquet          one row per paper: text, year, topic, 2-D coordinates
    ai_agents_bertopic.json topic words, topic coordinates, hierarchy figure

Corpus: summaries of AI-agent research papers on arXiv, from the open
"Autonomous-Agents" list by Teemu Maatta (MIT licence):
    https://github.com/tmgthb/Autonomous-Agents

Usage (laptop or Google Colab):
    pip install bertopic umap-learn hdbscan pyarrow
    git clone --depth 1 https://github.com/tmgthb/Autonomous-Agents.git
    python prepare_data.py Autonomous-Agents            # TF-IDF + SVD embeddings
    python prepare_data.py Autonomous-Agents --minilm   # sentence-transformer embeddings (GPU helps)
"""
import glob, html, json, os, re, sys

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             CountVectorizer, TfidfVectorizer)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from umap import UMAP

SEED = 42
SRC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "Autonomous-Agents"
USE_MINILM = "--minilm" in sys.argv
OUT = os.path.dirname(os.path.abspath(__file__))

# Words the list's summary template repeats in almost every entry
# ("X: introduces a framework that utilizes ...").  They describe the
# writing style, not the research, so the topic words leave them out.
TEMPLATE_WORDS = {
    "introduces", "introduce", "framework", "utilizes", "utilize", "uses", "use",
    "using", "demonstrates", "demonstrate", "approach", "method", "methods",
    "proposes", "propose", "paper", "novel", "results", "based", "enables",
    "achieves", "performance", "new", "significantly", "existing", "et", "al",
}
STOP = list(ENGLISH_STOP_WORDS | TEMPLATE_WORDS)

UMAP_KW = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
HDBSCAN_KW = dict(min_cluster_size=100, min_samples=10, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
VECTORIZER_KW = dict(stop_words=STOP, min_df=5, ngram_range=(1, 2),
                     token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b")

# ---------------------------------------------------------------- 1. parse
link = re.compile(r"^\[(?P<title>.+?)\]\(https?://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})(v\d+)?\)")
rows = []
files = [os.path.join(SRC, "README.md")] + sorted(glob.glob(os.path.join(SRC, "resources", "*.md")))
for f in files:
    cur = None
    for line in open(f, encoding="utf-8"):
        line = html.unescape(line.rstrip())
        m = link.match(line)
        if m:
            if cur: rows.append(cur)
            cur = {"arxiv_id": m["id"], "title": m["title"].strip(), "bullets": []}
        elif cur and line.startswith("- "):
            b = re.sub(r"^- [^:]{1,60}:\s+", "- ", line)      # drop "SystemName:" prefix
            cur["bullets"].append(b[2:].strip())
        elif cur and (line.startswith("#") or line.startswith("---")) and cur["bullets"]:
            rows.append(cur); cur = None
    if cur: rows.append(cur)

df = pd.DataFrame(rows)
df["summary"] = df["bullets"].apply(" ".join)
df = df.drop(columns="bullets")
df = df[df["summary"].str.split().str.len() >= 15]
df = df.drop_duplicates("arxiv_id").reset_index(drop=True)
df["year"] = 2000 + df["arxiv_id"].str[:2].astype(int)
df = df[(df["year"] >= 2023) & (df["year"] <= 2026)].reset_index(drop=True)
docs = (df["title"] + ". " + df["summary"]).tolist()
print(f"documents: {len(df):,}  ({df['year'].value_counts().sort_index().to_dict()})")

# ------------------------------------------------------------ 2. embeddings
if USE_MINILM:
    from sentence_transformers import SentenceTransformer
    EMB_NAME = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(EMB_NAME)
    emb = embedder.encode(docs, show_progress_bar=True, batch_size=64)
else:
    # BERTopic's documented scikit-learn embedding backend (CPU only, no torch)
    EMB_NAME = "TF-IDF (1-2 grams) + TruncatedSVD 100-d (LSA)"
    embedder = make_pipeline(
        TfidfVectorizer(stop_words=STOP, min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True),
        TruncatedSVD(n_components=100, random_state=SEED),
        Normalizer(),
    )
    emb = embedder.fit_transform(docs)

# ---------------------------------------------------------------- 3. BERTopic
model = BERTopic(
    embedding_model=embedder,
    umap_model=UMAP(**UMAP_KW),
    hdbscan_model=HDBSCAN(**HDBSCAN_KW),
    vectorizer_model=CountVectorizer(**VECTORIZER_KW),
    calculate_probabilities=False,
    verbose=True,
)
topics, _ = model.fit_transform(docs, embeddings=emb)
topics = np.asarray(topics)
df["topic"] = topics
n_topics = len(set(topics)) - (1 if -1 in topics else 0)
print(f"topics: {n_topics}, unassigned: {(topics == -1).mean():.1%}")


def short_label(words, n=3):
    """Top terms without repeats: 'reinforcement learning' replaces 'learning'."""
    chosen = []
    for term in words:
        if any(term in c for c in chosen):
            continue
        chosen = [c for c in chosen if c not in term] + [term]
        if len(chosen) == n:
            break
    return ", ".join(chosen)


topic_rows = []
for tid in sorted(set(topics.tolist())):
    words = model.get_topic(tid) or []
    sub = df[df.topic == tid]
    topic_rows.append({
        "topic": int(tid),
        "count": int(len(sub)),
        "label": short_label([w for w, _ in words]) if tid != -1 else "unassigned",
        "words": [[w, round(float(s), 5)] for w, s in words[:10]],
        "by_year": {int(y): int(n) for y, n in sub["year"].value_counts().items()},
        "examples": sub.sample(min(6, len(sub)), random_state=SEED)["title"].tolist(),
    })
labels = {r["topic"]: f"{r['topic']}: {r['label']}" for r in topic_rows}

# -------------------------------------------- 4. hierarchy (BERTopic's own figure)
hier = model.hierarchical_topics(docs)
model.set_topic_labels({t: l for t, l in labels.items() if t != -1})
fig_h = model.visualize_hierarchy(hierarchical_topics=hier, custom_labels=True,
                                  width=1000, height=max(600, 22 * n_topics),
                                  title="Hierarchical clustering of topics")

# ------------------------------------------------ 5. 2-D projections of documents
xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine",
          random_state=SEED).fit_transform(emb)
xy50 = UMAP(n_neighbors=50, n_components=2, min_dist=0.1, metric="cosine",
            random_state=7).fit_transform(emb)
pca = PCA(n_components=2, random_state=SEED).fit_transform(emb)
df[["x", "y"]] = np.round(xy, 3)
df[["x50", "y50"]] = np.round(xy50, 3)
df[["pca_x", "pca_y"]] = np.round(pca, 4)

topic_coords = []
for tid in sorted(set(topics.tolist()) - {-1}):
    sel = topics == tid
    topic_coords.append({"topic": int(tid), "x": round(float(xy[sel, 0].mean()), 3),
                         "y": round(float(xy[sel, 1].mean()), 3), "count": int(sel.sum()),
                         "label": labels[tid]})

# ---------------------------------------------------------------- 6. save
df[["arxiv_id", "title", "summary", "year", "topic", "x", "y", "x50", "y50", "pca_x", "pca_y"]] \
    .to_parquet(os.path.join(OUT, "papers.parquet"), index=False)
payload = {
    "corpus": {"name": "AI-agent research papers on arXiv (Autonomous-Agents list)",
               "n_docs": len(df), "source": "github.com/tmgthb/Autonomous-Agents (MIT)",
               "by_year": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()}},
    "params": {"embedding": EMB_NAME, "umap": UMAP_KW,
               "hdbscan": {k: v for k, v in HDBSCAN_KW.items() if k != "prediction_data"},
               "vectorizer": "CountVectorizer, English + template stop words, min_df=5, 1-2 grams",
               "n_topics": int(n_topics), "n_unassigned": int((topics == -1).sum())},
    "template_words": sorted(TEMPLATE_WORDS),
    "topics": topic_rows,
    "topic_coords": topic_coords,
    "hierarchy_fig": json.loads(fig_h.to_json()),
}
with open(os.path.join(OUT, "ai_agents_bertopic.json"), "w") as f:
    json.dump(payload, f, separators=(",", ":"))
print("saved papers.parquet and ai_agents_bertopic.json to", OUT)
