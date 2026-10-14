"""
Homework 7 - data preparation (run once, offline).

Corpus: short summaries of AI-agent research papers on arXiv, collected in
the open "Autonomous-Agents" list by Teemu Maatta (MIT licence):
    https://github.com/tmgthb/Autonomous-Agents

Pipeline (BERTopic-style, all steps visible):
    text -> TF-IDF -> TruncatedSVD (LSA embedding, 100-d) -> UMAP 5-d
         -> HDBSCAN clusters (topic -1 = outliers) -> c-TF-IDF topic words
    plus UMAP 2-d and PCA 2-d for the projection view.

Usage:
    git clone --depth 1 https://github.com/tmgthb/Autonomous-Agents.git
    python prepare_data.py Autonomous-Agents
Writes data/papers.parquet and data/topics.json for app.py.
"""
import glob, html, json, os, re, sys

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             CountVectorizer, TfidfVectorizer)
from sklearn.preprocessing import normalize

SEED = 42
SRC = sys.argv[1] if len(sys.argv) > 1 else "Autonomous-Agents"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

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
yymm = df["arxiv_id"].str[:4]
df["year"] = 2000 + yymm.str[:2].astype(int)
df["month"] = pd.to_datetime("20" + yymm.str[:2] + "-" + yymm.str[2:] + "-01")
df = df[(df["year"] >= 2023) & (df["year"] <= 2026)].reset_index(drop=True)
df["text"] = df["title"] + ". " + df["summary"]
print(f"documents: {len(df):,}  ({df['year'].value_counts().sort_index().to_dict()})")

# ---------------------------------------------------- 2. LSA embeddings
tfidf = TfidfVectorizer(stop_words=STOP, min_df=5, max_df=0.5,
                        ngram_range=(1, 2), sublinear_tf=True)
X = tfidf.fit_transform(df["text"])
svd = TruncatedSVD(n_components=100, random_state=SEED)
emb = normalize(svd.fit_transform(X))
print(f"TF-IDF vocab {X.shape[1]:,}; SVD explains {svd.explained_variance_ratio_.sum():.1%}")

# ------------------------------------------- 3. UMAP -> HDBSCAN (topics)
u5 = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
               random_state=SEED).fit_transform(emb)
labels = HDBSCAN(min_cluster_size=80, min_samples=10).fit_predict(u5)

# Renumber topics by size: 0 = largest, -1 stays outliers
sizes = pd.Series(labels[labels >= 0]).value_counts()
remap = {old: new for new, old in enumerate(sizes.index)}
remap[-1] = -1
df["topic"] = [remap[l] for l in labels]
print(f"topics: {len(sizes)}, outliers (topic -1): {(df.topic == -1).mean():.1%}")

# --------------------------------------------------------- 4. c-TF-IDF
cv = CountVectorizer(stop_words=STOP, min_df=5, ngram_range=(1, 2))
docs_per_topic = df[df.topic >= 0].groupby("topic")["text"].apply(" ".join)
C = cv.fit_transform(docs_per_topic).toarray().astype(float)
terms = np.array(cv.get_feature_names_out())
tf = C / C.sum(axis=1, keepdims=True)
A = C.sum() / C.shape[0]                    # average words per class
idf = np.log(1 + A / C.sum(axis=0))         # f(t) = frequency across classes
ctfidf = tf * idf

def short_label(ranked, n=3):
    """Top terms without repeats: 'reinforcement learning' replaces 'learning'."""
    chosen = []
    for term in ranked:
        if any(term in c for c in chosen):
            continue
        chosen = [c for c in chosen if c not in term] + [term]
        if len(chosen) == n:
            break
    return chosen

topics = []
for k, t in enumerate(docs_per_topic.index):
    order = np.argsort(-ctfidf[k])[:12]
    words = [{"term": terms[i], "score": float(ctfidf[k, i])} for i in order]
    sub = df[df.topic == t]
    topics.append({
        "topic": int(t),
        "label": f"{t}: " + " · ".join(short_label([w["term"] for w in words])),
        "size": int(len(sub)),
        "words": words,
        "by_year": {int(y): int(n) for y, n in sub["year"].value_counts().items()},
        "examples": sub.sample(min(6, len(sub)), random_state=SEED)["title"].tolist(),
    })

# --------------------------------------------- 5. 2-D projections for display
u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
               random_state=SEED).fit_transform(emb)
u2b = umap.UMAP(n_neighbors=50, min_dist=0.1, metric="cosine",
                random_state=7).fit_transform(emb)
p2 = PCA(n_components=2, random_state=SEED).fit_transform(emb)
df[["umap_x", "umap_y"]] = u2
df[["umap50_x", "umap50_y"]] = u2b
df[["pca_x", "pca_y"]] = p2

cols = ["arxiv_id", "title", "summary", "year", "month", "topic",
        "umap_x", "umap_y", "umap50_x", "umap50_y", "pca_x", "pca_y"]
df[cols].to_parquet(os.path.join(OUT, "papers.parquet"), index=False)
meta = {
    "n_docs": len(df), "n_topics": len(topics),
    "outlier_share": float((df.topic == -1).mean()),
    "svd_variance": float(svd.explained_variance_ratio_.sum()),
    "params": {"tfidf": "min_df=5, max_df=0.5, 1-2 grams, sublinear tf",
               "svd": 100, "umap_cluster": "n_neighbors=15, 5-d, cosine, seed 42",
               "hdbscan": "min_cluster_size=80, min_samples=10"},
    "topics": topics,
}
json.dump(meta, open(os.path.join(OUT, "topics.json"), "w"), indent=1)
print("saved to", OUT)
