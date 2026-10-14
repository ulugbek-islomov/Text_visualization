# Homework 7 – What is AI-agent research about?

Streamlit app exploring 14,955 summaries of AI-agent papers on arXiv (2023–2026).
Source: [Autonomous-Agents](https://github.com/tmgthb/Autonomous-Agents) list by Teemu Maatta (MIT licence).

| Tab | Assignment part | Views |
|---|---|---|
| ① Words & counts | honest frequency view | sorted bars next to a word cloud (same counts), **raw counts vs TF-IDF keyness** for the selected topic, year keyness 2026 vs 2025, concordance (KWIC) |
| ② Topics (BERTopic) | topic-model view + topic selector | c-TF-IDF bars, share by year, topic sizes incl. −1, topic map, **BERTopic hierarchy**, rising/falling topics |
| ③ Embedding map | embedding projection (optional) | UMAP (two settings) and PCA, show/hide unassigned |
| ④ Notes | what each view reveals / how it misleads | summary table, parameters |

BERTopic runs offline in `prepare_data.py`; the app only reads `papers.parquet` and
`ai_agents_bertopic.json`, so it needs no bertopic, sentence-transformers or torch.

## Deploy (free public URL)
1. Create a public GitHub repo. **Add file → Upload files**, drag in all files together (no folders):
   `app.py`, `requirements.txt`, `papers.parquet`, `ai_agents_bertopic.json`, `README.md`, `prepare_data.py`.
2. https://share.streamlit.io → **Create app** → repo, branch `main`, file `app.py` → **Deploy**.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the BERTopic results (optional; laptop or Google Colab)
```
pip install bertopic umap-learn hdbscan pyarrow
git clone --depth 1 https://github.com/tmgthb/Autonomous-Agents.git
python prepare_data.py Autonomous-Agents            # TF-IDF + SVD embeddings (CPU, ~1 min)
python prepare_data.py Autonomous-Agents --minilm   # all-MiniLM-L6-v2 embeddings (use a Colab GPU)
```
