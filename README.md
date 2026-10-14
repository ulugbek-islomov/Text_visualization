# Homework 7 – What is AI-agent research about?

Streamlit app exploring ~15,000 summaries of AI-agent papers on arXiv (2023–2026).
Source: [Autonomous-Agents](https://github.com/tmgthb/Autonomous-Agents) list by Teemu Maatta (MIT licence).

| Tab | Rubric part | Views |
|---|---|---|
| ① Frequency | honest frequency view (3) | sorted bars, keyness 2026 vs 2025, concordance (KWIC) |
| ② Topics | topic model viz (4) | topic sizes incl. topic −1, c-TF-IDF words, share by year, rising/falling topics |
| ③ Embedding map | projection with caveats (3) | UMAP (two settings) and PCA, coloured by topic |
| ④ Notes | reveals / misleads | summary table, pipeline parameters |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the data (optional, ~1 min)
```
pip install umap-learn
git clone --depth 1 https://github.com/tmgthb/Autonomous-Agents.git
python prepare_data.py Autonomous-Agents
```

## Deploy (free public URL)
1. Create a public GitHub repo and upload `app.py`, `requirements.txt`, `README.md`, `prepare_data.py` and the `data/` folder.
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py` → **Deploy**.
3. Copy the `https://<name>.streamlit.app` link for the submission.
