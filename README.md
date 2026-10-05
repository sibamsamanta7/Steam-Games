# Steam Games: Market Intelligence and Recommendation System

Two end-to-end data science projects built on a snapshot of **130,100 Steam titles** (prices, tags, genres, descriptions, reviews, playtime and owner estimates):

1. **Market intelligence**: statistical analysis and machine learning to understand what drives traction on Steam.
2. **Recommendation system**: a content-based and hybrid recommender using NLP (TF-IDF, LSA), with free-text search and explainable results.

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu/data)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)

## Contents

- [Projects at a glance](#projects-at-a-glance)
- [Project 1: Market intelligence](#project-1-market-intelligence)
- [Project 2: Recommendation system](#project-2-recommendation-system)
- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Limitations](#limitations)
- [License](#license)

## Projects at a glance

| | Market intelligence | Recommendation system |
|---|---|---|
| **Notebook** | `steam-game-market-intelligence.ipynb` | `steam-game-recommendation-system.ipynb` |
| **Question** | What separates games that get traction from those that do not? | What should I play next if I liked game X? |
| **Methods** | Non-parametric tests, effect sizes, multiple-testing correction, gradient boosting, k-means | TF-IDF, truncated SVD (LSA), cosine similarity, Bayesian-average re-ranking |
| **Outputs** | Market findings, genre opportunity map, traction model, market segments | Similar-game lists, taste-profile recommendations, free-text search, explanations |
| **Runtime** | About 1 minute on one CPU core | About 1.5 minutes on one CPU core |

## Project 1: Market intelligence

**Notebook:** [`steam-game-market-intelligence.ipynb`](steam-game-market-intelligence.ipynb)

Turns raw store metadata into decisions for studios, publishers and analysts: where the market is crowded, what correlates with traction, and which segments are under-served. "Traction" is defined as reaching 100 or more Steam reviews.

**What is inside**

- Data loading with automatic repair of the malformed CSV header
- Data quality audit and feature engineering (parsed dates, owner midpoints, language and tag counts, multi-hot genres)
- Market overview: supply growth, pricing and free-to-play trends
- Statistical analysis with effect sizes and confidence intervals
  - Spearman correlations
  - Mann-Whitney and Kruskal-Wallis tests
  - Chi-square tests with odds ratios
  - Benjamini-Hochberg false-discovery control
  - Lorenz curve and Gini coefficient
- Genre opportunity map (demand share versus supply share)
- Machine learning with a time-based split and leakage controls
  - Traction classifier (logistic regression baseline and gradient boosting)
  - Review-sentiment regressor
  - Tag-based market segmentation with k-means

**Selected findings**

- The market is winner-take-most: the top 1% of games hold about 75% of all reviews (Gini 0.97), and only 18.6% of released titles reach 100+ reviews.
- Hit rate rises from about 21% for games priced at $0-2 to about 37% for $10-20, while sentiment barely changes across price bands.
- Games that support Mac or Linux are about twice as likely to reach 100+ reviews (odds ratio 2.10). This is an association, not a causal effect.
- Launch-time metadata predicts traction (ROC-AUC 0.841, top decile about 4.2x the base rate); the number of supported languages is the strongest feature. Review sentiment is essentially unpredictable from store metadata.
- Tag-based segments differ sharply in traction, from about 49% (Dating Sim / Visual Novel) to about 17% (VR / Casual / Indie).

## Project 2: Recommendation system

**Notebook:** [`steam-game-recommendation-system.ipynb`](steam-game-recommendation-system.ipynb)

A recommender for the Steam catalogue that works from the catalogue alone, using game descriptions, user-voted tags and review statistics. It answers three questions: *games like X*, *what to play given what I like and dislike*, and *"I want something like ..."* in free text.

**What is inside**

- Cleaning pipeline: header repair, removal of duplicate rows, and a quality-filtered catalogue of **41,513 games** (reviewed, English-language, non-adult, one entry per title)
- NLP pipeline: HTML and URL stripping, domain-aware stop words, uni- and bi-gram TF-IDF, keyword extraction
- Hybrid feature space combining description text and tags
- Recommenders
  - Content-based similarity (TF-IDF hybrid)
  - Latent Semantic Analysis (150-dimensional dense vectors)
  - Quality-aware re-ranking with a Bayesian-average rating
  - Personalised recommendations from a liked and disliked taste profile
  - Free-text semantic search
  - Explainable results (shared tags and keywords)
  - Optional transformer sentence embeddings (off by default)
- Offline evaluation on 500 random query games with genre-based relevance, diversity, coverage and popularity-bias metrics
- t-SNE map of the embedding space

**Evaluation**

No user interaction data exists, so relevance is measured against Steam's official genres, which the models never see. Differences of about 0.01 are within noise at this sample size.

| Model | Genre Jaccard@10 |
|---|---|
| LSA, 150-d (text + tags) | 0.587 |
| TF-IDF hybrid (text + tags) | 0.571 |
| Hybrid + quality re-rank (beta = 0.25) | 0.566 |
| Tags only | about 0.56 |
| Description text only | about 0.46 |
| Random and popularity baselines | about 0.28 |

**Example usage**

```python
recommend_similar("Stardew Valley")                       # games like a title
recommend_for_user(liked=["Stellaris", "Factorio"],
                   disliked=["Counter-Strike"])           # taste profile
search_games("cozy farming with relationships and crafting")   # free-text search
recommend_with_reasons("Hades")                           # shows shared tags and keywords
```

## Dataset

- **Source:** [130K Steam Games: Prices, Tags, Reviews, CCU on Kaggle](https://www.kaggle.com/datasets/sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu/data)
- **Size:** 130,100 rows and 40 columns in `games.csv`
- **Prices** are in USD. Playtime columns are in minutes. `Estimated owners` is a range, not an exact count.
- **Zeros are common** for owners, reviews and playtime because the data includes playtests, demos and new releases.

**Known issues, handled in both notebooks**

- The raw header lists 39 column names but every row has 40 values (`Discount` and `DLC count` are merged into `DiscountDLC count`). Both notebooks supply the 40 correct names explicitly when reading the file.
- About 4,250 rows are identical to another row in every column except `AppID`. The recommendation notebook removes them.

## Getting started

```bash
git clone https://github.com/sibamsamanta7/Steam-Games.git
cd Steam-Games
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
# optional, only for the transformer step in the recommender
pip install sentence-transformers
```

Download the data from Kaggle (with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured) and place `games.csv` next to the notebooks:

```bash
kaggle datasets download -d sibamsamanta07/130k-steam-games-prices-tags-reviews-ccu --unzip
```

Then open a notebook with `jupyter notebook`.

- The **market intelligence** notebook looks for `games.csv` in the working directory and under `/kaggle/input`.
- The **recommendation** notebook reads the path from `DATA_PATH` in its configuration section (2.3). It defaults to the Kaggle input path, so change it to `games.csv` when running locally.
- Both notebooks also run directly on Kaggle with the dataset attached.

## Repository structure

```
Steam-Games/
|-- README.md
|-- steam-game-market-intelligence.ipynb
|-- steam-game-recommendation-system.ipynb
```

## Limitations

- **Review counts are a proxy for sales.** They leave out revenue, marketing spend, team size and storefront featuring.
- **Newer games have had less time to collect reviews.** The market notebook uses a mature cohort and a time-based split to reduce this bias; the recommender can only suggest games that already have a review track record, so recent releases are under-represented.
- **Findings are associations, not causes.** Price, platform and genre effects are confounded with studio size and budget.
- **The recommender is content-based.** With no play or rating histories it cannot learn what people with similar taste enjoyed, and its genre-based evaluation favours tag-based models. Only an online test (clicks, play time, wishlists) can measure real satisfaction.

## License

Add a `LICENSE` file before publishing. Check that the license you choose is compatible with the terms under which the underlying Steam data was collected.

## Author

Built by [@sibamsamanta7](https://github.com/sibamsamanta7). Dataset on Kaggle: [sibamsamanta07](https://www.kaggle.com/sibamsamanta07).
