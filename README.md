# AI Channels Feature Engineering

Вторая часть исследования вовлеченности аудитории в русскоязычных Telegram-каналах про AI/IT. Если `ai_channels_research` собирает и чистит данные, то здесь они превращаются в признаки: метрики читаемости текста, векторные представления постов, кластеры по темам и сравнение этих кластеров между разными моделями эмбеддингов.

На входе: итоговый датасет из `ai_channels_research` (10 479 постов, 10 каналов). На выходе: посты, размеченные по темам четырьмя разными способами, и сравнение того, насколько эти разметки согласуются друг с другом и как вовлеченность различается по темам.

## Что внутри

- **Очистка текста** (`notebooks/1_text_processing.ipynb`) — чистит текст от ссылок и форматирования, разбивает датасет на логические блоки: метаданные, текст, вовлеченность, реакции.
- **Метрики читаемости** (`notebooks/2_ruts_metrics.ipynb`) — считает лингвистические метрики текста через библиотеку `ruTS` (FRE, FKG, TTR и другие).
- **Связь метрик с вовлеченностью** (`notebooks/2_ruts_engagement_analysis.ipynb`) — проверяет, зависит ли вовлеченность аудитории от сложности и характеристик текста.
- **Векторизация** (`notebooks/3_vectorization.ipynb`) — подготовка текстов к четырем разным представлениям: TF-IDF, SBERT, E5, BGE-M3.
- **Кластеризация** (`notebooks/4_clusterization_tfidf.ipynb`, `..._sbert.ipynb`, `..._e5.ipynb`, `..._bge.ipynb`) — для каждой модели эмбеддингов отдельно подбираются параметры PCA и k, строятся кластеры (KMeans и HDBSCAN), кластерам присваиваются названия тем.
- **Сравнение кластеров** (`notebooks/cluster_comparison.ipynb`) — сводит результаты всех четырех моделей в одну таблицу метрик, приводит названия тем к единому списку категорий, строит кросс-таблицы совпадения тем между моделями и смотрит на вовлеченность и реакции по темам.
- **Анализ реакций** (`notebooks/reaction_analysis.ipynb`) — отдельный, не входящий в основной пайплайн ноутбук: сколько раз встречается каждая реакция у каждого канала.

## Пайплайн (порядок ноутбуков)

```
1_text_processing.ipynb
        |
        v
2_ruts_metrics.ipynb  ->  2_ruts_engagement_analysis.ipynb
        |
        v
3_vectorization.ipynb
        |
        v
4_clusterization_tfidf.ipynb
4_clusterization_sbert.ipynb
4_clusterization_e5.ipynb
4_clusterization_bge.ipynb
        |
        v
cluster_comparison.ipynb
```

`reaction_analysis.ipynb` стоит в стороне от этой цепочки — он читает исходный датасет напрямую и не связан с остальными шагами.

## Структура проекта

```
ai_channels_feature_engineering/
├── notebooks/
│   ├── 1_text_processing.ipynb
│   ├── 2_ruts_metrics.ipynb
│   ├── 2_ruts_engagement_analysis.ipynb
│   ├── 3_vectorization.ipynb
│   ├── 4_clusterization_tfidf.ipynb
│   ├── 4_clusterization_sbert.ipynb
│   ├── 4_clusterization_e5.ipynb
│   ├── 4_clusterization_bge.ipynb
│   ├── cluster_comparison.ipynb      # сравнение кластеров всех моделей
│   ├── reaction_analysis.ipynb       # отдельный анализ реакций
│   ├── utils.py                      # общие вспомогательные функции
│   └── tables/                       # csv из более ранних версий анализа вовлеченности, текущим кодом не генерируются
└── data/
    ├── dataset_raw.csv               # копия итогового датасета из ai_channels_research (вход пайплайна)
    ├── dataset_cleaned.csv           # результат 1_text_processing.ipynb
    ├── dataset_texts_only.csv        # только текст постов (channel, post_id, text, text_clean)
    ├── dataset_engagement_features_draft.csv  # черновая ветка с метриками вовлеченности и форматирования текста, не используется в текущем пайплайне
    ├── split/                        # посты, разбитые по блокам
    │   ├── 01_meta.csv
    │   ├── 02_text.csv
    │   ├── 02_text_with_metrics.csv  # + метрики ruTS
    │   ├── 02_text_for_vectorization.csv
    │   ├── 03_engagement.csv
    │   └── 04_reactions.csv
    ├── embeddings/                   # векторные представления (.npy) по каждой модели
    ├── results/
    │   ├── metrics/                  # метрики кластеризации (silhouette и др.) по моделям
    │   ├── final_clusters/           # посты с номером и названием кластера по моделям
    │   ├── candidate_examples/       # примеры постов по кластерам для ручной проверки
    │   ├── final_examples/
    │   └── channel_summary_table.csv
    └── plots/                        # все графики пайплайна (ноутбуки 2, 4, cluster_comparison)
        └── legacy_unused/             # графики из более ранних версий анализа, текущим кодом не генерируются
```

## Как это связано с ai_channels_research

Этот проект начинается с датасета, который `ai_channels_research` выгружает в `data/final/ai_publics_dataset.csv`. Этот файл вручную скопирован сюда как `data/dataset_raw.csv` и дальше проходит через весь пайплайн выше. Если исходный датасет в `ai_channels_research` обновится, `data/dataset_raw.csv` нужно будет скопировать заново и прогнать ноутбуки с начала.

## Установка

```bash
pip install -r requirements.txt
```

Виртуальные окружения (venv), в которых реально считался проект, в репозиторий не входят. `requirements.txt` собран из них отдельно: ноутбуки `1_text_processing`, `2_ruts_metrics`, `2_ruts_engagement_analysis`, `cluster_comparison` и `reaction_analysis` использовали один набор библиотек (`ruts`, `nltk`, `statsmodels`, `emoji`), а `3_vectorization` и `4_clusterization_*` — другой, с моделями эмбеддингов (`sentence-transformers`, `umap-learn`, `hdbscan`). Версии пакетов из этих двух окружений не проверялись на совместимость друг с другом в одном venv.
