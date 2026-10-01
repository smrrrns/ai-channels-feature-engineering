import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

def dunn_index(X, labels):
    """Индекс Данна для оценки качества кластеризации"""
    unique_clusters = np.unique(labels)
    clusters = [X[labels == c] for c in unique_clusters]
    
    # Межкластерные расстояния (минимальные)
    intercluster = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = cdist(clusters[i], clusters[j], metric='euclidean')
            intercluster.append(dist.min())
    min_intercluster = np.min(intercluster)
    
    # Внутрикластерные расстояния (максимальные)
    intracluster = []
    for cluster in clusters:
        if len(cluster) > 1:
            dist = pdist(cluster, metric='euclidean')
            intracluster.append(dist.max())
    max_intracluster = np.max(intracluster)
    
    return min_intercluster / max_intracluster if max_intracluster > 0 else 0

def compute_gap_statistic(X, k, n_refs=5, random_state=42):
    """Gap statistic для оценки числа кластеров"""
    km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    km.fit(X)
    orig_disp = km.inertia_
    
    ref_disps = []
    for _ in range(n_refs):
        random_reference = np.random.random_sample(size=X.shape)
        km_ref = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        km_ref.fit(random_reference)
        ref_disps.append(km_ref.inertia_)
    
    gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
    return gap

def get_cluster_report(df, cluster_col):
    """Создаёт отчет по кластерам: размеры, доли, проблемы"""
    sizes = df[cluster_col].value_counts().sort_index()
    n_clusters = len(sizes)
    total = len(df)
    
    report = {
        'n_clusters': n_clusters,
        'sizes': sizes.to_dict(),
        'min_size': sizes.min(),
        'max_size': sizes.max(),
        'max_size_pct': sizes.max() / total,
        'n_micro_clusters': sum(sizes < 50),
        'has_dump_cluster': (sizes.max() / total) > 0.4,
        'balance_ratio': sizes.min() / sizes.max() if sizes.max() > 0 else 0
    }
    return report

def get_centroid_examples(X, labels, df_texts, n_examples=20):
    """Возвращает примеры, ближайшие к центроиду каждого кластера"""
    from sklearn.metrics.pairwise import euclidean_distances
    
    results = {}
    unique_clusters = np.unique(labels)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # для HDBSCAN noise
            continue
            
        mask = labels == cluster_id
        cluster_points = X[mask]
        cluster_indices = np.where(mask)[0]
        
        # Центроид кластера
        centroid = cluster_points.mean(axis=0)
        
        # Расстояния до центроида
        distances = euclidean_distances(cluster_points, [centroid]).flatten()
        
        # Индексы ближайших постов
        nearest_idx = np.argsort(distances)[:n_examples]
        original_indices = cluster_indices[nearest_idx]
        
        results[cluster_id] = {
            'indices': original_indices,
            'texts': df_texts.iloc[original_indices]['text'].tolist(),
            'distances': distances[nearest_idx].tolist()
        }
    
    return results