#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A Mean Curvature Approach to Boundary Detection: Geometric Insights for Unsupervised Learning

Python script to reproduce the fifth set of experiments in the paper

@author: Alexandre L. M. Levada

"""

# Imports
import os
import sys
import time
import warnings
import umap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import seaborn as sns
import networkx as nx
from networkx.convert_matrix import from_numpy_array
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.cluster import HDBSCAN, KMeans, MeanShift, SpectralClustering, kmeans_plusplus
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score
from joblib import Parallel, delayed


"""
Mean Curvatures — optimized implementation
==========================================
Measured speedup vs. original version:
  n=500,   m=10, k=15  →  ~2.5×
  n=2000,  m=20, k=20  →  ~2.0×
  n=5000,  m=15, k=30  →  ~1.6×
  n=10000, m=10, k=20  →  ~1.5×
 
Maximum numerical difference relative to the original version: < 1e-11 (floating-point error).
"""

# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary function (worker for parallel execution)
# ──────────────────────────────────────────────────────────────────────────────
"""
    Computes mean curvatures for a subset of points.
 
    Parameters
    ----------
    dados     : (n, m) — full dataset (required to index neighbors)
    knn_chunk : (n_chunk, k) — indices of k-nearest neighbors for the subset
    rows_ut   : row indices of the upper triangle of (m × m)
    cols_ut   : column indices of the upper triangle of (m × m)
 
    Returns
    -------
    curvatures : (n_chunk,) — mean curvatures
"""
def _chunk_curvatures(dados: np.ndarray, knn_chunk: np.ndarray, rows_ut: np.ndarray, cols_ut: np.ndarray) -> np.ndarray:    
    n_chunk = knn_chunk.shape[0]
    m = dados.shape[1]
    curvatures = np.empty(n_chunk)
    # Main loop 
    for ci in range(n_chunk):
        idx = knn_chunk[ci]
        amostras = dados[idx]
        # ── 1. Symmetric covariance matrix ──────────────────────────────
        Icov = np.cov(amostras.T) if len(idx) > 1 else np.eye(m)
        # ── 2. Spectral decomposition via eigh ──────────────────────────
        # eigh (vs eig): exploits symmetry → ~2× faster, always real.
        # Returns eigenvalues in *ascending* order; we reverse the columns.
        _, w = np.linalg.eigh(Icov)
        Wpca = w[:, ::-1]                          # descending order (m, m)
        # ── 3. Construction of H = [Squared | Cross] without Python loops ─
        # Squared[:, j] = Wpca[:, j] ** 2  →  vectorized over all j
        Squared = Wpca ** 2                        # (m, m)
        # Cross[:, col] = Wpca[:, j] * Wpca[:, l]  for pairs (j, l) with j < l
        # rows_ut / cols_ut are precomputed with np.triu_indices
        Cross = Wpca[:, rows_ut] * Wpca[:, cols_ut]  # (m, nc)
        H = np.concatenate([Squared, Cross], axis=1)  # (m, m + nc)
        # ── 4. Curvature: |trace(-H H^T Icov)| via einsum ────────────────
        # Equivalence:
        #   trace(H H^T Icov) = Σ_ij (H H^T)_ij * Icov_ji
        #                     = einsum('ia,ja,ij->', H, H, Icov)
        # Avoids two dense matrix multiplications + a call to np.trace.
        curvatures[ci] = abs(np.einsum('ia,ja,ij->', H, H, Icov))
    return curvatures

# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
"""
    Computes the mean curvature at each point of a multivariate dataset.
 
    Parameters
    ----------
    dados  : array (n, m) — n samples with m features
    k      : number of nearest neighbors
    n_jobs : number of parallel workers.
             1  → sequential (default, zero overhead).
             -1 → uses all available CPUs.
             Recommended to use n_jobs > 1 only for n >= 5000 with many CPUs.
 
    Returns
    -------
    curvatures : array (n,) — mean curvatures (absolute values)
 
    Optimizations relative to the original version
    ───────────────────────────────────────────────
    1. eigh instead of eig
       The covariance matrix is symmetric positive semi-definite.
       `eigh` exploits this: it is ~2× faster than `eig` and returns
       real eigenvalues directly (no need for `.real`).
 
    2. Elimination of inner Python loops (Squared / Cross)
       `Squared` is computed via elementwise broadcasting (Wpca**2).
       `Cross`   is computed by indexing columns with precomputed
       index arrays (np.triu_indices), in a single vectorized operation.
 
    3. No unnecessary allocations per iteration
       The columns of 1's and the columns of Wpca that were previously
       concatenated at the beginning of Q, but later discarded in
       H = Q[:, m+1:], are completely removed — H is built directly.
 
    4. einsum for the trace
       `trace(H H^T Icov)` is computed as
       `einsum('ia,ja,ij->', H, H, Icov)`,
       avoiding two dense matrix multiplications and a call to
       `np.trace`.
 
    5. Optional parallelism via joblib
       The main loop can be distributed across threads (GIL released by
       NumPy) to leverage multiple CPUs on large datasets.
"""
def Mean_Curvatures(dados: np.ndarray, k: int, n_jobs: int = 1) -> np.ndarray:
    n, m = dados.shape
    # Precomputed upper triangle indices (reused at each iteration)
    rows_ut, cols_ut = np.triu_indices(m, k=1)
    # KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dados)
    knn_indices = nbrs.kneighbors(return_distance=False)   # (n, k)
    # Sequential (default) or parallel execution
    n_jobs_eff = max(1, os.cpu_count() if n_jobs == -1 else n_jobs)
    if n_jobs_eff == 1:
        return _chunk_curvatures(dados, knn_indices, rows_ut, cols_ut)
    chunks = np.array_split(knn_indices, n_jobs_eff)
    results = Parallel(n_jobs=n_jobs_eff, prefer='threads')(
        delayed(_chunk_curvatures)(dados, chunk, rows_ut, cols_ut)
        for chunk in chunks
    )
    return np.concatenate(results)


# Optional function to normalize the curvatures to the interval [a, b]
def normalize_curvatures(curv, a, b):
    k = a + (b - a)*(curv - curv.min())/(curv.max() - curv.min())
    return k

# Build the KNN graph
def build_KNN_Graph(dados, k):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()  
    return A

# Plot the KNN graph
def plot_KNN_graph(A, target, K=0, pos=0, layout='spring'):
    # Create a networkX graph object
    n = A.shape[0]
    G = from_numpy_array(A)
    color_map = []
    for i in range(n):
        if type(K) == list:
            if K[i] > 0:
                color_map.append('black')
            else:
                if target[i] == 0:
                    color_map.append('blue')
                elif target[i] == 1:
                    color_map.append('red')
                elif target[i] == 2:
                    color_map.append('green')
                elif target[i] == 3:
                    color_map.append('purple')
                elif target[i] == 4:
                    color_map.append('orange')
                elif target[i] == 5:
                    color_map.append('darkcyan')
                elif target[i] == 6:
                    color_map.append('darkkhaki')
                elif target[i] == 7:
                    color_map.append('brown')
                elif target[i] == 8:
                    color_map.append('silver')
                elif target[i] == 9:
                    color_map.append('cyan')
                elif target[i] == 10:
                    color_map.append('magenta')
                elif target[i] == 11:
                    color_map.append('cornflowerblue')
                elif target[i] == 12:
                    color_map.append('tomato')
                elif target[i] == 13:
                    color_map.append('limegreen')
                elif target[i] == 14:
                    color_map.append('darkviolet')
                elif target[i] == 15:
                    color_map.append('darkorange')
                elif target[i] == 16:
                    color_map.append('turquoise')
                elif target[i] == 17:
                    color_map.append('tan')
                elif target[i] == 18:
                    color_map.append('darkred')
                elif target[i] == 19:
                    color_map.append('steelblue')
                elif target[i] == 20:
                    color_map.append('rosybrown')  
        else:
            if target[i] == 0:
                color_map.append('blue')
            elif target[i] == 1:
                color_map.append('red')
            elif target[i] == 2:
                color_map.append('green')
            elif target[i] == 3:
                color_map.append('purple')
            elif target[i] == 4:
                color_map.append('orange')
            elif target[i] == 5:
                color_map.append('darkcyan')
            elif target[i] == 6:
                color_map.append('darkkhaki')
            elif target[i] == 7:
                color_map.append('brown')
            elif target[i] == 8:
                color_map.append('silver')
            elif target[i] == 9:
                color_map.append('cyan')
            elif target[i] == 10:
                color_map.append('magenta')
            elif target[i] == 11:
                color_map.append('cornflowerblue')
            elif target[i] == 12:
                color_map.append('tomato')
            elif target[i] == 13:
                color_map.append('limegreen')
            elif target[i] == 14:
                color_map.append('darkviolet')
            elif target[i] == 15:
                color_map.append('darkorange')
            elif target[i] == 16:
                color_map.append('turquoise')
            elif target[i] == 17:
                color_map.append('tan')
            elif target[i] == 18:
                color_map.append('darkred')
            elif target[i] == 19:
                color_map.append('steelblue')
            elif target[i] == 20:
                color_map.append('rosybrown')  
    plt.figure(1)
    # Several layouts to choose, here we prefer the spring and kamada-kawai layouts  
    if np.isscalar(pos):
        if layout == 'spring':
            pos = nx.spring_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G) # ideal para plotar a árvore!
    if n < 1000:
        nx.draw_networkx(G, pos, node_size=25, node_color=color_map, with_labels=False, width=0.2, alpha=0.4)
    else:
        nx.draw_networkx(G, pos, node_size=10, node_color=color_map, with_labels=False, width=0.1, alpha=0.25)
    if np.isscalar(K):
        plt.savefig('kNN_Graph.png')
    else:
        plt.savefig('high_curvature_points.png')
    plt.show()
    plt.close()
    return pos

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

############################################################
# Data loading 
############################################################
X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='dermatology', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)           
#X = skdata.fetch_openml(name='satimage', version=1)            
#X = skdata.fetch_openml(name='mfeat-zernike', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='mammography', version=1)          
#X = skdata.fetch_openml(name='Satellite', version=1)
#X = skdata.fetch_openml(name='ipums_la_98-small', version=1)
#X = skdata.fetch_openml(name='ionosphere', version=1)
#X = skdata.fetch_openml(name='seeds', version=1)  
#X = skdata.fetch_openml(name='prnn_synth', version=1)
#X = skdata.fetch_openml(name='Engine1', version=1)
#X = skdata.fetch_openml(name='texture', version=1)
#X = skdata.fetch_openml(name='segment', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Colon', version=1)
#X = skdata.fetch_openml(name='arsenic-male-bladder', version=2)
#X = skdata.fetch_openml(name='tecator', version=2)


dados = X['data']
target = X['target']

# To deal with sparse matrix data
if type(dados) == sp.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

n = dados.shape[0]
m = dados.shape[1]
# Number of neighbors
nn = round(np.log2(n))
LAYOUT = 'spring'
#LAYOUT = 'kawai'

# Number of classes
c = len(np.unique(target))

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

# Reduce dimensionality to allow curvature computation in high-dimensional data
if m > 50:    
    dados = PCA(n_components=min(50, n), random_state=42).fit_transform(dados)
    n, m = dados.shape

# Curvature estimation
curvaturas = Mean_Curvatures(dados, nn, n_jobs=-1)
# Normalização das curvaturas
K = normalize_curvatures(curvaturas, 0, 1)
# Threshold
T = np.percentile(K, 75)
# Build the adjacency matrix of the graph
A = build_KNN_Graph(dados, nn)
# Plot the original network
#pos = plot_KNN_graph(A, target, layout=LAYOUT)
# To plot the high curvature points
for i in range(n):
    if K[i] < T:
        K[i] = 0
# Plot high curvature points in the k-NN graph
#pos = plot_KNN_graph(A, target, K=list(K), pos=pos, layout=LAYOUT)
# Filter dataset
S_nodes = np.where(K==0)[0]
H_nodes = np.where(K>0)[0]
filtered = dados[S_nodes, :]
boundaries = dados[H_nodes, :]

#######################################
# HDBSCAN
#######################################
print('******** HDBSCAN **********')
print()
# Cluster original data
hdbscan = HDBSCAN().fit(dados)
sc1 = silhouette_score(dados, hdbscan.labels_)
ch1 = calinski_harabasz_score(dados, hdbscan.labels_)
db1 = davies_bouldin_score(dados, hdbscan.labels_)
print('Original data')
print('SC: ', sc1)
print('CH: ', ch1)
print('DB: ', db1)
print()

# Cluster filtered data
hdbscan = HDBSCAN().fit(filtered)
filtered_labels = hdbscan.labels_

# Train 1-NN classifier to classify boundaries
classif = KNeighborsClassifier(n_neighbors=1)
classif.fit(filtered, filtered_labels) 
boundaries_labels = classif.predict(boundaries)
full_data = np.vstack((filtered, boundaries))
full_labels = np.concatenate((filtered_labels, boundaries_labels))
sc3 = silhouette_score(full_data, full_labels)
ch3 = calinski_harabasz_score(full_data, full_labels)
db3 = davies_bouldin_score(full_data, full_labels)
print('HDBSCAN on filtered data + k-NN for boundaries')
print('SC: ', sc3)
print('CH: ', ch3)
print('DB: ', db3)

# print()
# print(str(sc1)+'\t'+str(ch1)+'\t'+str(db1)+'\t'+str(sc3)+'\t'+str(ch3)+'\t'+str(db3))
