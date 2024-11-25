#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Towards boundary points detection with mean curvature estimation for unsupervised learning

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
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score
from scipy.stats import wilcoxon

# Computes the curvatures of all samples in the training set
def Curvature_Estimation(dados, k, curv):
    n = dados.shape[0]
    m = dados.shape[1]
    # If data has more than 80 features, reduce to 25-D with PCA
    if m > 25:
        m = 10
        model = PCA(n_components=m)
        #model = umap.UMAP(n_components=m, min_dist=0.6)    
        dados = model.fit_transform(dados)
    # First fundamental form
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity', include_self=False)
    A = knnGraph.toarray()    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        # Computation of the first fundamental form
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)      # isolated points
        # Compute the eigenvectors
        v, w = np.linalg.eig(I)
        # Sort the eigenvalues
        ordem = v.argsort()
        # Select the eigenvectors in decreasing order (in columns)
        Wpca = w[:, ordem[::-1]]
        # Computation of the second fundamental form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Add a column of ones
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))
        # Gram-Schmidt ortogonalization (is it really necessary?)
        Q = Wpca
        # Discard the first m columns of H
        H = Q[:, (m+1):]        
        II = np.dot(H, H.T)
        S = -np.dot(II, I)
        if curv == 'gaussian':
            curvatures[i] = abs(np.linalg.det(S))
        elif curv == 'mean':
            curvatures[i] = abs(np.trace(S))
    return curvatures

# Optional function to normalize the curvatures to the interval [0, 1]
def normalize_curvatures(curv):
    if curv.max() != curv.min():
        k = (curv - curv.min())/(curv.max() - curv.min())
    else:
        k = curv
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
# Data loading (uncomment one dataset from the list below)
############################################################
# Small datasets
X = skdata.load_iris()
#X = skdata.load_wine()
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='diabetes', version=1)  
#X = skdata.fetch_openml(name='LED-display-domain-7digit', version=1)  
#X = skdata.fetch_openml(name='dermatology', version=1)
#X = skdata.fetch_openml(name='audiology', version=1)
#X = skdata.fetch_openml(name='led7', version=1) 
#X = skdata.fetch_openml(name='steel-plates-fault', version=1)
#X = skdata.fetch_openml(name='Flare', version=1)

### Large datasets  
#X = skdata.fetch_openml(name='pendigits', version=1)           
#X = skdata.fetch_openml(name='satimage', version=1)            
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='mfeat-morphological', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='mammography', version=1)          
#X = skdata.fetch_openml(name='USPS', version=1)   # OK
#X = skdata.fetch_openml(name='Satellite', version=1)
#X = skdata.fetch_openml(name='ipums_la_99-small', version=1)
#X = skdata.fetch_openml(name='ipums_la_98-small', version=1)
#X = skdata.fetch_openml(name='cardiotocography', version=1)
#X = skdata.fetch_openml(name='gas-drift', version=1)
#X = skdata.fetch_openml(name='first-order-theorem-proving', version=1)

### High dimensional data
#X = skdata.fetch_openml(name='AP_Colon_Ovary', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Colon', version=1)

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

# Number of classes
c = len(np.unique(target))

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

n = dados.shape[0]
m = dados.shape[1]
# Number of neighbors
nn = round(np.log2(n))
#nn = round(np.sqrt(n))
#nn = 5
LAYOUT = 'spring'
#LAYOUT = 'kawai'

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

# Curvature estimation
curvaturas = Curvature_Estimation(dados, nn, curv='mean')  
# Normalização das curvaturas
K = normalize_curvatures(curvaturas)

# Plot the normalized curvature
plt.figure(10)
plt.plot(K)
plt.show()
plt.close()

# Plot the smooth curvatures histogram
plt.figure(11)
sns.distplot(K, hist=False)
plt.title('Distribution of the mean curvatures')
plt.savefig('density.png')
plt.show()

kde = stats.gaussian_kde(K)
x = np.linspace(0, 1, 200)
dens = kde(x)

T = np.percentile(K, 75)

# Build the adjacency matrix of the graph
A = build_KNN_Graph(dados, nn)
# Plot the original network
pos = plot_KNN_graph(A, target, layout=LAYOUT)
# To plot the high curvature points
for i in range(n):
    if K[i] < T:
        K[i] = 0
# Plot high curvature points in the k-NN graph
pos = plot_KNN_graph(A, target, K=list(K), pos=pos, layout=LAYOUT)

####################################
# Kmeans
####################################
print('******** KMEANS **********')
print()
# Cluster original data
kmeans = KMeans(n_clusters=c, init='k-means++', random_state=42).fit(dados)
sc = silhouette_score(dados, kmeans.labels_)
ch = calinski_harabasz_score(dados, kmeans.labels_)
db = davies_bouldin_score(dados, kmeans.labels_)
print('Original data')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
print()

# Filter dataset
S_nodes = np.where(K==0)[0]
filtered = dados[S_nodes, :]
kmeans = KMeans(n_clusters=c, init='k-means++', random_state=42).fit(filtered)
sc = silhouette_score(filtered, kmeans.labels_)
ch = calinski_harabasz_score(filtered, kmeans.labels_)
db = davies_bouldin_score(filtered, kmeans.labels_)
print('Filtered data')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
print()

# Cluster original data with centers obtained with the filtered dataset
kmeans = KMeans(n_clusters=c, init=kmeans.cluster_centers_).fit(dados)
sc = silhouette_score(dados, kmeans.labels_)
ch = calinski_harabasz_score(dados, kmeans.labels_)
db = davies_bouldin_score(dados, kmeans.labels_)
print('Original data with centroids of S')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
print()


#######################################
# HDBSCAN
#######################################
print('******** HDBSCAN **********')
print()
# Cluster original data
#hdbscan = HDBSCAN(min_cluster_size=10).fit(dados)
hdbscan = HDBSCAN().fit(dados)
rotulos_original = hdbscan.labels_
sc = silhouette_score(dados, hdbscan.labels_)
ch = calinski_harabasz_score(dados, hdbscan.labels_)
db = davies_bouldin_score(dados, hdbscan.labels_)
print('Original data')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
print()

# Cluster filtered data
#hdbscan = HDBSCAN(min_cluster_size=10, store_centers='centroid').fit(filtered)
hdbscan = HDBSCAN(store_centers='centroid').fit(filtered)
sc = silhouette_score(filtered, hdbscan.labels_)
ch = calinski_harabasz_score(filtered, hdbscan.labels_)
db = davies_bouldin_score(filtered, hdbscan.labels_)
print('Filtered data')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
print()

###############################################
# KMEANS with HDBSCAN S-centroids
###############################################
print('******** KMEANS with HDBSCAN S-centroids **********')
print()

# Cluster original data with kmeans using centers obtained by HDBSCAN in the filtered dataset
centroids = hdbscan.centroids_
nc = centroids.shape[0]
kmeans = KMeans(n_clusters=nc, init=centroids).fit(dados)
sc = silhouette_score(dados, kmeans.labels_)
ch = calinski_harabasz_score(dados, kmeans.labels_)
db = davies_bouldin_score(dados, kmeans.labels_)
print('Original data with HDBSCAN centroids in S')
print('SC: ', sc)
print('CH: ', ch)
print('DB: ', db)
