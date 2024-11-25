#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Towards boundary points detection with mean curvature estimation for unsupervised learning

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from scipy.stats import median_abs_deviation


# Computes the curvatures of all samples in the training set
def Curvature_Estimation(dados, k, curv):
    n = dados.shape[0]
    m = dados.shape[1]
    # If data has more than 80 features, reduce to 25-D with PCA
    if m > 80:
        m = 25
        model = PCA(n_components=m)
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
        Q = Wpca
        # Discard the first m columns of H
        H = Q[:, (m+1):]        
        II = np.dot(H, H.T)
        S = -np.dot(II, I)          # I measures density and II measures degree of concavity/convexity
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


# Auxiliary function for data plotting
def PlotData2D(dados, labels, K, T):
    n = dados.shape[0]
    nclass = len(np.unique(labels))
    # Convert labels to integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to nunbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    # Numpy array
    rotulos = np.array(rotulos)
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'palegreen', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'palegreen', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'palegreen', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'palegreen']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'sienna', 'darkkhaki', 'palegreen', 'purple', 'salmon']
    plt.figure(1)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='.', alpha=0.5)    
    nome_arquivo = '2D_Data.png'    
    plt.title('2D features')
    plt.savefig(nome_arquivo)    
    print(T)
    points = np.where(K>T)[0]
    plt.scatter(dados[points, 0], dados[points, 1], c='green', marker='o', alpha=0.75)
    nome_arquivo = '2D_High_Curvature_Samples.png'
    plt.title('Mean curvature boundary points')
    plt.savefig(nome_arquivo)
    plt.close()


# Auxiliary function for data plotting
def HeatMap2D(dados, labels, K, n):
    intervalos = np.linspace(0, 1, n)
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    discrete_K = np.digitize(K, bins)
    plt.scatter(dados[:, 0], dados[:, 1], c=discrete_K, marker='.', cmap='coolwarm')
    #plt.colorbar()
    nome_arquivo = 'Heatmap_2D.png'
    plt.title('Heatmap of mean curvatures')
    plt.savefig(nome_arquivo)
    plt.close()


##########################################
# Início do script
##########################################

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Sintéticos 2D
dados, target = make_blobs(n_samples=400, centers=1, n_features=2)
#dados, target = make_blobs(n_samples=500, centers=2, n_features=2)
#dados, target = make_classification(n_samples=500, n_features=2, n_redundant=0, n_clusters_per_class=1, class_sep=2.0, shift=0.5)
#dados, target = make_moons(n_samples=1000, noise=0.1)

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))
nn = round(np.log2(n))      # number of neighbors in the k-NN graph

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

# Estimate the curvatures
curvatures = Curvature_Estimation(dados, nn, curv='mean')
K = normalize_curvatures(curvatures)

# Plot normalized curvatures
plt.figure(10)
plt.plot(K)
plt.show()
plt.close()

# Plot the smooth curvatures histogram
plt.figure(11)
sns.distplot(K, hist=False)
plt.title('Mean curvatures smooth histogram')
plt.show()

kde = stats.gaussian_kde(K)
x = np.linspace(0, 1, 200)
dens = kde(x)

# Computes the threshold
T = np.percentile(K, 75)

# Plot boundary points
PlotData2D(dados, target, K, T)
HeatMap2D(dados, target, K, n)