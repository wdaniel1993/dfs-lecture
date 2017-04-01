# visualize and compare clustering algorithms on the iris data set

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn import preprocessing
import hdbscan
import time
import pandas as pd
from sklearn import datasets



sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 20, 'linewidths':0}

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    print('number of clusters: ', len(set(labels)) - (1 if -1 in labels else 0))
    end_time = time.time()
    palette = sns.color_palette('Set2', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('{}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(1, -2, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig(algorithm.__name__ + '.pdf', bbox_inches='tight')
    
#iris = datasets.load_iris()
#%X = iris.data[:,:4]
#Y = iris.target

df = pd.read_csv('../data/s1.txt', sep='\s+')
data = preprocessing.scale(df.values)
print(data)

# k-means clustering
#plot_clusters(data, cluster.KMeans, (), {'n_clusters':15})

# affinity-propagationllll
#plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})

# mean-shift
#bandwidth = cluster.estimate_bandwidth(data)
#print('bandwidth: ', bandwidth)
#plot_clusters(data, cluster.MeanShift, (0.4,), {})

# spectral clustering
#plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters': 15})

# agglomerative clustering
#plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':15, 'linkage':'ward'})

# dbscan
#plot_clusters(data, cluster.DBSCAN, (), {'eps':0.15, 'min_samples':30})

# hdbscan
plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':30})