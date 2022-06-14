#!/usr/bin/env python3
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from common import describe_data, test_env

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def main():
    print("--- LAB4 --- ")

    # Show environment version
    modules = ['pandas', 'sklearn', 'numpy', 'matplotlib']
    test_env.versions(modules)

    # Read dataset ==> pandas.dataframe
    df = pd.read_csv('data/Wholesale_customers_data.csv')

    # Output overview and categorical
    describe_data.print_overview(df,
                                 file='results/overview.txt')
    describe_data.print_categorical(df,
                                    file='results/categorical.txt')

    # Preprocessing
    data = np.array([df['Fresh'].tolist(),
                     df['Milk'].tolist(),
                     df['Grocery'].tolist(),
                     df['Frozen'].tolist(),
                     df['Milk'].tolist(),
                     df['Detergents_Paper'].tolist(),
                     df['Delicassen'].tolist()
                     ], np.int32).T

    # Scaling
    mm = StandardScaler()
    data = mm.fit_transform(data)
    print(f'Data shape: {data.shape}')

    # Visualise dataset with TNCA with 2 dim
    X_tsne = TSNE(n_components=2).fit_transform(data)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title('TSNE')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('results/tsne.png')
    print("Save fig to results/tsne.png")
    plt.clf()

    # Choosing number of clusters using Elbow method for K-means
    cluster_range = list(range(1, 100))
    sse = []  # SSE : distortion of clusters
    for k in cluster_range:
        model = KMeans(n_clusters=k, random_state=0).fit(data)
        sse.append(model.inertia_)
    plt.title('KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.plot(cluster_range, sse)
    plt.scatter(cluster_range, sse)
    plt.savefig('results/elbow_kmeans.png')
    print("Save fig to results/elbow_kmeans.png")
    plt.clf()

    '''
    Note:
    Considering that the elbow appares around 15 ~ 20
    I will choose 18 for suitable cluster number
    '''
    km = KMeans(n_clusters=18, random_state=0).fit_predict(data)

    # Choosing parameters for DBSCAN
    '''
    print("-- param options for DBSCAN --")
    for eps in np.arange(0.1, 3, 0.5):
        for minPts in range(1, 10):
            db = DBSCAN(eps=eps, min_samples=minPts).fit_predict(data)
            if np.max(db) != -1 and np.max(db) != 0 :
                print(f'eps:{eps} minPts:{minPts}')
                print(f'Claster num : {np.max(db)}')
    print("-- END : param options for DBSCAN -- \n")
    '''

    '''
    Note:
    I don't see much change from graphs 
    I'll choose eps(0.5) and minPts(2) in this case
    '''
    db = DBSCAN(eps=0.5, min_samples=2).fit_predict(data)

    # K-means : Visualise dataset and clusters with TNCA with 2 dim
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=km)
    plt.title('TSNE : K-Means')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('results/tsne_kmeans.png')
    print("Save fig to results/tsne_kmeans.png")
    plt.clf()

    # DBSCAN : Visualise dataset and clusters with TNCA with 2 dim
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=db)
    plt.title('TSNE : DBSCAN')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.savefig('results/tsne_dbscan.png')
    print("Save fig to results/tsne_dbscan.png")
    plt.clf()

    print("-- END: LAB4 --")
    print('Done')


if __name__ == "__main__":
    main()
