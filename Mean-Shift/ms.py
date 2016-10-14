
# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import MeanShift
from sklearn.metrics import euclidean_distances


if __name__ == "__main__":
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)

    matplotlib.rcParams['font.sans-serif'] = [u'Times New Roman']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 10), facecolor='w')
    m = euclidean_distances(data, squared=True)
    bw = np.median(m)
    print bw
    for i, mul in enumerate(np.linspace(0.1, 0.4, 4)):
        band_width = mul * bw
        model = MeanShift(bin_seeding=True, bandwidth=band_width)
        ms = model.fit(data)
        centers = ms.cluster_centers_
        y_hat = ms.labels_
        n_clusters = np.unique(y_hat).size
        print 'bandwidth：', mul, band_width, '#cluster：', n_clusters

        plt.subplot(2, 2, i+1)
        plt.title(u'bandwidth：%.2f，#clusters：%d' % (band_width, n_clusters))
        clrs = []
        for c in np.linspace(16711680, 255, n_clusters):
            clrs.append('#%06x' % c)
        print clrs
        for k, clr in enumerate(clrs):
            cur = (y_hat == k)
            plt.scatter(data[cur, 0], data[cur, 1], c=clr, edgecolors='none')
        plt.scatter(centers[:, 0], centers[:, 1], s=150, c=clrs, marker='*', edgecolors='k')
        plt.grid(True)
    plt.tight_layout()
    plt.suptitle(u'MeanShift', fontsize=20)
    plt.subplots_adjust(top=0.92)
    plt.show()
