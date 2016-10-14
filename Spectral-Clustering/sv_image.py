# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
from PIL import Image
import time


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = [u'Times New Roman']
    matplotlib.rcParams['axes.unicode_minus'] = False

    pic = Image.open('/Users/yunhuaxiang/Desktop/ML2/12/12.file.png')
    pic = pic.convert('L')
    data = np.array(pic).astype(np.float) / 255

    plt.figure(figsize=(10, 5), facecolor='w')
    plt.subplot(121)
    plt.imshow(pic, cmap=plt.cm.gray, interpolation='nearest')
    plt.title(u'raw_pic', fontsize=18)
    n_clusters = 5

    affinity = image.img_to_graph(data)
    beta = 3
    affinity.data = np.exp(-beta * affinity.data / affinity.data.std()) + 10e-5
    # a = affinity.toarray()
    # b = np.diag(a.diagonal())
    # a -= b
    print 'start sc'
    y = spectral_clustering(affinity, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)
    print 'finish sc'
    y = y.reshape(data.shape)
    for n in range(n_clusters):
        data[y == n] = n
    plt.subplot(122)
    clrs = []
    for c in np.linspace(16776960, 16711935, n_clusters):
        clrs.append('#%06x' % c)
    cm = matplotlib.colors.ListedColormap(clrs)
    plt.imshow(data, cmap=cm, interpolation='nearest')
    plt.title(u'spectral clustering：%d clusters' % n_clusters, fontsize=18)
    plt.tight_layout()
    plt.show()
