import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pdb

def main():
    print("now loading ...")
    X = np.load("/data/unagi0/food/tmp/dst_feature/1000.npy")
    print("embedding ...")
    X_embedded = TSNE(n_components=10).fit_transform(X)
    print("calc kmeans ...")
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pdb.set_trace()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.savefig("1000.png")

if __name__ == "__main__":
    main()
