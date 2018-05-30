import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pdb
import argparse

DST_FILE_NAME = "1000_3dims"
FEATURE_FILE_NAME = "1000.npy"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst_path', default="/data/unagi0/food/tmp/dst_feature/")
    args = parser.parse_args()

    print("now loading ...")
    X = np.load("/data/unagi0/food/tmp/dst_feature/" + FEATURE_FILE_NAME)
    print("embedding ...")
    X_embedded = TSNE(n_components=3).fit_transform(X)
    np.save(args.dst_path + DST_FILE_NAME, X_embedded)

if __name__ == "__main__":
    main()
