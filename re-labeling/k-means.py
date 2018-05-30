import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pdb
import pandas as pd
from scipy import stats

FEATURE_FILE_NAME = "1000_3dims.npy"
CLEANED_FILE_NAME = "train_info_1000.csv"
num = 1000

def main():
    print("now loading ...")
    X = np.load("/data/unagi0/food/tmp/dst_feature/" + FEATURE_FILE_NAME)

    print("calc kmeans ...")
    kmeans = KMeans(n_clusters=211, random_state=0).fit(X)
    kmeans_labels = kmeans.labels_

    ###
    gt = pd.read_csv("/data/unagi0/food/annot/train_info.csv", header=None)[:num]
    ###

    gt_labels = gt[1]
    kmeans_labels = pd.DataFrame(kmeans_labels)
    df = pd.concat([gt_labels, kmeans_labels], axis=1)
    groups = df.groupby(0)
    mode = groups.agg(lambda x: stats.mode(x)[0])
    dst = kmeans_labels.merge(mode, left_on=0, right_index=True, how='left')
    dst = dst.drop(columns=0)
    dst = pd.concat([gt.drop(columns=1), dst], axis=1)
    dst.to_csv("/data/unagi0/food/tmp/dst_feature/" + CLEANED_FILE_NAME, header=False, index=False)

if __name__ == "__main__":
    main()
