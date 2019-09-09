"""
refer to multi_class2multi_task_MNIST
"""
import numpy as np
import argparse

from data_synthesizer.multi_class2multi_task_MNIST import mutliclass2multitask, data_crop
from data_synthesizer import write2file


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        data_dir = "C:/Users/zjs50/Downloads/personal/MTSEM/data/animals_with_attributes/AwA2-features/Animals_with_Attributes2/Features/ResNet101"
        parser.add_argument(
            "-ff", "--feature_file",
            default=data_dir + "/AwA2-features.txt"
        )
        parser.add_argument(
            "-lf", "--label_file",
            default=data_dir + "/AwA2-labels.txt"
        )
        parser.add_argument("-pnc", "--pca_n_components", default=500, type=int)
        parser.add_argument("-nspl", "--n_sample_p_label", default=100, type=int)
        parser.add_argument("-np", "--n_pos", default=20, type=int)
        parser.add_argument("-nnl", "--n_neg_l", default=2, type=int)
        parser.add_argument("-ld", "--label_dim", default=50, type=int)
        parser.add_argument("-rs", '--random_shuffle', default=True, action="store_false")
        parser.add_argument("-s", "--seed", default=2019, type=int)
        parser.add_argument("-dn", '--dir_name', type=str, default="../data/AwA2")
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


def read_feature(feature_file):
    with open(feature_file, "r") as f:
        features = []
        for line in f:
            feature = line.rstrip().split(" ")
            features.append(list(map(float, feature)))
    features = np.array(features)
    print("features shape: %s" % str(features.shape))
    return features


def read_label(label_file):
    with open(label_file, "r") as f:
        labels = []
        for line in f:
            label = int(line.rstrip()) - 1     # to 0-index
            labels.append(label)
    print("labels shape: %s" % str(len(labels)))
    return np.array(labels)


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    args = ArgParse().parse_args()
    np.random.seed(args.seed)
    features = read_feature(args.feature_file)
    labels = read_label(args.label_file)
    pca = PCA(n_components=args.pca_n_components)
    features = pca.fit_transform(features)
    features, labels = data_crop(
        features=features,
        labels=labels,
        label_dim=args.label_dim,
        n_sample_p_label=args.n_sample_p_label,
        random_shuffle=args.random_shuffle
    )
    data = mutliclass2multitask(
        features=features,
        labels=labels,
        n_pos=args.n_pos,
        n_neg_l=args.n_neg_l,
        label_dim=args.label_dim,
        random_shuffle=args.random_shuffle,
        verbose=False
    )
    write2file(
        data=data,
        meta_data=None,
        dir_name=args.dir_name,
    )
