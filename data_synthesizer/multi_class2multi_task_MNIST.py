"""
build multi-task MTL-MNIST data based on
@article{rosenbaum2017routing,
  title={Routing networks: Adaptive selection of non-linear functions for multi-task learning},
  author={Rosenbaum, Clemens and Klinger, Tim and Riemer, Matthew},
  journal={ICLR},
  year={2018}
}
"""
import argparse
import numpy as np
import warnings

from data_synthesizer import write2file


def mutliclass2multitask(
        features,
        labels,
        n_pos,
        n_neg_l,
        label_dim=10,
        random_shuffle=True,
        verbose=True
):
    """
    features.shape[0] == labels.shape[0]
    labels in index format, int

    the data format is to be read by data_generator
    :param features:
    :param labels:
    :param n_pos: # positive instances per task
    :param n_neg_l: # negative instances from the other labels per task
    :param label_dim: dimension of label, same to the number of tasks
    :return:
    """
    data = {
        "feature": [],
        "label": [],
        "id": []
    }
    duplicate_instance = False     # if data is enough, all data instances are distinct
    label_instance_ind = []
    for label in range(label_dim):
        instance_ind = np.argwhere(labels == label).squeeze()
        if random_shuffle:
            np.random.shuffle(instance_ind)
        label_size = instance_ind.shape[0]
        if n_pos > label_size:
            raise ValueError("n_pos is too large compared with label %d: %d>%d" % (label, n_pos, label_size))
        if n_pos + n_neg_l > label_size:
            warnings.warn("not enough negative sample for even single task per label")
        label_instance_ind.append(
            instance_ind[:n_pos + (label_dim - 1) * n_neg_l]
        )
        if n_pos + (label_dim - 1) * n_neg_l > label_size:
            duplicate_instance = True
            warnings.warn("duplicate_instance is used because not enough instances")
    task_instance_ind = []
    for task in range(label_dim):
        pos_instance_ind = label_instance_ind[task][:n_pos]
        neg_instance_ind = []
        for label in range(label_dim):
            if label == task:
                continue
            if not duplicate_instance:
                relative_id = (task - label + label_dim) % label_dim - 1
                neg_instance_ind.append(
                    label_instance_ind[label][
                        n_pos + relative_id * n_neg_l: n_pos + (relative_id + 1) * n_neg_l
                    ]
                )
            else:
                neg_instance_label = label_instance_ind[label][n_pos:]
                np.random.shuffle(neg_instance_label)
                neg_instance_ind.append(
                    neg_instance_label[:n_neg_l]
                )
        neg_instance_ind = np.concatenate(
            neg_instance_ind,
            axis=0
        )
        task_instance_ind.append(
            [
                pos_instance_ind,
                neg_instance_ind
            ]
        )
    if verbose:
        print(task_instance_ind)
    one_hot_label = np.eye(2, dtype=np.float32)
    data_size_bynow = 0
    for task in range(label_dim):
        for pn in range(2):
            # pn=0: neg #
            # print("ind shape: %s" % str(task_instance_ind[task][1 - pn].shape))
            feature = features[task_instance_ind[task][1 - pn]]
            # print("feature_shape: %s" % str(feature.shape))
            label = np.tile(
                one_hot_label[pn],
                reps=(feature.shape[0], 1)
            )
            for loc_ind in range(feature.shape[0]):
                data["feature"].append(feature[loc_ind])
                data["label"].append(label[loc_ind])
                data["id"].append(
                    "t%d_%d" % (task, loc_ind + data_size_bynow)
                )
            data_size_bynow += feature.shape[0]
    return data


def data_crop(
        features,
        labels,
        label_dim,
        n_sample_p_label,
        random_shuffle=True
):
    label_instance_ind = []
    for label in range(label_dim):
        instance_ind = np.argwhere(labels == label).squeeze()
        if random_shuffle:
            np.random.shuffle(instance_ind)
        label_instance_ind.append(instance_ind[:n_sample_p_label])
    label_instance_ind = np.concatenate(label_instance_ind)
    features_cropped = features[label_instance_ind]
    labels_cropped = labels[label_instance_ind]
    return features_cropped, labels_cropped


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dd", "--data_dir", default="C:/Users/jpz5181/Documents/GitHub/MTSEM/data/MNIST")
        parser.add_argument("-pnc", "--pca_n_components", default=64, type=int)
        parser.add_argument("-nspl", "--n_sample_p_label", default=200, type=int)
        parser.add_argument("-np", "--n_pos", default=20, type=int)
        parser.add_argument("-nnl", "--n_neg_l", default=20, type=int)
        parser.add_argument("-ld", "--label_dim", default=10, type=int)
        parser.add_argument("-rs", '--random_shuffle', default=True, action="store_false")
        parser.add_argument("-s", "--seed", default=2019, type=int)
        parser.add_argument("-dn", '--dir_name', type=str, default="../data/MNIST_MTL_imba")
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    from mnist import MNIST
    from sklearn.decomposition import PCA
    args = ArgParse().parse_args()
    np.random.seed(args.seed)
    mndata = MNIST(args.data_dir)
    images, labels = mndata.load_training()
    images = np.array(images, dtype=np.float32)
    pca = PCA(n_components=args.pca_n_components)
    images = pca.fit_transform(images)
    labels = np.array(labels, dtype=np.int8)
    images, labels = data_crop(
        features=images,
        labels=labels,
        label_dim=args.label_dim,
        n_sample_p_label=args.n_sample_p_label,
        random_shuffle=args.random_shuffle
    )
    data = mutliclass2multitask(
        features=images,
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
