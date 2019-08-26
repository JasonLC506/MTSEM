from scipy.io import loadmat
import numpy as np
from data_synthesizer.topic_task_sparse_v2 import write2file


def preprocess(filename, check=False):
    data_raw = loadmat(filename)
    task_indexes_raw = np.squeeze(data_raw['task_indexes'])
    features = np.transpose(data_raw['x'])
    ys = np.squeeze(data_raw['y'])
    if check:
        print(np.histogram(ys))
        return None
    else:
        labels = y2label(ys)
    task_dim = task_indexes_raw.shape[0]
    ids = []
    task_indexes = task_indexes_raw - 1.1
    j = 0
    for i in range(ys.shape[0]):
        if j < task_dim - 1 and i >= task_indexes[j + 1]:
            j += 1
        print("%d, %d, %d" % (i, j, task_indexes[j]))
        t = j
        ids.append("%d_%d" % (t, i))
    return {
        "feature": features,
        "label": labels,
        "id": ids
    }


def y2label(ys, split=(10, 20, 71)):
    labels = []
    for i in range(ys.shape[0]):
        y = ys[i]
        flag = False
        for j in range(len(split)):
            if y < split[j]:
                labels.append(j)
                flag = True
                break
        if not flag:
            raise ValueError("y value %f cannot be found in %s" % (y, str(split)))

    cat_dim = len(split)
    labels_onehot = np.zeros([len(labels), cat_dim])
    labels_onehot[np.arange(len(labels)), np.array(labels)] = 1.0
    return labels_onehot


if __name__ == "__main__":
    data = preprocess("../data/school_splits/school_b.mat")
    write2file(
        data=data,
        meta_data=None,
        dir_name="../data/school"
    )

