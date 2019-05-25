import numpy as np

from experiment import DataGeneratorFull


def distribution(data_dir):
    data_loader = DataGeneratorFull(
        label_file=data_dir + "label",
        feature_file=data_dir + "feature",
        task_file=data_dir + "id",
    )
    for d in data_loader.generate(
        batch_size=data_loader.data_size,
        random_shuffle=False
    ):
        task_label_dist = np.einsum(
            "ij,ik->jk",
            d["task"],
            d["label"]
        )
        return task_label_dist / np.sum(task_label_dist, axis=-1, keepdims=True)


if __name__ == "__main__":
    task_label_distribution = distribution(data_dir="../data/synthetic_topic_task_sparse_v2/")
    for task in range(task_label_distribution.shape[0]):
        print("task_%02d: %s" % (task, str(task_label_distribution[task])))
