import numpy as np
import math
import warnings
from datetime import datetime
import _pickle as cPickle
import os


EPSILON = 0.1


class DataGeneratorBase(object):
    def __init__(self, **kwargs):
        self.features = None
        self.labels = None
        self.weights = None
        self.tasks = None
        self.data_size = 0
        self.tasks_size = 0

    def generate(
            self,
            batch_size=1,
            random_shuffle=True
    ):
        index = np.arange(self.data_size)
        if random_shuffle:
            np.random.shuffle(index)
        if batch_size > self.data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, self.data_size))
            batch_size = self.data_size
        max_batch = int(math.ceil(float(self.data_size) / float(batch_size)))
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(self.data_size, (i_batch + 1) * batch_size)]
            data_batch = {
                "feature": self.features[batch_index],
                "label": self.labels[batch_index],
                "weight": self.weights[batch_index],
                "task": self.tasks[batch_index],
                "instance_index": batch_index
            }
            yield data_batch

    @property
    def feature_dim(self):
        return self.features.shape[-1]

    @property
    def label_dim(self):
        return self.labels.shape[-1]

    @property
    def task_dim(self):
        return self.tasks_size


class DataGeneratorFull(DataGeneratorBase):
    """
    load all data and then generate (in case data size fitting in memory)
    """
    def __init__(
            self,
            feature_file,
            label_file,
            weight_file=None,
            task_file=None,
            topic_file=None,
            weight_task=True
    ):
        super(DataGeneratorFull, self).__init__()
        self.weight_task = weight_task
        self._initialize(
            feature_file=feature_file,
            label_file=label_file,
            weight_file=weight_file,
            task_file=task_file,
            topic_file=topic_file
        )

    def _initialize(
            self,
            feature_file,
            label_file,
            weight_file=None,
            task_file=None,
            topic_file=None
    ):
        self.features = self._read_feature(feature_file)
        self.labels = self._read_label(label_file)
        assert self.features.shape[0] == self.labels.shape[0]
        self.data_size = self.labels.shape[0]
        self.tasks = self._read_task(task_file=task_file, topic_file=topic_file)
        assert self.tasks.shape[0] == self.data_size
        self.weights = self._read_weight(weight_file)
        assert self.weights.shape[0] == self.data_size

    def _read_label(
            self,
            label_file,
            separator=",",
            normalize=True,
            epsilon=EPSILON
    ):
        labels = []
        with open(label_file, "r") as ff:
            for line in ff:
                label = line.rstrip().split(separator)
                label = list(map(float, label))
                labels.append(label)
        if len(labels) > 0:
            print("dimension of input: [%d, %d]" % (len(labels), len(labels[0])))
        labels = np.array(labels, dtype=np.float32)
        if normalize:
            labels = labels + epsilon
            labels = labels / np.sum(labels, keepdims=True, axis=-1)
        return labels

    def _read_feature(
            self,
            feature_file,
            separator=","
    ):
        return self._read_label(
            label_file=feature_file,
            separator=separator,
            normalize=False
        )

    def _read_weight(
            self,
            weight_file
    ):
        if weight_file is None:
            weights = np.ones(self.data_size, dtype=np.float32)
            if self.weight_task:
                weights = self._task_reweight(weights=weights, tasks=self.tasks)
        else:
            weights = []
            with open(weight_file, "r") as wf:
                for line in wf:
                    weight = float(line.rstrip())
                    weights.append(weight)
            weights = np.array(weights, dtype=np.float32)
        return weights

    def _task_reweight(
            self,
            weights,
            tasks
    ):
        """
        :param weights: original weight matrix
        :param tasks: tasks matrix in one-hot encoding with float dtype
        :return: new weights
        """
        task_weights = np.tensordot(
            weights,
            tasks,
            axes=[0, 0]
        )
        print("task_weights sum before reweight: %s" % str(task_weights))
        assert np.all(task_weights > 0.5)
        mean_weights = np.mean(task_weights, keepdims=True)
        task_weights = mean_weights / task_weights
        print("task_weights for each task: %s" % str(task_weights))
        weights_new = np.tensordot(
            tasks,
            task_weights,
            axes=[-1, -1]
        )
        return weights_new

    def _read_task(
            self,
            task_file,
            topic_file=None
    ):
        if task_file is None:
            tasks = np.zeros(self.data_size, dtype=np.int32)
            self.tasks_size = 1
            # self.tasks_dictionary = {None: 0}
        else:
            # topic file #
            if topic_file:
                topics = cPickle.load(open(topic_file, 'rb'))
            else:
                topics = None
            tasks = []
            with open(task_file, 'r') as tf:
                for line in tf:
                    task = self._task_id_extract(
                        instance_id=line.rstrip(),
                        topics=topics
                    )
                    tasks.append(task)
            tasks_list = sorted(list(set(tasks)))
            tasks_dictionary = {}
            for i in range(len(tasks_list)):
                tasks_dictionary[tasks_list[i]] = i
            task_dictionary_file = task_file + "_dictionary"
            if os.path.exists(task_dictionary_file):
                tasks_dictionary_old = cPickle.load(open(task_dictionary_file, 'rb'))
                if not dictionary_equal(tasks_dictionary_old, tasks_dictionary):
                    print("old task dictionary: %s" % str(tasks_dictionary_old))
                    print("task dictionary: %s" % str(tasks_dictionary))
                    warnings.warn(
                        "dictionary updated, use the old one in %s" % task_dictionary_file
                    )
                    tasks_dictionary = tasks_dictionary_old
            else:
                with open(task_file + "_dictionary", 'wb') as tdf:
                    cPickle.dump(tasks_dictionary, tdf)
            print("task dictionary: %s" % str(tasks_dictionary))
            tasks = list(map(lambda x: tasks_dictionary[x], tasks))
            tasks = np.array(tasks, dtype=np.int32)
            self.tasks_size = len(tasks_list)
        # one-hot encoding #
        identity = np.eye(self.tasks_size, dtype=np.float32)
        tasks_one_hot = identity[tasks]
        return tasks_one_hot

    @staticmethod
    def _task_id_extract(instance_id, topics=None):
        """ data dependent: extract task_id from instance id """
        # add topic as prefix if exist #
        task_id = instance_id.split("_")[0]
        if topics:
            topic = topics[instance_id]
            topic = np.argmax(topic).squeeze()
            return "%d_%s" % (topic, task_id)
        return task_id


class DataGeneratorTrainTest(DataGeneratorFull):
    def __init__(
            self,
            feature_file,
            label_file,
            weight_file=None,
            task_file=None,
            topic_file=None,
            train_ratio=0.7,
            valid_ratio=0,
            stage_wise=True
    ):
        super(DataGeneratorTrainTest, self).__init__(
            feature_file=feature_file,
            label_file=label_file,
            weight_file=weight_file,
            task_file=task_file,
            topic_file=topic_file
        )
        self.train, self.valid, self.test = self.train_test_split(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            stage_wise=stage_wise
        )

    def train_test_split(
            self,
            train_ratio,
            valid_ratio=0,
            stage_wise=True
    ):
        train_index, valid_index, test_index = self._train_test_index_split(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            stage_wise=stage_wise
        )
        train_generator = self._data_assign(train_index)
        valid_generator = self._data_assign(valid_index)
        test_generator = self._data_assign(test_index)
        return train_generator, valid_generator, test_generator

    def _train_test_index_split(
            self,
            train_ratio,
            valid_ratio=0,
            stage_wise=True
    ):
        assert train_ratio + valid_ratio <= 1.0
        index = np.arange(self.data_size)
        np.random.shuffle(index)
        if not stage_wise:
            train_index, valid_index, test_index = self._train_test_index_split_basic(
                index=index,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio
            )
        else:
            train_indices, valid_indices, test_indices = [], [], []
            for task_id in range(self.tasks_size):
                train_index_tmp, valid_index_tmp, test_index_tmp = self._train_test_index_split_basic(
                    index=index[np.argwhere(self.tasks[index, task_id] > 0.5)].squeeze(),
                    train_ratio=train_ratio,
                    valid_ratio=valid_ratio
                )
                train_indices.append(train_index_tmp)
                valid_indices.append(valid_index_tmp)
                test_indices.append(test_index_tmp)
            train_index = np.concatenate(train_indices, axis=0)
            valid_index = np.concatenate(valid_indices, axis=0)
            test_index = np.concatenate(test_indices, axis=0)

        return train_index, valid_index, test_index

    def _train_test_index_split_basic(
            self,
            index,
            train_ratio,
            valid_ratio=0
    ):
        data_size = index.shape[0]
        train_num = int(train_ratio * data_size)
        valid_num = int(valid_ratio * data_size)
        test_num = data_size - train_num - valid_num
        train_index = index[:train_num]
        valid_index = index[train_num: train_num + valid_num]
        test_index = index[train_num + valid_num:]
        return train_index, valid_index, test_index

    def _data_assign(
            self,
            index
    ):
        generator = DataGeneratorBase()
        generator.features = self.features[index]
        generator.labels = self.labels[index]
        generator.weights = self.weights[index]
        generator.tasks = self.tasks[index]
        generator.data_size = len(index)
        generator.tasks_size = self.tasks_size
        return generator


def dictionary_equal(d1, d2):
    if len(d1) != len(d2):
        return False
    for k in d1:
        if k not in d2:
            return False
        if d1[k] != d2[k]:
            return False
    return True
