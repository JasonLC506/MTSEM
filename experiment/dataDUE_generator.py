import _pickle as cPickle
import os
from datetime import datetime
import numpy as np
import warnings
import math


PADDLE_INDEX = -1


class DataDUELoader(object):
    def __init__(
            self,
            meta_data_file,
            batch_data_dir=None,
            id_map=None,
            dataToken=None,
            max_doc_length=30,
            paddle_index=PADDLE_INDEX,
            no_doc_index=False
    ):
        with open(meta_data_file, "rb") as f:
            meta_data = cPickle.load(f)
            self.E = meta_data["E"]
            self.U = meta_data["U"]
            self.D = meta_data["D"]
            self.Nd = meta_data["Nd"]
            self.V = meta_data["V"]

        self.id_map = id_map
        self.batch_data_dir = batch_data_dir
        self.dataToken = dataToken
        self.max_doc_length = max_doc_length
        self.paddle_index = paddle_index if paddle_index > 0 else self.V + paddle_index + 1
        self.data = dict()
        self.data_size = self.n_reactions = 0
        self.no_doc_index = no_doc_index
        self._data_read()

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
        print("max_batch: %d" % max_batch)
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(self.data_size, (i_batch + 1) * batch_size)]
            reactions_batch = self.data["reactions"][batch_index]
            feature_indexs, labels, task_ids, weights = self._data_process_before_generate(reactions=reactions_batch)
            yield {
                "feature_index": feature_indexs,
                "label": labels,
                "weight": weights,
                "task": task_ids
            }

    def _data_read(self):
        self._read_reactions()
        self._read_text()

    def _read_reactions(self):
        reactions = []
        file_list = os.listdir(self.batch_data_dir)
        start = datetime.now()
        for fn in file_list:
            with open(self.batch_data_dir + fn, "rb") as f:
                posts = cPickle.load(f)

            for post_id in posts:
                if post_id not in self.id_map:
                    continue
                document_id = self.id_map[post_id]
                users, emots = posts[post_id]
                for i in range(len(users)):
                    reactions.append(
                        [
                            int(document_id),
                            int(users[i]),
                            int(emots[i])
                        ]
                    )
        duration = datetime.now() - start
        print("it takes %f sec for read batch reaction files" % duration.total_seconds())
        print("total reactions: %d" % len(reactions))
        self.data["reactions"] = np.array(reactions, dtype=np.int64)
        self.data_size = self.n_reactions = len(reactions)

    def _read_text(self):
        """
        paddle and truncate document
        """
        texts = []
        paddle = [self.paddle_index for _ in range(self.max_doc_length)]
        for i in range(len(self.dataToken)):
            new_text = self.dataToken[i] + paddle
            texts.append(new_text[:self.max_doc_length])
        print("total documents: %d" % len(texts))
        self.data["texts"] = np.array(texts, dtype=np.int64)

    def _data_process_before_generate(
            self,
            reactions
    ):
        # document_id #
        dids = reactions[:, 0]
        # users #
        users = reactions[:, 1]
        # emoticons #
        emoticon_eye = np.eye(self.E, dtype=np.float32)
        emots = emoticon_eye[reactions[:, 2]]         # one-hot
        # text #
        texts = self.data["texts"][dids]   # lookup text by document_id

        weights = np.ones(dids.shape[0], dtype=np.float32)

        return texts, emots, users, weights


if __name__ == "__main__":
    data_dir = "C:/Users/jpz5181/Documents/GitHub/PSEM/data/CNN_foxnews/"
    data_prefix = "_CNN_foxnews_combined_K10"

    id_map_file = data_dir + "id_map" + data_prefix
    postcontent_dataW_file = data_dir + "dataW" + data_prefix
    postcontent_dataToken_file = data_dir + "dataToken" + data_prefix
    word_dictionary_file = data_dir + "word_dictionary" + data_prefix

    id_map, id_map_reverse = cPickle.load(open(id_map_file, "rb"))
    # dataW = cPickle.load(open(postcontent_dataW_file, "rb"), encoding='bytes')
    # print(dataW.nnz)
    dataToken = cPickle.load(open(postcontent_dataToken_file, "rb"))
    word_dictionary = cPickle.load(open(word_dictionary_file, "rb"))
    # print(word_dictionary)

    # data_name = sys.argv[1]
    data_name = "CNN_nolike"
    data_dir = "C:/Users/jpz5181/Documents/GitHub/PSEM/data/" + data_name + "/"

    batch_rBp_dir = data_dir + "train/"
    batch_valid_on_shell_dir = data_dir + "on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "on_shell/test/"
    batch_test_off_shell_dir = data_dir + "off_shell/test/"

    meta_data_train_file = data_dir + "meta_data_train"
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid"
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test"
    meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid"
    meta_data_on_test_file = data_dir + "meta_data_on_shell_test"

    data_loader = DataDUELoader(
        meta_data_file=meta_data_train_file,
        batch_data_dir=batch_rBp_dir,
        id_map=id_map_reverse,
        dataToken=dataToken
    )
    for data_batched in data_loader.generate(batch_size=2):
        print(data_batched)
        break
