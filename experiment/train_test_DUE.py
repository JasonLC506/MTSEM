import argparse
import os
import _pickle as cPickle
import numpy as np

from experiment import DataDUELoader, json_reader, evaluate_DUE
from models import MtLinAdapt
from common.readlogboard import read


RANDOM_SEED_NP = 2019


def train(
        config_file,
        meta_data_file,
        id_map,
        dataToken,
        batch_data_dir_train,
        batch_data_dir_valid=None,
        max_doc_length=30,
        model_name=None,
        restore_path=None,
        no_doc_index=False,
        feature_group_file="../ckpt/feature_group",
        w0_file="../ckpt/w0"
):
    np.random.seed(RANDOM_SEED_NP)
    data_train = DataDUELoader(
        meta_data_file=meta_data_file,
        batch_data_dir=batch_data_dir_train,
        id_map=id_map,
        dataToken=dataToken,
        max_doc_length=max_doc_length,
        no_doc_index=no_doc_index
    )
    if batch_data_dir_valid is not None:
        data_valid = DataDUELoader(
            meta_data_file=meta_data_file,
            batch_data_dir=batch_data_dir_valid,
            id_map=id_map,
            dataToken=dataToken,
            max_doc_length=max_doc_length,
            no_doc_index=no_doc_index
        )
    else:
        data_valid = None

    model_spec = json_reader(config_file)
    model = MtLinAdapt(
        feature_shape=data_train.V + 1,
        feature_dim=max_doc_length,
        label_dim=data_train.E,
        task_dim=data_train.U,
        model_spec=model_spec,
        model_name=model_name
    )
    model.initialization(
        feature_group_file=feature_group_file,
        w0_file=w0_file
    )
    if restore_path is not None:
        model.restore(restore_path)

    # train #
    results = model.train(
        data_generator=data_train,
        data_generator_valid=data_valid
    )
    print("train_results: %s" % str(results))

    best_epoch = read(
        directory="../summary/" + model.model_name,
        main_indicator="epoch_losses_valid_00"
    )[0]
    print("best_epoch by validation loss: %d" % best_epoch)


def test(
        config_file,
        meta_data_file,
        id_map,
        dataToken,
        batch_data_dir,
        max_doc_length=30,
        model_name=None,
        restore_path=None,
        no_doc_index=False
):
    np.random.seed(RANDOM_SEED_NP)
    data = DataDUELoader(
        meta_data_file=meta_data_file,
        batch_data_dir=batch_data_dir,
        id_map=id_map,
        dataToken=dataToken,
        max_doc_length=max_doc_length,
        no_doc_index=no_doc_index
    )

    model_spec = json_reader(config_file)
    model = MtLinAdapt(
        feature_shape=data.V + 1,
        feature_dim=max_doc_length,
        label_dim=data.E,
        task_dim=data.U,
        model_spec=model_spec,
        model_name=model_name
    )
    model.initialization()

    def performance(
            model_local,
            data_local
    ):
        preds = model_local.predict(
            data_generator=data_local
        )
        labels = []
        for data_batched in data_local.generate(
                batch_size=model_spec["batch_size"],
                random_shuffle=False
        ):
            labels.append(data_batched["label"])
        labels = np.concatenate(labels, axis=0)
        # one-hot to index #
        trues = np.argmax(labels, axis=-1)

        perf = evaluate_DUE(
            preds=preds,
            trues=trues
        )
        return perf

    if restore_path is not None:
        if not isinstance(restore_path, list):
            restore_paths = [restore_path]
        else:
            restore_paths = restore_path
        for restore_path in restore_paths:
            model.restore(restore_path)
            perf = performance(
                model_local=model,
                data_local=data
            )
            print("ckpt_path: %s" % restore_path)
            print("performance: %s" % str(perf))
    else:
        perf = performance(
            model_local=model,
            data_local=data
        )
        print("random initialization")
        print("performance: %s" % str(perf))


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dm", "--data_name", default="CNN_nolike")
        parser.add_argument("-mdl", "--max_doc_length", default=30, type=int)
        parser.add_argument("-cf", "--config_file", default="../models/mt_lin_adapt_config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-mn", "--model_name", default=None)
        parser.add_argument("-di", "--doc_index", default=False, action="store_true")
        parser.add_argument("-fgf", "--feature_group_file", default="../ckpt/CNN_nolike/feature_group_CNN_nolike")
        parser.add_argument("-wf", "--w0_file", default="../ckpt/CNN_nolike/param_pretrained_CNN_nolike")
        parser.add_argument("-vt", "--valid_test", default="t", choices=["v", "t"])
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()

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

    data_name = args.data_name
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

    train(
        config_file=args.config_file,
        meta_data_file=meta_data_train_file,
        id_map=id_map_reverse,
        dataToken=dataToken,
        batch_data_dir_train=batch_rBp_dir,
        batch_data_dir_valid=batch_valid_on_shell_dir,
        max_doc_length=args.max_doc_length,
        model_name=args.data_name if args.model_name is None else args.model_name + "_" + args.data_name,
        restore_path=args.restore_path,
        no_doc_index=not args.doc_index,
        feature_group_file=args.feature_group_file,
        w0_file=args.w0_file
    )

    # if args.restore_path is not None:
    #     if os.path.isdir(args.restore_path):
    #         file_names = list(os.listdir(args.restore_path))
    #         del file_names[file_names.index("checkpoint")]
    #         restore_paths = list(map(lambda x: x[:9], file_names))      # 9: len("epoch_%03d")
    #         restore_paths = sorted(list(set(restore_paths)))
    #         restore_paths = list(map(lambda x: os.path.join(args.restore_path, x), restore_paths))
    #     else:
    #         restore_paths = [args.restore_path]
    # else:
    #     restore_paths = None
    #
    # test(
    #     config_file=args.config_file,
    #     meta_data_file=meta_data_train_file,
    #     id_map=id_map_reverse,
    #     dataToken=dataToken,
    #     batch_data_dir=batch_valid_on_shell_dir if args.valid_test == "v" else batch_test_on_shell_dir,
    #     max_doc_length=args.max_doc_length,
    #     model_name=args.data_name if args.model_name is None else args.model_name + "_" + args.data_name,
    #     restore_path=restore_paths,
    #     no_doc_index=not args.doc_index
    # )
