import argparse
import numpy as np

from experiment import DataGeneratorTrainTest, DataGeneratorFull, simple_evaluate
from models import Models
from experiment import json_reader, HyperparameterTuner, dict_conservative_update, dict_update
from common.readlogboard import read


RANDOM_SEED_NP = 2019


def run_single(
        feature_file,
        label_file,
        weight_file=None,
        task_file=None,
        train_pattern="",
        valid_pattern="",
        Model=Models['fc'],
        model_spec={},
        restore_path=None,
        partial_restore_paths=None,
        model_name=None,
        full_split_saver=False
):
    """
    using previously split data for experiment
    :param feature_file:
    :param label_file:
    :param weight_file:
    :param task_file:
    :param train_pattern: pattern to distinguish training files
    :param valid_pattern: pattern to distinguish valid files
    :param Model:
    :param config_file:
    :param restore_path:
    :param partial_restore_paths: dictionary of paths for each partial saver
    :param model_name:
    :param full_split_saver:
    :return:
    """
    np.random.seed(RANDOM_SEED_NP)

    # initialize data #
    datas = []
    for data_file_pattern in [train_pattern, valid_pattern]:
        datas.append(
            DataGeneratorFull(
                feature_file=feature_file + data_file_pattern,
                label_file=label_file + data_file_pattern,
                weight_file=weight_file if weight_file is None else weight_file + data_file_pattern,
                task_file=task_file if task_file is None else task_file + data_file_pattern
            )
        )
    data_train, data_valid = datas

    # initialize model #

    def init_model(data):
        model = Model(
            feature_dim=data.feature_dim,
            label_dim=data.label_dim,
            task_dim=data.task_dim,
            model_spec=model_spec,
            model_name=model_name
        )
        model.initialization()
        return model

    model = init_model(data=data_train)
    if partial_restore_paths is not None:
        model.partial_restore(**partial_restore_paths)
    if restore_path is not None:
        model.restore(restore_path)

    # train #
    train_final_results = model.train(
        data_generator=data_train,
        data_generator_valid=data_valid,
        full_split_saver=full_split_saver
    )
    print("training output: %s" % str(train_final_results))

    best_epoch = read(
        directory="../summary/" + model.model_name,
        main_indicator="epoch_losses_valid_00"
    )[0]

    # test #
    data_test = data_valid

    model = init_model(data=data_test)
    model.restore(
        save_path="../ckpt/" + model.model_name + "/epoch_%03d" % int(best_epoch)       # dependent on save path
    )
    results = simple_evaluate(
        model=model,
        data=data_test
    )
    return {
        "training_final": train_final_results,
        "best_epoch": best_epoch,
        "valid_best": results
    }


def result2perf(result):
    """
    process returned result to performance for tuning
    ! highly dependent on output format and info to extract
    :param result: {
                        "training_final":
                        "best_epoch":
                        "valid_best":
    }
    :return:
    """
    perf_primary = result["valid_best"][0][0][0]
    perf_train_final = result["training_final"][0][0][0]
    perf_additionals = []
    best_epoch = result["best_epoch"]
    perf = [perf_primary, perf_train_final] + perf_additionals + [best_epoch]
    return perf


def run_ht(
        ht_config_file,
        ht_config_file_additional,
        ht_save_path,
        ht_start_id=None,
        ht_end_id=None,
        find_best=False,
        **kwargs
):
    ht_config = json_reader(ht_config_file)
    if ht_config_file_additional is not None:
        ht_config_additional = json_reader(ht_config_file_additional)
        ht_config = dict_update(
            dict_base=ht_config,
            dict_new=ht_config_additional
        )
    model_spec_config_base = json_reader(kwargs["config_file"])

    ht = HyperparameterTuner(
        save_path=ht_save_path
    )
    ht_config["model_spec"] = dict_conservative_update(
        dict_base=model_spec_config_base,
        dict_new=ht_config["model_spec"]
    )       # update model spec for different models to avoid redundant hps
    print(ht_config)                             # verbose
    ht.initialize(hps=ht_config)

    # remove "config_file" for safety, add "model_spec" to be updated by single hp_config #
    del kwargs["config_file"]
    kwargs["model_spec"] = model_spec_config_base
    for test_id, hp_config in ht.generate(
        start_id=ht_start_id,
        end_id=ht_end_id
    ):
        kwargs_single = dict_conservative_update(
            dict_base=kwargs,
            dict_new=hp_config
        )
        print("run %d with config: \n%s" % (test_id, str(kwargs_single)))
        result_single = run_single(
            **kwargs_single
        )
        perf = result2perf(result=result_single)
        ht.read_perf(id_=test_id, perf=perf)
    if find_best:
        print(ht.find_best())


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dd", "--data_dir", default="../data/MNIST_MTL/")
        parser.add_argument('-md', "--model_dir", default="../models/")

        # parser.add_argument("-ff", "--feature_file", default="posts_content_all_features_[CLS]_SUM_joined_sampled_20")
        parser.add_argument("-ff", "--feature_file", default="feature_remained_80")
        # parser.add_argument("-lf", "--label_file", default="posts_reactions_all_joined_sampled_20")
        parser.add_argument("-lf", "--label_file", default="label_remained_80")
        parser.add_argument("-wf", "--weight_file", default=None)
        # parser.add_argument("-tf", "--task_file", default="posts_content_all_text_ids_joined_sampled_20")
        parser.add_argument("-tf", "--task_file", default="id_remained_80")
        parser.add_argument("-tp", "--train_pattern", default="_remained_80", type=str)
        parser.add_argument("-vp", "--valid_pattern", default="_sampled_20", type=str)
        parser.add_argument("-M", "--Model", default="shared_bottom")
        parser.add_argument("-cf", "--config_file", default="config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-rpb", "--restore_path_bottom", default=None)
        parser.add_argument("-rpt", "--restore_path_top", default=None)
        parser.add_argument("-rpr", "--restore_path_regularization", default=None)
        parser.add_argument("-fss", "--full_split_saver", default=False, action="store_true")
        parser.add_argument("-mn", "--model_name", default="ht/MNIST_MTL/shared_bottom")
        parser.add_argument("-hcf", "--ht_config_file", default="../experiment/ht_configs/MNIST_MTL/ht_config.json")
        parser.add_argument("-hcfa", "--ht_config_file_additional", default=None, type=str)
        parser.add_argument("-hsp", "--ht_save_path", default="../ht_log/MNIST_MTL/shared_bottom/", type=str)
        parser.add_argument("-hsi", "--ht_start_id", default=5, type=int)
        parser.add_argument("-hei", "--ht_end_id", default=10, type=int)
        parser.add_argument("-fb", "--find_best", default=False, action="store_true")
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()
    print("input: %s" % args.__dict__)
    run_ht(
        ht_config_file=args.ht_config_file,
        ht_config_file_additional=args.ht_config_file_additional,
        ht_save_path=args.ht_save_path,
        ht_start_id=args.ht_start_id,
        ht_end_id=args.ht_end_id,
        find_best=args.find_best,
        feature_file=args.data_dir + args.feature_file,
        label_file=args.data_dir + args.label_file,
        weight_file=None if args.weight_file is None else args.data_dir + args.weight_file,
        task_file=None if args.task_file is None else args.data_dir + args.task_file,
        train_pattern=args.train_pattern,
        valid_pattern=args.valid_pattern,
        Model=Models[args.Model],
        config_file=args.model_dir + args.Model + "_" + args.config_file,
        restore_path=args.restore_path,
        partial_restore_paths={
            "save_path_bottom": args.restore_path_bottom,
            "save_path_task_specific_top": args.restore_path_top,
            "save_path_regularization": args.restore_path_regularization
        },
        model_name=args.model_name + ("" if args.ht_start_id is None else str(args.ht_start_id)),
        full_split_saver=args.full_split_saver
    )
