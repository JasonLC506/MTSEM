import argparse
import numpy as np
import copy

from experiment import DataGeneratorTrainTest, DataGeneratorFull, simple_evaluate
from models import Models
from experiment import json_reader, HyperparameterTuner, dict_conservative_update, dict_update, train_model, init_model


RANDOM_SEED_NP = 2019


def run_single(
        feature_file,
        label_file,
        weight_file=None,
        task_file=None,
        train_pattern="",
        valid_pattern="",
        **model_kwargs_wrap
):
    """
    using previously split data for experiment
    :param feature_file:
    :param label_file:
    :param weight_file:
    :param task_file:
    :param train_pattern: pattern to distinguish training files
    :param valid_pattern: pattern to distinguish valid files
    :param model_kwargs_wrap: {
        "model_kwargs": {
            Model:
            config_file:
            model_name:
            partial_restore_paths:
            restore_path:
            full_split_saver:
        }
        "model_primary_kwargs": {
        }
    }
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
    model_kwargs = model_kwargs_wrap["model_kwargs"]
    model_primary_kwargs = model_kwargs_wrap["model_primary_kwargs"]

    model_spec = model_kwargs["model_spec"]
    if model_primary_kwargs is not None:
        # train primary model to initialize model #
        model_spec_primary_ = json_reader(model_primary_kwargs['config_file'])
        model_spec_primary = dict_conservative_update(
            dict_base=model_spec_primary_,
            dict_new=model_spec
        )  # make sure the shared hps are consistent
        model_spec_primary["optim_params"] = model_spec_primary_["optim_params"]
        model_spec_primary["max_epoch"] = model_spec_primary_["max_epoch"]
        [best_epoch_primary, results_], model_primary_save_path, _m = train_model(
            data_train=data_train,
            data_valid=data_valid,
            model_spec=model_spec_primary,
            **model_primary_kwargs
        )
        # for models use parts of primary model (e.g., cross_stitch)
        for path_name in model_kwargs["partial_restore_paths"]:
            name_pattern = model_kwargs["partial_restore_paths"][path_name]
            if name_pattern is None:
                partial_restore_path = None
            else:
                partial_restore_path = name_pattern.format(best_epoch_primary)
            model_kwargs["partial_restore_paths"][path_name] = partial_restore_path
        # for models use full of primary model for graph definition (e.g., dmtrl_Tucker)
        if "primary_model_ckpt" in model_spec:
            model_spec["primary_model_ckpt"] = model_primary_save_path.format(best_epoch_primary)

    [best_epoch, train_final_results], model_best_save_path, model = train_model(
        data_train=data_train,
        data_valid=data_valid,
        **model_kwargs
    )
    # test #
    data_test = data_valid

    model = init_model(
        data=data_test,
        Model=model_kwargs["Model"],
        model_spec=model_spec,
        model_name=model_kwargs["model_name"]
    )
    model.restore(
        save_path=model_best_save_path.format(int(best_epoch))
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
    perf_primary = result["valid_best"]["perf"]
    perf_train_final = result["training_final"][0][0][0]
    perf_additionals = [result["valid_best"]["results"][0][0][0]]
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
    model_spec_config_base = json_reader(kwargs["model_kwargs"]["config_file"])

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
    del kwargs["model_kwargs"]["config_file"]
    kwargs["model_kwargs"]["model_spec"] = model_spec_config_base
    for test_id, hp_config in ht.generate(
        start_id=ht_start_id,
        end_id=ht_end_id
    ):
        kwargs_single = copy.deepcopy(kwargs)
        kwargs_single["model_kwargs"] = dict_conservative_update(
            dict_base=kwargs["model_kwargs"],
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
        parser.add_argument("-ff", "--feature_file", default="feature_train")
        # parser.add_argument("-lf", "--label_file", default="posts_reactions_all_joined_sampled_20")
        parser.add_argument("-lf", "--label_file", default="label_train")
        parser.add_argument("-wf", "--weight_file", default=None)
        # parser.add_argument("-tf", "--task_file", default="posts_content_all_text_ids_joined_sampled_20")
        parser.add_argument("-tf", "--task_file", default="id_train")
        parser.add_argument("-tp", "--train_pattern", default="_train", type=str)
        parser.add_argument("-vp", "--valid_pattern", default="_test", type=str)
        parser.add_argument("-M", "--Model", default="dmtrl_Tucker")
        parser.add_argument("-cf", "--config_file", default="config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-rpb", "--restore_path_bottom", default=None)
        parser.add_argument("-rpt", "--restore_path_top", default=None)
        parser.add_argument("-rpr", "--restore_path_regularization", default=None)
        parser.add_argument("-mn", "--model_name", default="ht/MNIST_MTL/dmtrl_Tucker_test")
        parser.add_argument("-fss", "--full_split_saver", default=False, action="store_true")
        parser.add_argument("-Mp", "--Model_primary", default=None)
        parser.add_argument("-cfp", "--config_file_primary", default="config.json")
        parser.add_argument("-rpp", "--restore_path_primary", default=None)
        parser.add_argument("-rpbp", "--restore_path_bottom_primary", default=None)
        parser.add_argument("-rptp", "--restore_path_top_primary", default=None)
        parser.add_argument("-rprp", "--restore_path_regularization_primary", default=None)
        parser.add_argument("-mnp", "--model_name_primary", default="dmtrl_Tucker_test_primary")
        parser.add_argument("-fssp", "--full_split_saver_primary", default=False, action="store_true")
        parser.add_argument("-hcf", "--ht_config_file", default="../experiment/ht_configs/shared_bottom_config.json")
        parser.add_argument("-hcfa", "--ht_config_file_additional", default=None, type=str)
        parser.add_argument("-hsp", "--ht_save_path", default="../ht_log/MNIST_MTL/dmtrl_Tucker/", type=str)
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
        model_kwargs=dict(
            Model=Models[args.Model],
            config_file=args.model_dir + args.Model + "_" + args.config_file,
            restore_path=args.restore_path,
            partial_restore_paths={
                "save_path_bottom": args.restore_path_bottom,
                "save_path_task_specific_top": args.restore_path_top,
                "save_path_regularization": args.restore_path_regularization
            },
            model_name=args.model_name,
            full_split_saver=args.full_split_saver
        ),
        model_primary_kwargs=None if args.Model_primary is None else dict(
            Model=Models[args.Model_primary],
            config_file=args.model_dir + args.Model_primary + "_" + args.config_file,
            restore_path=args.restore_path_primary,
            partial_restore_paths={
                "save_path_bottom": args.restore_path_bottom_primary,
                "save_path_task_specific_top": args.restore_path_top_primary,
                "save_path_regularization": args.restore_path_regularization_primary
            },
            model_name=args.model_name_primary,
            full_split_saver=args.full_split_saver_primary
        )
    )
