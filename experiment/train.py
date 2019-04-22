import argparse
import numpy as np

from experiment import DataGeneratorTrainTest, DataGeneratorFull, StageWiseSample
from models import Models
from experiment import json_reader, dict_conservative_update
from common.readlogboard import read


RANDOM_SEED_NP = 2019


def run(
        feature_file,
        label_file,
        weight_file=None,
        task_file=None,
        topic_file=None,
        train_ratio=0.7,
        valid_ratio=0,
        stage_wise=True,
        num_repeats=1,
        **model_kwargs_wrap
):
    """

    :param feature_file:
    :param label_file:
    :param weight_file:
    :param task_file:
    :param topic_file: only effective in test
    :param train_ratio:
    :param valid_ratio:
    :param stage_wise:
    :param num_repeats:
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
    best_epoch_run, results_run = [], []

    model_kwargs = model_kwargs_wrap["model_kwargs"]
    model_primary_kwargs = model_kwargs_wrap["model_primary_kwargs"]
    for i_run in range(num_repeats):
        test_sampler = StageWiseSample(
            feature_file=feature_file,
            label_file=label_file,
            task_file=task_file,
            sample_rate=1 - train_ratio - valid_ratio
        )
        test_sampler.sample()
        print("done test sampling")
        # initialize data #
        data = DataGeneratorTrainTest(
            feature_file=feature_file + test_sampler.output_pattern_remain,
            label_file=label_file + test_sampler.output_pattern_remain,
            weight_file=weight_file if weight_file is None else weight_file + test_sampler.output_pattern_remain,
            task_file=task_file if task_file is None else task_file + test_sampler.output_pattern_remain,
            train_ratio=train_ratio / (train_ratio + valid_ratio),
            valid_ratio=valid_ratio / (train_ratio + valid_ratio),
            stage_wise=stage_wise
        )
        data_train = data.train
        data_valid = data.valid if data.valid.data_size > 0 else None

        # initialize model #
        model_spec = json_reader(model_kwargs['config_file'])
        if model_primary_kwargs is not None:
            # train primary model to initialize model #
            model_spec_primary_ = json_reader(model_primary_kwargs['config_file'])
            model_spec_primary = dict_conservative_update(
                dict_base=model_spec_primary_,
                dict_new=model_spec
            )                         # make sure the shared hps are consistent
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

        [best_epoch, results], model_best_save_path, model = train_model(
            data_train=data_train,
            data_valid=data_valid,
            model_spec=model_spec,
            **model_kwargs
        )
        best_epoch_run.append(best_epoch)

        # test #
        data_test = DataGeneratorFull(
            feature_file=feature_file + test_sampler.output_pattern,
            label_file=label_file + test_sampler.output_pattern,
            weight_file=weight_file if weight_file is None else weight_file + test_sampler.output_pattern,
            task_file=task_file if task_file is None else task_file + test_sampler.output_pattern,
            topic_file=topic_file,
            weight_task=True
        )
        data_test = data_test if data_test.data_size > 0 else None

        if data_test is not None:
            model = init_model(
                data=data_test,
                Model=model_kwargs["Model"],
                model_spec=model_spec,
                model_name=model_kwargs["model_name"]
            )
            model.restore(
                save_path=model_best_save_path.format(int(best_epoch))
            )
            results = model.test(
                data_generator=data_test
            )
            print("testing output: %s" % str(results))
        results_run.append(results)

    print("results:\n %s" % str(np.array(results_run)))
    print("best_epochs:\n %s" % str(np.array(best_epoch_run)))


def init_model(
        data,
        Model,
        model_spec,
        model_name
):
    model = Model(
        feature_dim=data.feature_dim,
        label_dim=data.label_dim,
        task_dim=data.task_dim,
        model_spec=model_spec,
        model_name=model_name
    )
    model.initialization()
    return model


def train_model(
        Model,
        model_spec,
        model_name=None,
        partial_restore_paths=None,
        restore_path=None,
        full_split_saver=False,
        data_train=None,
        data_valid=None,
        **kwargs
):
    model = init_model(
        data=data_train,
        Model=Model,
        model_spec=model_spec,
        model_name=model_name
    )
    if partial_restore_paths is not None:
        model.partial_restore(**partial_restore_paths)
    if restore_path is not None:
        model.restore(restore_path)

    # train #
    results = model.train(
        data_generator=data_train,
        data_generator_valid=data_valid,
        full_split_saver=full_split_saver
    )
    print("training output: %s" % str(results))

    best_epoch = read(
        directory="../summary/" + model.model_name,
        main_indicator="epoch_losses_valid_00"
    )[0]
    save_path = "../ckpt/" + model.model_name + "/epoch_{0:03d}"
    return [best_epoch, results], save_path, model


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-dd", "--data_dir", default="../data/")
        parser.add_argument('-md', "--model_dir", default="../models/")

        # parser.add_argument("-ff", "--feature_file", default="posts_content_all_features_[CLS]_SUM_joined_sampled_20")
        parser.add_argument("-ff", "--feature_file", default="MNIST_MTL/feature")
        # parser.add_argument("-lf", "--label_file", default="posts_reactions_all_joined_sampled_20")
        parser.add_argument("-lf", "--label_file", default="MNIST_MTL/label")
        parser.add_argument("-wf", "--weight_file", default=None)
        # parser.add_argument("-tf", "--task_file", default="posts_content_all_text_ids_joined_sampled_20")
        parser.add_argument("-tf", "--task_file", default="MNIST_MTL/id")
        parser.add_argument("--topic_file", default=None)
        parser.add_argument("-tr", "--train_ratio", default=0.7, type=float)
        parser.add_argument("-vr", "--valid_ratio", default=0.1, type=float)
        parser.add_argument("-sw", "--stage_wise", default=True, action='store_false')
        parser.add_argument("-M", "--Model", default="topic_task_sparse_layer_wise")
        parser.add_argument("-cf", "--config_file", default="config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-rpb", "--restore_path_bottom", default=None)
        parser.add_argument("-rpt", "--restore_path_top", default=None)
        parser.add_argument("-rpr", "--restore_path_regularization", default=None)
        parser.add_argument("-mn", "--model_name", default="topic_task_sparse_layer_wise_MNIST_MTL_test")
        parser.add_argument("-fss", "--full_split_saver", default=False, action="store_true")
        parser.add_argument("-Mp", "--Model_primary", default=None)
        parser.add_argument("-cfp", "--config_file_primary", default="config.json")
        parser.add_argument("-rpp", "--restore_path_primary", default=None)
        parser.add_argument("-rpbp", "--restore_path_bottom_primary", default=None)
        parser.add_argument("-rptp", "--restore_path_top_primary", default=None)
        parser.add_argument("-rprp", "--restore_path_regularization_primary", default=None)
        parser.add_argument("-mnp", "--model_name_primary", default="cross_stitch_primary")
        parser.add_argument("-fssp", "--full_split_saver_primary", default=False, action="store_true")
        parser.add_argument("-nr", "--num_repeats", default=1, type=int)
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()
    print("input: %s" % args.__dict__)
    run(
        feature_file=args.data_dir + args.feature_file,
        label_file=args.data_dir + args.label_file,
        weight_file=None if args.weight_file is None else args.data_dir + args.weight_file,
        task_file=None if args.task_file is None else args.data_dir + args.task_file,
        topic_file=None if args.topic_file is None else args.data_dir + args.topic_file,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        stage_wise=args.stage_wise,
        num_repeats=args.num_repeats,
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
