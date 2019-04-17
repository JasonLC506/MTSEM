import argparse
import numpy as np

from experiment import DataGeneratorTrainTest, DataGeneratorFull, StageWiseSample
from models import FC, SharedBottom, InterTaskL2, DmtrlTucker, CrossStitch, MMoE, MultilinearRelationshipNetwork, \
    TopicTaskSparse
from experiment import json_reader
from common.readlogboard import read


Models = {
    'fc': FC,
    'shared_bottom': SharedBottom,
    'inter_task_l2': InterTaskL2,
    'dmtrl_Tucker': DmtrlTucker,
    'cross_stitch': CrossStitch,
    "mmoe": MMoE,
    "multilinear_relationship_network": MultilinearRelationshipNetwork,
    "topic_task_sparse": TopicTaskSparse
}

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
        Model=FC,
        config_file=None,
        restore_path=None,
        partial_restore_paths=None,
        model_name=None,
        full_split_saver=False,
        num_repeats=1
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
    :param Model:
    :param config_file:
    :param restore_path:
    :param partial_restore_paths: dictionary of paths for each partial saver
    :param model_name:
    :param full_split_saver:
    :param num_repeats:
    :return:
    """
    np.random.seed(RANDOM_SEED_NP)
    best_epoch_run, results_run = [], []
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
        model_spec = json_reader(config_file)

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

        model = init_model(data=data)
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
            model = init_model(data=data_test)
            model.restore(
                save_path="../ckpt/" + model.model_name + "/epoch_%03d" % int(best_epoch)       # dependent on save path
            )
            results = model.test(
                data_generator=data_test
            )
            print("testing output: %s" % str(results))
        results_run.append(results)

    print("results:\n %s" % str(np.array(results_run)))
    print("best_epochs:\n %s" % str(np.array(best_epoch_run)))


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
        parser.add_argument("-M", "--Model", default="cross_stitch")
        parser.add_argument("-cf", "--config_file", default="config.json")
        parser.add_argument("-rp", "--restore_path", default=None)
        parser.add_argument("-rpb", "--restore_path_bottom", default=None)
        parser.add_argument("-rpt", "--restore_path_top", default=None)
        parser.add_argument("-rpr", "--restore_path_regularization", default=None)
        parser.add_argument("-mn", "--model_name", default="cross_stitch_MNIST_MTL")
        parser.add_argument("-fss", "--full_split_saver", default=False, action="store_true")
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
        Model=Models[args.Model],
        config_file=args.model_dir + args.Model + "_" + args.config_file,
        restore_path=args.restore_path,
        partial_restore_paths={
            "save_path_bottom": args.restore_path_bottom,
            "save_path_task_specific_top": args.restore_path_top,
            "save_path_regularization": args.restore_path_regularization
        },
        model_name=args.model_name,
        full_split_saver=args.full_split_saver,
        num_repeats=args.num_repeats
    )
