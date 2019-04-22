import argparse
import re
import os

from experiment import json_reader, HyperparameterTuner, dict_conservative_update, dict_update


def ht_find_best(
        ht_config_file,
        ht_config_file_additional,
        ht_save_path,
        ht_perf_files,
        model_spec_config_file,
):
    ht_config = json_reader(ht_config_file)
    if ht_config_file_additional is not None:
        ht_config_additional = json_reader(ht_config_file_additional)
        ht_config = dict_update(
            dict_base=ht_config,
            dict_new=ht_config_additional
        )
    model_spec_config_base = json_reader(model_spec_config_file)

    ht = HyperparameterTuner(
        save_path=ht_save_path
    )
    ht_config["model_spec"] = dict_conservative_update(
        dict_base=model_spec_config_base,
        dict_new=ht_config["model_spec"]
    )       # update model spec for different models to avoid redundant hps
    print(ht_config)                             # verbose
    ht.initialize(hps=ht_config)
    for ht_perf_file in ht_perf_files:
        ht.restore(
            perfs_file=ht_perf_file
        )
    print(ht.find_best(max_best=True))


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-md', "--model_dir", default="../models/")
        parser.add_argument("-M", "--Model", default="shared_bottom")

        parser.add_argument("-cf", "--config_file", default="config.json")
        parser.add_argument("-hcf", "--ht_config_file", default="../experiment/ht_configs/MNIST_MTL/ht_config.json")
        parser.add_argument("-hsp", "--ht_save_path", default="../ht_log/MNIST_MTL/shared_bottom/", type=str)
        parser.add_argument("-hcfa", "--ht_config_file_additional", default=None, type=str)
        parser.add_argument("-hpf", "--ht_perf_file", default=r'../ht_log/MNIST_MTL/shared_bottom/[\d]*_perfs')
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()
    print("input: %s" % args.__dict__)
    ht_perf_dir = "/".join(args.ht_perf_file.split("/")[:-1])
    ht_perf_files = []
    for filename_ in os.listdir(ht_perf_dir):
        filename = ht_perf_dir + "/" + filename_
        if re.match(args.ht_perf_file, filename):
            ht_perf_files.append(filename)
    print("ht_perf_files: %s" % str(ht_perf_files))
    ht_find_best(
        ht_config_file=args.ht_config_file,
        ht_config_file_additional=args.ht_config_file_additional,
        ht_save_path=args.ht_save_path,
        ht_perf_files=ht_perf_files,
        model_spec_config_file=args.model_dir + args.Model + "_" + args.config_file
    )
