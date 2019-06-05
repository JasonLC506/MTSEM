"""
copy and paste _best_hp.json from ht_log to configs/
"""
import os
import argparse
from experiment import json_dumper, json_reader
from models import Models
import warnings

def read_and_dump(filename, filename_target):
    if not os.path.exists(filename):
        warnings.warn("'%s' not exists" % filename)
        return None
    config = json_reader(filename)
    model_spec = config["model_spec"]
    with open(filename_target, "w") as f:
        json_dumper(model_spec, f)


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-sd", "--source_dir", default="../ht_log/MNIST_MTL")
        parser.add_argument("-td", "--target_dir", default="../configs/MNIST_MTL")
        parser.add_argument("-p", "--pattern", default=None)
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()
    print(args.pattern)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    for fn in os.listdir(args.source_dir):
        if fn not in Models:
            print("'%s' is not a valid model name" % fn)
            continue
        if args.pattern:
            if fn not in args.pattern.split(','):
                print("'%s' is not in defined pattern" % fn)
                continue
        filename = os.path.join(args.source_dir, fn, "_best_hp.json")
        filename_target = os.path.join(args.target_dir, fn + "_config.json")
        read_and_dump(
            filename=filename,
            filename_target=filename_target
        )
    print("done")
