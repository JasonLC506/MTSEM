import json
import tensorflow as tf
import os


def json_reader(file_name):
    diction = json.load(open(file_name, "r"))

    def parser(a):
        def cell_parser(b, index):
            if isinstance(b[index], list) or isinstance(b[index], dict):
                b[index] = parser(b[index])
            else:
                if b[index] == "tanh":
                    b[index] = tf.nn.tanh
                elif b[index] == "relu":
                    b[index] = tf.nn.relu
                elif index == "bottom" and "config_file: " in b[index]:
                    b[index] = json_reader(b[index][13:])[index]
                elif index == "optim_params" and "config_file: " in b[index]:
                    b[index] = json_reader(b[index][13:])[index]
        if isinstance(a, list):
            for i in range(len(a)):
                cell_parser(a, i)
        elif isinstance(a, dict):
            for i in a:
                cell_parser(a, i)
        return a
    return parser(diction)


def json_dumper(json_dict, file):
    def reverse_parser(a):
        def cell_parser(b, index):
            if isinstance(b[index], list) or isinstance(b[index], dict):
                b[index] = reverse_parser(b[index])
            else:
                if "<function tanh at" in str(b[index]):
                    b[index] = "tanh"
                elif "<function relu at" in str(b[index]):
                    b[index] = "relu"
                elif index == "bottom" and "config_file: " in b[index]:
                    b[index] = json_reader(b[index][13:])[index]
                elif index == "optim_params" and "config_file: " in b[index]:
                    b[index] = json_reader(b[index][13:])[index]
        if isinstance(a, list):
            for i in range(len(a)):
                cell_parser(a, i)
        elif isinstance(a, dict):
            for i in a:
                cell_parser(a, i)
        return a
    return json.dump(reverse_parser(json_dict), file, indent=2)


if __name__ == "__main__":
    config_file = "../ht_log/test_config.json"
    config_file_out = "../ht_log/test_config_recover.json"
    print(json_dumper(json_reader(config_file), open(config_file_out, "w")))
