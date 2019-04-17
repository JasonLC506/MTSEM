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


if __name__ == "__main__":
    print(json_reader("../dynamic/config.json"))
