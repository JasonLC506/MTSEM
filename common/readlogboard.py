import _pickle as cPickle
import numpy as np
import os
import re


valid_pattern = re.compile(r'.*_valid.*')


def best(steps, value, global_best_step=None, min_best=True):
    value = value.squeeze()
    assert value.ndim == 1
    if min_best:
        best_step = steps[np.argmin(value)]
        best_value = np.min(value)
    else:
        best_step = steps[np.argmax(value)]
        best_value = np.max(value)
    if global_best_step is not None:
        global_best_index = np.argwhere(steps == global_best_step)
        global_best_value = value[global_best_index]
    else:
        global_best_index, global_best_value = None, None
    return best_step, best_value, global_best_step, global_best_value


def read(directory, main_indicator="epoch_losses_valid_00"):
    files = sorted(list(os.listdir(directory)))
    variables = {"step": np.array([], dtype=np.int32)}
    for file in files:
        with open(os.path.join(directory, file), 'rb') as f:
            variables_step = cPickle.load(f)
            for name in variables_step:
                value = variables_step[name].value
                value = np.array(value)
                value_expanded = np.expand_dims(value, axis=0)
                if name in variables:
                    variables[name] = np.concatenate([variables[name], value_expanded], axis=0)
                else:
                    variables[name] = value_expanded
            variables["step"] = np.concatenate([variables["step"], np.array([int(file)])], axis=0)

    names = list(variables.keys())
    names_train = []
    names_valid = []
    for name in names:
        if name == "step":
            continue
        elif valid_pattern.match(name):
            names_valid.append(name)
        else:
            names_train.append(name)
    names_train.sort()
    names_valid.sort()

    # main_indicator #
    bests = best(variables['step'], variables[main_indicator])
    global_best_step = bests[0]
    return global_best_step, names_train, names_valid, variables


if __name__ == "__main__":
    read("../summary/FC")
