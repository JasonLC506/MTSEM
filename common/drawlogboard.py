import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np
import argparse
import os
import re

from common.readlogboard import read


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


def draw_single(axis, name, value, steps, global_best_step=None):
    value = value.squeeze()
    if value.ndim == 1:
        axis.plot(steps, value)
        bests = best(steps, value, global_best_step)
        print(
            "best %s: at %d, with value %f, global best value: %f" % (
                name, int(bests[0]), float(bests[1]), float(bests[-1])
            )
        )
    else:
        value = value.reshape([value.shape[0], -1])
        for i in range(value.shape[-1]):
            axis.plot(steps, value[:, i], label="%d" % i)
            bests = best(steps, value[:, i], global_best_step)
            print(
                "best %s: at %d, with value %f, global best value: %f" % (
                    name + "_%d" % i, int(bests[0]), float(bests[1]), float(bests[-1])
                )
            )
    axis.set_title(name)
    axis.legend()
    return axis


def draw(directory, main_indicator="epoch_losses_valid_00"):
    global_best_step, names_train, names_valid, variables = read(
        directory=directory,
        main_indicator=main_indicator
    )

    # draw #
    fig, axes = plt.subplots(nrows=2, ncols=max(len(names_train), len(names_valid)))
    for i in range(len(names_train)):
        axis = axes[0, i]
        draw_single(
            axis=axis,
            name=names_train[i],
            value=variables[names_train[i]],
            steps=variables["step"],
            global_best_step=global_best_step
        )
    for i in range(len(names_valid)):
        axis = axes[1, i]
        draw_single(
            axis=axis,
            name=names_valid[i],
            value=variables[names_valid[i]],
            steps=variables["step"],
            global_best_step=global_best_step
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw("../summary/FC")
