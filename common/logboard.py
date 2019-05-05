import numpy as np
import os
import _pickle as cPickle
import warnings


class LogBoardVariable(object):
    def __init__(
            self,
            name,
            min_best=True
    ):
        self.name = name
        self.value = None
        self.min_best = min_best


class LogBoard(object):
    def __init__(
            self,
            directory="../summary/",
            initial_step=0
    ):
        self.directory = None
        self._mkdir(directory=directory)
        self.variables = {}
        self.step = initial_step

    def _mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory

    def add(
            self,
            name,
            min_best=True
    ):
        assert name not in self.variables
        self.variables[name] = LogBoardVariable(
                name=name,
                min_best=min_best
        )

    def record(
            self,
            name,
            value
    ):
        if name not in self.variables:
            warnings.warn("new variable added: %s" % name)
            self.add(name)
        self.variables[name].value = value

    def write(
            self,
            step=None
    ):
        if step is not None:
            self.step = step
        with open(os.path.join(self.directory, "%03d" % self.step), 'wb') as f:
            cPickle.dump(self.variables, f)
        self.step += 1


class LogBoardFake(LogBoard):
    """ fake log board without output functions """
    def __init__(self, **kwargs):
        super(LogBoardFake, self).__init__(**kwargs)

    def _mkdir(self, directory):
        pass

    def write(self, **kwargs):
        pass
