"""
hyperparamter tuner
"""
import numpy as np
import copy
import warnings
import os

from experiment import json_reader, json_dumper


class HyperParameter(object):
    """ wrap hyperparameter """
    def __init__(self, name, options=None, best=None):
        self.name = name
        self.options = [] if options is None else options
        self.best = best


class HyperparameterTuner(object):
    def __init__(self, save_path):
        self.hp = dict()                 # dictionary of all hp
        self.hp_names = []               # order of hps in each combination
        self.id2hps = dict()             # dictionary of id to hp combination
        self.id2perf = dict()            # dictionary of id to performances
        save_path_dir = "/".join(save_path.split("/")[:-1])
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        self.save_path_hps = save_path + "_hps"
        self.save_path_perfs = save_path + "_perfs"
        self.save_path_best_hp = save_path + "_best_hp.json"
        self.save_path_best_perf = save_path + "_best_perf"
        self.save_path_log = save_path + "_log"
        self.start_id_ = 0

    def initialize(
            self,
            hps,
            hp_options_keyword="hp_options",
            scope_separator="/"
    ):
        """
        :param hps: {hp_name: {"hp_option": }}
        :param hp_options_keyword: str to identity options
        :param scope_separator: separate scope in name string
        """
        queue = [(hps, "")]
        while len(queue) > 0:
            hp_node, hp_name_prefix = queue.pop()
            if isinstance(hp_node, dict):
                if hp_options_keyword in hp_node:
                    self.hp[hp_name_prefix] = HyperParameter(
                        name=hp_name_prefix,
                        options=hp_node[hp_options_keyword]
                    )
                else:
                    for hp_name in hp_node:
                        queue.append(
                            (hp_node[hp_name], hp_name_prefix + scope_separator + hp_name)
                        )
            else:
                # where single value is given #
                self.hp[hp_name_prefix] = HyperParameter(
                    name=hp_name_prefix,
                    options=[hp_node]
                )
        # verbose #
        for hp_name in self.hp:
            self.log_write("hp '%s': {options: %s}" % (hp_name, str(self.hp[hp_name].options)))

        # build id2hps #
        self._hps_build()
        return self

    def _hps_build(self):
        """
        using all combinations of options of hps as hps sets
        """
        self.hp_names = hp_names = sorted(list(self.hp.keys()))
        hp_sorted = [self.hp[hp_name] for hp_name in hp_names]
        n_hps = len(self.hp_names)
        n_ops_hps = list(map(lambda x: len(x.options), hp_sorted))
        total_ops = np.prod(np.array(n_ops_hps))
        self.log_write("total_ops: %d" % total_ops)
        per_ids = np.arange(
            total_ops,
            dtype=np.int32
        ).reshape(n_ops_hps)

        for id_ in range(total_ops):
            index = np.squeeze(np.argwhere(per_ids == id_), axis=0)
            self.id2hps[id_] = [hp_sorted[i].options[index[i]] for i in range(n_hps)]

    def generate(
            self,
            start_id=None,
            end_id=None
    ):
        if start_id is None:
            start_id = self.start_id_
        if end_id is None:
            end_id = len(self.id2hps)
        for id_ in range(start_id, min([end_id, len(self.id2hps)])):
            hps = self.id2hps[id_]
            hps_config = self.out_pattern(
                keys=self.hp_names,
                values=hps
            )
            self.log_write("testing %d/%d: hps: %s" % (id_ + 1, len(self.id2hps), str(hps_config)))
            yield id_, hps_config
            self.write2file(
                id_=id_,
                result=hps,
                filename=self.save_path_hps
            )

    def find_best(
            self,
            primary_index=0,
            max_best=False
    ):
        if len(self.id2hps) != len(self.id2perf):
            warnings.warn("perf %d!= hps %d" % (len(self.id2perf), len(self.id2hps)))
        perfs = np.array(list(self.id2perf.values()))
        ids = np.array(list(self.id2perf.keys()))
        if max_best:
            argbest = np.argmax
        else:
            argbest = np.argmin
        best_ind = argbest(perfs[:, primary_index], axis=0)
        best_id = ids[best_ind]
        best_hps = self.out_pattern(
            keys=self.hp_names,
            values=self.id2hps[best_id]
        )
        best_perf = self.id2perf[best_id]
        with open(self.save_path_best_hp, "w") as f:
            json_dumper(best_hps, f)
        self.write2file(
            id_=best_id,
            result=best_perf,
            filename=self.save_path_best_perf
        )
        return best_hps, best_perf

    def read_perf(
            self,
            id_,
            perf,
            write_out=True
    ):
        self.id2perf[id_] = perf
        if write_out:
            self.write2file(
                id_=id_,
                result=perf,
                filename=self.save_path_perfs,
            )

    def restore(
            self,
            perfs_file,
            id_separator="\t",
            field_separator="|",
    ):
        with open(perfs_file, "r") as f:
            for line in f:
                try:
                    id_, perfs = line.rstrip("\n").split(id_separator)
                    id_ = int(id_)
                    perfs = list(map(float, perfs.split(field_separator)))
                    self.read_perf(
                        id_=id_,
                        perf=perfs,
                        write_out=False
                    )
                except:
                    warnings.warn("perf parsing problem with line %s" % line.rstrip("\n"))

    @staticmethod
    def write2file(
            id_,
            result,
            filename,
            id_separator="\t",
            field_separator="|"
    ):
        with open(filename, "a") as f:
            f.write(
                str(id_) + id_separator + field_separator.join(list(map(str, result))) + "\n"
            )

    def log_write(self, string):
        with open(self.save_path_log, "a") as f:
            f.write(string.rstrip("\n") + "\n")

    @staticmethod
    def out_pattern(
            keys,
            values,
            scope_separator="/"
    ):
        """
        it should be the reverse process of initialize
        """
        assert len(keys) == len(values)
        out = {}
        for i in range(len(keys)):
            hp_name = keys[i]
            value = values[i]
            hp_name_scopes = hp_name.strip(scope_separator).split(scope_separator)
            current_scope = out
            for j in range(len(hp_name_scopes) - 1):
                hp_name_scope = hp_name_scopes[j]
                current_scope.setdefault(hp_name_scope, {})
                current_scope = current_scope[hp_name_scope]
            if hp_name_scopes[-1] in current_scope:
                raise RuntimeError("hp duplicate define")
            current_scope[hp_name_scopes[-1]] = value
        return out


def dict_conservative_update(
        dict_base,
        dict_new
):
    """
    update the dict_base with dict_new, if same key, cascade to inner dict
    :param dict_base:
    :param dict_new:
    :return: dict_updated
    """
    dict_updated = copy.deepcopy(dict_base)
    for key_name in dict_base:
        if key_name in dict_new:
            if isinstance(dict_new[key_name], dict) and not isinstance(dict_base[key_name], dict):
                dict_updated[key_name] = dict_new[key_name]
                continue
            try:
                assert type(dict_base[key_name]) == type(dict_new[key_name])
            except AssertionError:
                print("base '%s': %s" % (key_name, str(dict_base[key_name])))
                print("new  '%s': %s" % (key_name, str(dict_new[key_name])))
                raise RuntimeError
            if isinstance(dict_base[key_name], dict):
                dict_updated[key_name] = dict_conservative_update(
                    dict_base=dict_base[key_name],
                    dict_new=dict_new[key_name]
                )
            else:
                dict_updated[key_name] = dict_new[key_name]
    return dict_updated


def dict_update(
        dict_base,
        dict_new
):
    """
    update the dict_base with dict_new, add new key if needed, cascade to inner dict
    :param dict_base:
    :param dict_new:
    :return: dict_updated
    """
    dict_updated = copy.deepcopy(dict_base)
    for key_name in dict_new:
        if key_name in dict_base:
            if isinstance(dict_new[key_name], dict) and not isinstance(dict_base[key_name], dict):
                dict_updated[key_name] = dict_new[key_name]
                continue
            try:
                assert type(dict_base[key_name]) == type(dict_new[key_name])
            except AssertionError:
                print("base '%s': %s" % (key_name, str(dict_base[key_name])))
                print("new  '%s': %s" % (key_name, str(dict_new[key_name])))
                raise RuntimeError
            if isinstance(dict_base[key_name], dict):
                dict_updated[key_name] = dict_update(
                    dict_base=dict_base[key_name],
                    dict_new=dict_new[key_name]
                )
            else:
                dict_updated[key_name] = dict_new[key_name]
        else:
            dict_updated[key_name] = dict_new[key_name]
    return dict_updated


if __name__ == "__main__":
    ht = HyperparameterTuner(save_path="../ht_log/test")
    config = json_reader("../ht_log/test_config.json")
    print(config)
    config_model = json_reader("../models/dmtrl_Tucker_config.json")
    print(config_model)
    print("conservative update")
    config_model_updated = dict_conservative_update(
        dict_base=config_model,
        dict_new=config
    )
    print(config_model_updated)
    ht.initialize(
        hps=config_model_updated
    )
    print(ht.id2hps)
    for id_ in ht.id2hps.keys():
        print("id: %d" % id_)
        print(ht.out_pattern(
            keys=ht.hp_names,
            values=ht.id2hps[id_]
        ))
        break

    print("start testing")
    for test_id, hyper_config in ht.generate():
        # print("test_id: %d" % test_id)
        # print(hyper_config)
        perf = np.random.random([2])
        ht.read_perf(id_=test_id, perf=perf)

    print("performance")
    print(np.array(list(ht.id2perf.values())))
    hps, perf = ht.find_best(
        primary_index=0
    )
    print(hps)
    print(perf)
