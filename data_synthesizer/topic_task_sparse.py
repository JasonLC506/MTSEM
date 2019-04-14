"""
generate synthetic data based on topic-task-sparse assumption
"""
import numpy as np
from scipy import special
import os
import _pickle as cPickle
import argparse


DTYPE = np.float32


class topicTaskSparse(object):
    def __init__(
            self,
            T,
            G=1,
            M=1,
            sigma_w0=1.0,
            sigma_wg=0.5,
            sigma_x0=1.0,
            sigma_xg=1.0,
            sigma_y=1.0
    ):
        """
        :param T:  # tasks
        :param G:  # topics
        :param M:  # special tasks in each topic
        :param sigma_w0: # stddev of global weights
        :param sigma_wg: # stddev of weights of special tasks
        :param sigma_x0: # stddev of cluster centers of features
        :param sigma_xg: # stddev of within-cluster features
        :param sigma_y:  # stddev of y given x, w
        """
        self.pars = {
            "T": T,
            "G": G,
            "M": M,
            "sigma_w0": sigma_w0,
            "sigma_wg": sigma_wg,
            "sigma_x0": sigma_x0,
            "sigma_xg": sigma_xg,
            "sigma_y": sigma_y,
            "random_seed": 0
        }
        self.variable = dict()

    def generate_variable(
            self,
            x_shape,
            w_shape,
            random_seed,
            verbose=True
    ):
        """
        generate cluster centers of features, weights
        :param x_shape: feature shape, !list
        :param w_shape: basic weight shape [x_shape, y_shape], !list
        :param random_seed: enable reproducible
        :param verbose: visualization
        :return:
        """
        self.pars["random_seed"] = random_seed
        np.random.seed(random_seed)
        self.pars["x_shape"] = x_shape
        self.pars["w_shape"] = w_shape

        # generate #
        w0 = np.random.normal(
            loc=0.0,
            scale=self.pars["sigma_w0"],
            size=self.pars['w_shape']
        )
        theta_g = np.random.normal(
            loc=0.0,
            scale=self.pars['sigma_x0'],
            size=[self.pars["G"]] + self.pars["x_shape"]
        )
        w_gt = []
        specials = []
        for g in range(self.pars["G"]):
            w_g = np.zeros(
                shape=[self.pars["T"]] + self.pars["w_shape"],
                dtype=DTYPE
            )
            t_index = np.arange(w_g.shape[0])
            np.random.shuffle(t_index)
            special_indices = t_index[:self.pars["M"]]
            w_g[special_indices] = np.random.normal(
                loc=0.0,
                scale=self.pars["sigma_wg"],
                size=[special_indices.shape[0]] + self.pars["w_shape"]
            )
            w_gt.append(w_g)
            specials.append(special_indices)
        w_gt = np.array(w_gt)
        specials = np.array(specials)

        self.variable = {
            "w0": w0,
            "theta_g": theta_g,
            "w_gt": w_gt,
            "specials": specials
        }

        if verbose:
            print("variables: %s" % str(self.variable))

    initialize = generate_variable

    def generate_data(
            self,
            n_gt,
            y_shape=None
    ):
        """
        generate data
        :param n_gt: number of samples per topic per task
        :param y_shape: shape of y, if None, default list(w_shape[-1]), currently for multi-class classification
        :return:
        """
        data = {
            "feature": [],
            "label": [],
            "id": []
        }
        if y_shape is None:
            y_shape = [self.pars['w_shape'][-1]]
        data_size_bynow = 0
        for g in range(self.pars["G"]):
            for t in range(self.pars["T"]):
                x_gt = np.random.normal(
                    loc=self.variable["theta_g"][g],
                    scale=self.pars["sigma_xg"],
                    size=[n_gt] + self.pars["x_shape"]
                )
                w_gt = self.variable["w0"] + self.variable["w_gt"][g, t]
                y_gt = np.tensordot(
                    a=x_gt,
                    b=w_gt,
                    axes=(1, 0)
                ) + np.random.normal(
                    loc=0.0,
                    scale=self.pars['sigma_y'],
                    size=[n_gt] + y_shape
                )
                y_gt = special.softmax(y_gt, axis=-1)
                sample_id = np.arange(n_gt) + data_size_bynow

                for i in range(n_gt):
                    data["feature"].append(x_gt[i])
                    data["label"].append(y_gt[i])
                    data["id"].append("t%d_g%d_%d" % (t, g, sample_id[i]))

                data_size_bynow += n_gt
        return data


def write2file(
        data,
        meta_data=None,
        dir_name="../data/synthetic",
        separator=","
):
    """
    write to directory as files, format readable for ../experiment/data_generator
    :param data:
    :param meta_data:
    :param dir_name:
    :param separator: separator to separate dimensions of features and labels
    :return:
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if meta_data is not None:
        with open(os.path.join(dir_name, "meta_data"), 'wb') as f:
            cPickle.dump(meta_data, f)
    for key_name in data:
        with open(os.path.join(dir_name, key_name), 'w') as f:
            for i in range(len(data[key_name])):
                data_instance = data[key_name][i]
                if isinstance(data_instance, np.ndarray):
                    # feature or label #
                    data_str = separator.join(
                        list(map(
                            str,
                            data_instance.tolist()
                        ))
                    )
                elif isinstance(data_instance, str):
                    data_str = data_instance
                else:
                    print("data_instance: %s" % str(data_instance))
                    raise RuntimeError("unknown data_instance type %s" % str(type(data_instance)))
                f.write(data_str + "\n")
    print("done")


class ArgParse(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-T", "--T", type=int, default=12)
        parser.add_argument("-G", "--G", type=int, default=3)
        parser.add_argument("-M", "--M", type=int, default=1)
        parser.add_argument("-sw0", "--sigma_w0", type=float, default=1.0)
        parser.add_argument("-swg", "--sigma_wg", type=float, default=0.5)
        parser.add_argument("-sx0", "--sigma_x0", type=float, default=1.0)
        parser.add_argument("-sxg", "--sigma_xg", type=float, default=1.0)
        parser.add_argument("-sy", "--sigma_y", type=float, default=1.0)
        parser.add_argument("-xd", "--x_dim", type=int, default=64)
        parser.add_argument("-yd", "--y_dim", type=int, default=5)
        parser.add_argument("-rs", "--random_seed", type=int, default=2019)
        parser.add_argument("-v", "--verbose", default=True, action="store_false")
        parser.add_argument("-n_gt", "--n_gt", type=int, default=500)
        parser.add_argument("-dn", "--dir_name", type=str, default="../data/synthetic_topic_task_sparse")
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = ArgParse().parse_args()
    synthesizer = topicTaskSparse(
        T=args.T,
        G=args.G,
        M=args.M,
        sigma_w0=args.sigma_w0,
        sigma_wg=args.sigma_wg,
        sigma_x0=args.sigma_x0,
        sigma_xg=args.sigma_xg,
        sigma_y=args.sigma_y
    )
    synthesizer.generate_variable(
        x_shape=[args.x_dim],
        w_shape=[args.x_dim, args.y_dim],
        random_seed=args.random_seed,
        verbose=args.verbose
    )
    data = synthesizer.generate_data(
        n_gt=args.n_gt
    )
    write2file(
        data=data,
        meta_data=synthesizer.pars,
        dir_name=args.dir_name
    )
