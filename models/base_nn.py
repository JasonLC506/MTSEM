# base class of Neural Network #
import tensorflow as tf
import numpy as np
import math
import warnings
from datetime import datetime
import _pickle as cPickle
import os

from common import LogBoard
from common import LogBoardFake


DEFAULT_SCOPE_NAME = "NN"
OPTIMIZERSEPARATOR = "OptimizerSeparator"
HISTORY_CUTOFF = 200
TF_RANDOM_SEED = 2018

LOG_PATH = "../log"
CKPT_PATH = "../ckpt"

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)


class NN(object):
    def __init__(self, graph, scope=None):
        # inherit graph or create new #
        if graph is None:
            graph = tf.Graph()          # if the only model/first model
        self.scope_name = DEFAULT_SCOPE_NAME if scope is None else scope
        self.graph = graph
        self._setup_graph()                                           # reward model details
        self.sess = None

    def _setup_graph(self):
        with self.graph.as_default():
            tf.set_random_seed(TF_RANDOM_SEED)
            # when for sharing, set self.scope_name is None
            with tf.device("/cpu:0"):
                with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE):
                    self._setup_placeholder()
                    self._setup_net()
                    self._setup_loss()
                    with tf.variable_scope("OptimizerSeparator", reuse=tf.AUTO_REUSE):
                        # protect optimizer from double model sharing #
                        self._setup_optim()
                    self.saver = tf.train.Saver(max_to_keep=1000)
            self.init = tf.global_variables_initializer()
        return self.graph

    def _setup_placeholder(self):
        raise NotImplementedError

    def _setup_net(self):
        raise NotImplementedError

    def _setup_loss(self):
        raise NotImplementedError

    def _setup_optim(self):
        raise NotImplementedError

    @staticmethod
    def _train_w_generator(
            data_generator,
            fn_feed_dict,
            op_optimizer,
            op_losses,
            session,
            op_data_size=None,
            fn_op_update=lambda x: None,
            batch_size=512,
            max_epoch=10,
            verbose=True,
            op_savers=None,
            save_path_prefixs=["NN"],
            data_generator_valid=None,
            log_board_dir=None
    ):
        """
        mini-batch training
        :param data_generator
        :param fn_feed_dict: function to build feed_dict for specific graph, args: data, batch_index
        :param op_optimizer: tf operator of optimizer
        :param op_losses: tf operator of loss and other values to extract
        :param op_data_size: tf operator to calculate effective data size
        :param batch_size:
        :param max_epoch:
        :param session:
        :param verbose: print result every epoch
        :param op_savers: tf operator of saver
        :param save_path_prefixs: prefix for save path
        :param data_generator_valid: valid data generator
        :param log_board_dir: directory for log_board save, None for not save
        :return:
        """
        start = datetime.now()
        steps = 0                                               # counting global SGD parameter update steps
        epoch_losses = []
        data_sizes = []
        if log_board_dir is not None:
            log_board = LogBoard(directory=log_board_dir)
        else:
            log_board = LogBoardFake()
        for epoch in range(max_epoch):
            epoch_losses = [np.zeros(1) for _ in range(len(op_losses))]
            data_sizes = 0
            i_batch = 0
            for data_batched in data_generator.generate(batch_size=batch_size):
                for data_key in data_batched:
                    batch_size_true = data_batched[data_key].shape[0]
                    break
                feed_dict = fn_feed_dict(data_batched, batch_index=np.arange(batch_size_true))
                op_update = fn_op_update(steps)          # generates op_update given global steps
                fetch = [op_optimizer] + op_losses
                if op_data_size is not None:
                    fetch.append(op_data_size)
                if op_update is not None:
                    fetch.append(op_update)
                results = session.run(fetch, feed_dict=feed_dict)
                losses = results[1: 1 + len(op_losses)]
                if op_data_size is not None:
                    data_sizes_batch = results[1 + len(op_losses)]
                else:
                    data_sizes_batch = batch_size_true if "weight" not in data_batched else data_batched["weight"].sum()
                epoch_losses = [epoch_losses[i_] + losses[i_] for i_ in range(len(op_losses))]
                data_sizes += data_sizes_batch
                i_batch += 1
                steps += 1
            # multi-task task-specific losses calculation #
            data_sizes = np.array(data_sizes, dtype=np.float32)
            log_board.record("data_sizes", data_sizes)
            data_size = np.sum(data_sizes)
            for i in range(len(epoch_losses)):
                epoch_loss = epoch_losses[i]
                if np.all(epoch_loss.shape == data_sizes.shape):
                    epoch_losses[i] = np.divide(
                        epoch_loss,
                        np.maximum(1.0, data_sizes)
                    )
                else:
                    epoch_losses[i] = epoch_loss / max(1.0, float(data_size))
                log_board.record("epoch_losses_%02d" % i, epoch_losses[i])
            if data_generator_valid is not None:
                valid_losses = NN._train_w_generator(
                    data_generator=data_generator_valid,
                    fn_feed_dict=fn_feed_dict,
                    op_optimizer=op_losses[0],                           # making op_optimizer ineffective
                    op_losses=op_losses,
                    session=session,
                    op_data_size=op_data_size,
                    fn_op_update=lambda x: None,
                    batch_size=batch_size,
                    max_epoch=1,                                    # only need 1 epoch to cal loss
                    verbose=False
                )
                epoch_losses_valid, data_sizes_valid = valid_losses
                for i in range(len(epoch_losses_valid)):
                    log_board.record("epoch_losses_valid_%02d" % i, epoch_losses_valid[i])
                log_board.record("data_sizes_valid", data_sizes_valid)
            else:
                valid_losses = None
            end = datetime.now()
            log_board.write(step=epoch)
            message = "epoch %d takes %fs, loss: %s, \n with data_sizes: %s" % (
                epoch,
                (end-start).total_seconds(),
                str(epoch_losses),
                str(data_sizes)
            )
            message = message + "\n" + "valid_loss: %s" % str(valid_losses)
            if verbose:
                print(message)
                log_path = os.path.join(LOG_PATH, save_path_prefixs[-1])
                with open(log_path, "a") as logf:
                    logf.write(message + "\n")
                if op_savers is not None:
                    for op_saver, save_path_prefix in zip(op_savers, save_path_prefixs):
                        save_path = os.path.join(CKPT_PATH, save_path_prefix)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, "epoch_%03d" % epoch)
                        op_saver.save(
                            sess=session,
                            save_path=save_path
                        )
            start = end
        return epoch_losses, data_sizes

    @staticmethod
    def _feed_forward_w_generator(
            data_generator,
            fn_feed_dict,
            output,
            session,
            batch_size=512
    ):
        """
        using mini-batch to calculate feed forward
        :param data: dictionary of different entries of the data set, with the same shape[0]
        :param fn_feed_dict: function to build feed_dict for specific graph, args: data, batch_index
        :param output: the list of outputs needed
        :param session:
        :param batch_size:
        :return:
        """
        output_result = None
        for data_batched in data_generator.generate(batch_size=batch_size, random_shuffle=False):
            for data_key in data_batched:
                batch_size_true = data_batched[data_key].shape[0]
                break
            feed_dict = fn_feed_dict(data_batched, batch_index=np.arange(batch_size_true))
            output_batch = session.run(output, feed_dict=feed_dict)
            if output_result is None:
                output_result = output_batch
            else:
                output_result = [np.concatenate([output_result[i_field], output_batch[i_field]])
                                 for i_field in range(len(output_result))]
        return output_result

    @staticmethod
    def _feed_forward(
            data,
            fn_feed_dict,
            output,
            session,
            batch_size=512
    ):
        """
        using mini-batch to calculate feed forward
        :param data: dictionary of different entries of the data set, with the same shape[0]
        :param fn_feed_dict: function to build feed_dict for specific graph, args: data, batch_index
        :param output: the list of outputs needed
        :param session:
        :param batch_size:
        :return:
        """
        data_size = data["feature"].shape[0]
        if batch_size > data_size / 2:
            warnings.warn("too large batch_size %d compared with data_size %d, set to data_size" %
                          (batch_size, data_size))
            batch_size = data_size
        index = np.arange(data_size).astype(np.int64)
        max_batch = math.ceil(float(data_size) / float(batch_size))
        output_result = None
        for i_batch in range(max_batch):
            batch_index = index[i_batch * batch_size: min(data_size, (i_batch + 1) * batch_size)]
            feed_dict = fn_feed_dict(data, batch_index)
            output_batch = session.run(output, feed_dict=feed_dict)
            if output_result is None:
                output_result = output_batch
            else:
                output_result = [np.concatenate([output_result[i_field], output_batch[i_field]])
                                 for i_field in range(len(output_result))]
        return output_result

    @staticmethod
    def setup_session(graph):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        return sess

    def initialize(self, model_import=None, sess=None):
        """
        initialize model
        :param model_import: a global model to use __class__ == Reward
        """
        if model_import is not None:
            raise ValueError("model_import is not implemented")
        else:
            if sess is not None:
                self.sess = sess
            if self.sess is None:
                raise RuntimeError("No tf session assigned")
            self.sess.run(self.init)

    def save(self, save_path="%s/NN" % CKPT_PATH):
        self.saver.save(
            sess=self.sess,
            save_path=save_path
        )

    def restore(self, save_path="%s/NN" % CKPT_PATH):
        self.saver.restore(
            sess=self.sess,
            save_path=save_path
        )

    def partial_restore(self, **kwargs):
        """
        restore subset of graph variables by each saver defined
        """
        pass

    @staticmethod
    def last_relevant(output, length):
        """
        cutoff method for RNN
        """
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @staticmethod
    def fc_layers(input, layer_units, activation, name_pref=""):
        layers = [input]
        for i in range(len(layer_units)):
            layers.append(
                tf.layers.dense(
                    inputs=layers[i],
                    units=layer_units[i],
                    name=name_pref + "_%02d" % i,
                    activation=activation
                )
            )
        return layers, layers[-1]

    @staticmethod
    def scope_name_join(scope_name_parent, scope_name):
        if scope_name_parent == "":
            return scope_name
        else:
            return scope_name_parent + "/" + scope_name

    @staticmethod
    def fixed_input_load(
            input_shape,
            dtype=tf.float32,
            trainable=False,
            name="fixed_input"
    ):
        fixed_input = tf.Variable(
            tf.constant(0, shape=input_shape, dtype=dtype),
            trainable=trainable,
            name=name
        )
        fixed_input_placeholder = tf.placeholder(
            dtype=dtype,
            shape=input_shape
        )
        fixed_input_init = fixed_input.assign(fixed_input_placeholder)
        return fixed_input, fixed_input_placeholder, fixed_input_init

    @staticmethod
    def initialize_fixed_input(
            sess,
            fixed_input_placeholder,
            fixed_input_init,
            fixed_input_file=None,
            fixed_input_data=None
    ):
        if fixed_input_data is None:
            fixed_input_data = cPickle.load(
                open(fixed_input_file, 'rb')
            )
        sess.run(
            fixed_input_init,
            feed_dict={
                fixed_input_placeholder: fixed_input_data
            }
        )


if __name__ == "__main__":
    reward = NN()
