"""
single model for all tasks
"""
import tensorflow as tf
import numpy as np

from models import NN

Optimizer = tf.train.AdamOptimizer


class FC(NN):
    def __init__(
            self,
            feature_dim,
            label_dim,
            task_dim,
            model_spec,
            model_name=None
    ):
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.task_dim = task_dim
        self.model_spec = model_spec
        if model_name is None:
            self.model_name = model_spec["name"]
        else:
            self.model_name = model_name

        super(FC, self).__init__(graph=None)

    def initialization(self):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)

    def _setup_placeholder(self):
        with tf.name_scope("placeholder"):
            self.feature = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.feature_dim],
                name="feature"
            )
            self.label = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.label_dim],
                name="label"
            )
            self.weight = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name="weight"
            )
            self.task = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.task_dim],
                name="task"
            )

    def _setup_net(self):
        feature_dropout = tf.layers.dropout(
            self.feature,
            rate=self.model_spec["dropout_feature"],
            name="feature_dropout"
        )
        hidden = tf.layers.dense(
            feature_dropout,
            units=self.model_spec["hidden_dimension"],
            activation=self.model_spec["activation"],
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="hidden"
        )
        hidden_dropout = tf.layers.dropout(
            hidden,
            rate=self.model_spec["dropout_hidden"],
            name="hidden_dropout"
        )
        logits = tf.layers.dense(
            hidden_dropout,
            units=self.label_dim,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name="logits"
        )
        self.logits = logits
        self.label_pred = tf.nn.softmax(
            logits=logits,
            axis=-1,
            name="softmax"
        )

    def _setup_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.logits,
            name="cross_entropy"
        )
        # task-specific loss #
        weight_task = tf.expand_dims(self.weight, axis=-1) * self.task
        self.loss_cross_entropy_task = tf.tensordot(cross_entropy, weight_task, axes=[0, 0])
        self.weight_sum_task = tf.reduce_sum(weight_task, axis=0)  # for average calculation
        # sum-up to total loss
        self.loss_cross_entropy = tf.reduce_sum(self.loss_cross_entropy_task)
        self.loss = self.loss_cross_entropy

    def _setup_optim(self):
        # TODO:: using BERT optimizer with warm-up and weight decay
        self.optimizer = Optimizer(
            learning_rate=self.model_spec["learning_rate"],
            epsilon=1e-06,
            name="optimizer"
        ).minimize(self.loss)

    def train(
            self,
            data_generator,
            data_generator_valid=None
    ):
        """
        :param data_generator: data_batched: {
                    "feature":
                    "label"
        }
        :param data_generator_valid: valid data generator
        """
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss, self.loss_cross_entropy_task],
            session=self.sess,
            op_data_size=self.weight_sum_task,
            batch_size=self.model_spec["batch_size"],
            max_epoch=self.model_spec["max_epoch"],
            data_generator_valid=data_generator_valid,
            op_savers=[self.saver],
            save_path_prefixs=[self.model_name],
            log_board_dir="../summary/" + self.model_name
        )
        return results

    def _fn_feed_dict_train(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.feature: data["feature"][batch_index],
            self.label: data["label"][batch_index],
            self.weight: data["weight"][batch_index],
            self.task: data["task"][batch_index]
        }
        return feed_dict

    def test(
            self,
            data_generator
    ):
        """
        :param data_generator: data_batched: {
                    "feature":
                    "label"
        }
        """
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.loss,
            op_losses=[self.loss, self.loss_cross_entropy_task],
            session=self.sess,
            op_data_size=self.weight_sum_task,
            batch_size=self.model_spec["batch_size"],
            max_epoch=1,
            data_generator_valid=None,
            op_savers=None,
            save_path_prefixs=[self.model_name]
        )
        return results

    def predict(
            self,
            data_generator
    ):
        # TODO:: predict distributions for new instances
        raise NotImplementedError
