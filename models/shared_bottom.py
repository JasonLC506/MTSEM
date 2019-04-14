"""
shared-bottom network
to be used as super-class for other NN-based MTL networks
"""
import tensorflow as tf

from models import NN

Optimizer = tf.train.AdamOptimizer


class SharedBottom(NN):
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

        super(SharedBottom, self).__init__(graph=None)

    def initialization(self):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)

    def partial_restore(
            self,
            save_path_bottom=None,
            save_path_task_specific_top=None,
            **kwargs
    ):
        if self.sess is None:
            self.initialization()
        if save_path_bottom is not None:
            self.bottom_saver.restore(
                sess=self.sess,
                save_path=save_path_bottom
            )
            print("bottom restored from %s" % save_path_bottom)
        if save_path_task_specific_top is not None:
            self.task_specific_top_saver.restore(
                sess=self.sess,
                save_path=save_path_task_specific_top
            )
            print("task_specific_top restored from %s" % save_path_task_specific_top)

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
            )                                         # TODO: for the case task_dim large, sparse design is needed

    def _setup_net(self):
        feature_extracted, self.bottom_saver = self._setup_bottom(
            feature=self.feature,
            model_spec=self.model_spec["bottom"],
            scope="bottom"
        )

        self.task_logits, self.task_weight_list, self.task_bias_list, self.task_specific_top_saver = \
            self._setup_task_specific_top(
                feature=feature_extracted,
                scope="task_specific_top"
            )

        self.logits = tf.einsum(
            "ijk,ij->ik",
            self.task_logits,
            self.task
        )
        self.label_pred = tf.nn.softmax(
            logits=self.logits,
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
        self.regularization_loss = self._setup_regularization()
        self.loss = self.loss_cross_entropy + self.regularization_loss

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
            data_generator_valid=None,
            full_split_saver=False
    ):
        """
        :param data_generator: data_batched: {
                    "feature":
                    "label"
        }
        :param data_generator_valid: valid data generator
        :param full_split_saver: if save all network splits separately
        """
        if full_split_saver:
            op_savers, save_path_prefixs = self._train_full_split_saver()
        else:
            op_savers = [self.saver]
            save_path_prefixs = [self.model_name]
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss, self.loss_cross_entropy, self.loss_cross_entropy_task],
            session=self.sess,
            op_data_size=self.weight_sum_task,
            batch_size=self.model_spec["batch_size"],
            max_epoch=self.model_spec["max_epoch"],
            data_generator_valid=data_generator_valid,
            op_savers=op_savers,
            save_path_prefixs=save_path_prefixs,
            log_board_dir="../summary/" + self.model_name
        )
        return results

    def _train_full_split_saver(self):
        op_savers = [self.saver, self.bottom_saver, self.task_specific_top_saver]
        save_path_prefixs = list(map(
            lambda x: self.model_name + x,
            ["", "_bottom", "_task_specific_top"]
        ))
        return op_savers, save_path_prefixs

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

    def predict(
            self,
            data_generator
    ):
        results = self._feed_forward_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_predict,
            output=[self.label_pred],
            session=self.sess,
            batch_size=self.model_spec["batch_size"]
        )[0]
        return results

    def _fn_feed_dict_predict(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.feature: data["feature"][batch_index],
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
            op_losses=[self.loss_cross_entropy, self.loss_cross_entropy_task],
            session=self.sess,
            op_data_size=self.weight_sum_task,
            batch_size=self.model_spec["batch_size"],
            max_epoch=1,
            data_generator_valid=None,
            op_savers=None,
            save_path_prefixs=[self.model_name]
        )
        return results

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
        """
        :param feature: shape=(batch_size, feature_dim)
        :return:
        """
        with tf.variable_scope(scope):
            # expand shared feature to task_dim #
            task_ones = tf.ones(shape=[self.task_dim])
            feature_task_expanded = tf.einsum(
                "ik,j->ijk",
                feature,
                task_ones
            )

            weights, biases = [], []
            # hidden layers #
            hidden_a, hidden_a_weights, hidden_a_biases = self.tf_task_specific_dense(
                inputs=feature_task_expanded,
                units=self.model_spec["task_hidden_a_dimension"],
                task_dim=self.task_dim,
                activation=self.model_spec['activation'],
                name="hidden_a"
            )
            weights += hidden_a_weights
            biases += hidden_a_biases
            hidden_a_dropout = tf.layers.dropout(
                hidden_a,
                rate=self.model_spec["dropout_task_hidden_a"],
                name="task_hidden_a_dropout"
            )
            # logit layer #
            logits, logits_weights, logits_biases = self.tf_task_specific_dense(
                inputs=hidden_a_dropout,
                units=self.label_dim,
                task_dim=self.task_dim,
                activation=None,
                name="logits"
            )
            weights += logits_weights
            biases += logits_biases

            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
        return logits, weights, biases, saver

    def _setup_regularization(self):
        if "regularization_l2" in self.model_spec:
            regularization_loss = 0.0
            for weight in self.task_weight_list:
                regularization_loss += tf.pow(tf.norm(
                    weight,
                    ord='euclidean'
                ), 2.0)
            regularization_loss = self.model_spec["regularization_l2"] * regularization_loss
        else:
            regularization_loss = 0.0
        return regularization_loss

    @staticmethod
    def _setup_bottom(
            feature,
            model_spec,
            scope="bottom"
    ):
        with tf.variable_scope(scope):
            feature_dropout = tf.layers.dropout(
                feature,
                rate=model_spec["dropout_feature"],
                name="feature_dropout"
            )
            hidden = tf.layers.dense(
                feature_dropout,
                units=model_spec["hidden_dimension"],
                activation=model_spec["activation"],
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name="hidden"
            )
            hidden_dropout = tf.layers.dropout(
                hidden,
                rate=model_spec["dropout_hidden"],
                name="hidden_dropout"
            )
            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
        return hidden_dropout, saver

    @staticmethod
    def tf_task_specific_dense(
            inputs,
            units,
            task_dim,
            activation=None,
            use_bias=True,
            kernel_initial_value=0.01,
            bias_initial_value=0.0001,
            trainable=True,
            name=None
    ):
        """
        :param inputs: shape=(batch_size, task_dim, input_dim)
        :param units: int or long
        :param task_dim: int
        :param activation:
        :param use_bias:
        :param kernel_initial_value:
        :param bias_initial_value:
        :param trainable:
        :param name:
        :return:
        """
        # weight initial value #
        elaborate_weight_initial_value = False
        if isinstance(kernel_initial_value, float):
            mean = 0.0
            stddev = kernel_initial_value
        elif isinstance(kernel_initial_value, tuple):
            mean, stddev = kernel_initial_value
        else:
            elaborate_weight_initial_value = True
        if not elaborate_weight_initial_value:
            weight_initial_value = tf.random.normal(
                shape=[task_dim, units, inputs.shape[-1].value],
                mean=mean,
                stddev=stddev
            )
        else:
            weight_initial_value = kernel_initial_value

        weight = tf.Variable(
            initial_value=weight_initial_value,
            dtype=tf.float32,
            name=None if name is None else name + "_weight",
            trainable=trainable
        )
        bias = tf.Variable(
            initial_value=tf.random.normal(
                shape=(task_dim, units),
                mean=0.0,
                stddev=bias_initial_value
            ),
            dtype=tf.float32,
            name=None if name is None else name + "_bias",
            trainable=trainable
        )                                           # not using zero initialization to avoid norm NaN optimization issue
        hidden = tf.einsum(
            "ijk,jlk->ijl",
            inputs,
            weight
        )        # (batch_size, task_dim, input_dim),(task_dim, units, inputs.shape[-1])->(batch_size, task_dim, units)
        if use_bias:
            hidden = hidden + bias
        if activation:
            hidden = activation(hidden)
        return hidden, [weight], [bias]
