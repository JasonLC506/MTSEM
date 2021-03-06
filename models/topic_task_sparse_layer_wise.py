"""
topic task sparse network
optimizer structure inspired by
  https://github.com/jaehong-yoon93/CGES/blob/master/main.py
"""
import tensorflow as tf
import warnings

from models import SharedBottom
from common import StochasticGradientDescentOptimizer


class TopicTaskSparseLayerWise(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(TopicTaskSparseLayerWise, self).__init__(**kwargs)

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
        with tf.variable_scope(scope):
            # topic task block #
            def weight_topic_task_unflatten(weight_):
                return self.dim_unflatten(
                    weight=weight_,
                    axis=0,                           # weight_.shape = (topic_dim * task_dim, ...)
                    dims=[self.model_spec["topic_dim"], self.task_dim]
                )

            def logits_topic_task_unflatten(logits_):
                return self.dim_unflatten(
                    weight=logits_,
                    axis=1,                            # logits_.shape = (batch_size, topic_dim * task_dim, out_dim)
                    dims=[self.model_spec["topic_dim"], self.task_dim]
                )
            _topic_task_logits_, topic_task_weights, topic_task_biases, topic_task_saver = \
                self._setup_task_specific_block(
                    feature=feature,
                    task_dim=self.model_spec["topic_dim"] * self.task_dim,
                    model_spec=self.model_spec,
                    out_dim=self.label_dim,
                    scope="topic_task_block_"
                )
            # topic_task_logits = logits_topic_task_unflatten(topic_task_logits_)

            # global block #
            _global_logits, global_weights, global_biases, global_saver = self._setup_task_specific_block(
                feature=feature,
                task_dim=1,
                model_spec=self.model_spec,
                out_dim=self.label_dim,
                scope="global_block_"
            )           # with task_dim = 1

            topic_task_logits_ = self._combine_task_specific_block(
                feature=feature,
                weights=[topic_task_weights, global_weights],
                biases=[topic_task_biases, global_biases],
                model_spec=self.model_spec,
                scope="combine_task_specific_block"
            )
            topic_task_logits = logits_topic_task_unflatten(topic_task_logits_)

            # gate block #
            gate_logits_, gate_weights, gate_biases = self.tf_task_specific_dense(
                inputs=tf.expand_dims(feature, axis=1),
                units=self.model_spec["topic_dim"],
                task_dim=1
            )
            gate_logits = self.dim_unflatten(
                weight=gate_logits_,
                axis=1
            )               # same as tf.squeeze(, axis=1)
            gate = tf.nn.softmax(
                gate_logits,
                axis=-1
            )
            self.gate = gate

            # gated_average #
            average_topic_task_logits = tf.einsum(
                "ijkl,ij->ikl",
                topic_task_logits,
                gate
            )    # shape=(batch_size, task_dim, label_dim)
            logits = average_topic_task_logits

            self.topic_task_weight_list = topic_task_weights
            self.topic_task_bias_list = topic_task_biases
            self.global_weight_list = global_weights
            self.global_bias_list = global_biases
            weights = global_weights
            biases = global_biases
            if self.model_spec["topic_dim"] > 1.5:     # only when topic_dim > 1, otherwise zero values cause NaN issue
                weights += gate_weights
                biases += gate_biases

            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
            self._setup_variable_verbose()
        return logits, weights, biases, saver

    def _setup_optim(self):
        warnings.warn("SGD optimizer is restricted")
        with tf.variable_scope("optim"):
            _sgd_optimizer = StochasticGradientDescentOptimizer(
                optim_params=self.model_spec["optim_params"]
            )
            sgd_optimizer = _sgd_optimizer.minimize(self.loss_mean)
            steps = _sgd_optimizer.steps
            learning_rate = _sgd_optimizer.learning_rate
            # updates weight based on sparsity regularization #
            with tf.control_dependencies(
                control_inputs=[sgd_optimizer]
            ):
                additional_optimizer = self._setup_additional_optim(
                    learning_rate=learning_rate
                )
            self.optimizer = tf.group(
                sgd_optimizer, additional_optimizer
            )
            self.steps, self.learning_rate = steps, learning_rate
            self.optim_saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )

    def _setup_regularization(self, **kwargs):
        """
            add entropy of topic penalty so that the topic distribution is sharp for empirical study
            add reconstruction loss of topic
        """
        regularization_loss = super(TopicTaskSparseLayerWise, self)._setup_regularization(**kwargs)

        # entropy regularization #
        if "regularization_topic_entropy" in self.model_spec:  # backward compatible
            # smoothing entropy for numerical stability
            uniform_gate = tf.ones_like(self.gate, dtype=tf.float32) / (1.0 * self.model_spec["topic_dim"])
            gate_smooth = 0.9 * self.gate + 0.1 * uniform_gate
            topic_entropy_loss = - tf.einsum(
                "ij,ij->i",
                gate_smooth,
                tf.math.log(gate_smooth)
            )
            topic_entropy_loss_mean = tf.reduce_mean(topic_entropy_loss)
            topic_entropy_regularization = self.model_spec["regularization_topic_entropy"]  # default 0.1
            regularization_loss += (topic_entropy_regularization * topic_entropy_loss_mean)

        # reconstruction loss #
        if "regularization_reconstruction" in self.model_spec:  # backward compatible
            feature_target = tf.layers.dense(
                self.gate,
                units=self.feature_dim,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                name="feature_target"
            )  # warning: trainable variable defined outside of _setup_net()
            reconstruction_loss = tf.reduce_sum(tf.math.square(feature_target - self.feature), axis=-1)
            reconstruction_loss_mean = tf.reduce_mean(reconstruction_loss)
            reconstruction_regularization = self.model_spec["regularization_reconstruction"]  # default 0.1
            regularization_loss += (reconstruction_regularization * reconstruction_loss_mean)
        return regularization_loss

    def partial_restore(
            self,
            **kwargs
    ):
        super(TopicTaskSparseLayerWise, self).partial_restore(**kwargs)
        if "save_path_optim" in kwargs:
            save_path_optim = kwargs["save_path_optim"]
        else:
            save_path_optim = None
        if save_path_optim is not None:
            self.optim_saver.restore(
                sess=self.sess,
                save_path=save_path_optim
            )
            print("optim retored from %s" % save_path_optim)

    @staticmethod
    def _combine_task_specific_block(
            feature,
            weights,
            biases,
            model_spec,
            scope="combine_task_specific_block"
    ):
        """
        combine task and global block layer-wise
        mimicking the layers in super._setup_task_specific_block
        :param feature:
        :param weights: [task_weights, global_weights]
        :param biases: [task_biases, global_biases]
        :param model_spec:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope):
            task_dim = weights[0][0].shape[0].value
            task_ones = tf.ones(shape=[task_dim])
            feature_task_expanded = tf.einsum(
                "ik,j->ijk",
                feature,
                task_ones
            )
            weights_combined = []
            biases_combined = []
            for i in range(len(weights[0])):
                weights_combined.append(
                    weights[0][i] + weights[1][i]
                )
                biases_combined.append(
                    biases[0][i] + biases[1][i]
                )
            # in the special case only two layers: len(weights_combined) == 2 #
            hidden_a = tf.einsum(
                "ijk,jlk->ijl",
                feature_task_expanded,
                weights_combined[0]
            )
            hidden_a = hidden_a + biases_combined[0]
            if model_spec["activation"] is not None:
                hidden_a = model_spec["activation"](hidden_a)
            hidden_a_dropout = tf.layers.dropout(
                hidden_a,
                rate=model_spec["dropout_task_hidden_a"],
                name="task_hidden_a_dropout"
            )
            logits = tf.einsum(
                "ijk,jlk->ijl",
                hidden_a_dropout,
                weights_combined[1]
            )
            logits = logits + biases_combined[1]
        return logits

    def _setup_variable_verbose(self):
        def weight_reshape_to_norm(weight_origin):
            weight_topic_task = self.dim_unflatten(
                weight=weight_origin,
                axis=0,
                dims=[self.model_spec["topic_dim"], self.task_dim]
            )
            return weight_topic_task

        weights_norm = {}
        weights_norm_l0 = {}
        for weight in self.topic_task_weight_list + self.topic_task_bias_list:
            print("weight in sparsity regularization: %s" % weight.name)
            weight_reshaped = weight_reshape_to_norm(weight)
            weight_norm = tf.norm(
                tf.reshape(weight_reshaped, [self.model_spec["topic_dim"], self.task_dim, -1]),
                ord="euclidean",
                axis=-1
            )
            weights_norm[weight.name] = weight_norm
            weights_norm_l0[weight.name] = tf.math.count_nonzero(weight_norm)
        self.weights_norm = weights_norm
        self.weights_norm_l0 = weights_norm_l0

        weights_norm_global = {}
        for weight in self.global_weight_list + self.global_bias_list:
            weight_norm = tf.norm(
                tf.reshape(weight, [1, -1]),
                ord="euclidean",
                axis=-1
            )
            weights_norm_global[weight.name] = weight_norm
        self.weights_norm_global = weights_norm_global

    def _train_full_split_saver(self):
        op_savers = [self.bottom_saver, self.task_specific_top_saver, self.optim_saver, self.saver]
        save_path_prefixs = list(map(
            lambda x: self.model_name + x,
            ["_bottom", "_task_specific_top", "_optim", ""]
        ))
        return op_savers, save_path_prefixs

    def _setup_additional_optim(
            self,
            **kwargs
    ):
        def weight_reshape_to_norm(weight_origin):
            weight_topic_task = self.dim_unflatten(
                weight=weight_origin,
                axis=0,
                dims=[self.model_spec["topic_dim"], self.task_dim]
            )
            return weight_topic_task

        def weight_reshape_from_norm(weight):
            weight_origin = self.dims_flatten(
                weight=weight,
                axes=(0, 2)
            )
            return weight_origin

        op_proximal_operator = self._setup_proximal_operator(
            weight_list=self.topic_task_weight_list + self.topic_task_bias_list,
            learning_rate=kwargs["learning_rate"],
            regularization_lambda=self.model_spec["sparse_regularization_lambda"],
            weight_reshape_to_norm=weight_reshape_to_norm,
            weight_reshape_from_norm=weight_reshape_from_norm
        )
        return op_proximal_operator

    @staticmethod
    def proximal_operator(
            weight,
            **kwargs
    ):
        return proximal_operator_element_sparse(
            weight=weight,
            **kwargs
        )

    def _setup_proximal_operator(
            self,
            weight_list,
            learning_rate,
            regularization_lambda,
            reciprocal_stable_factor=0.0001,
            weight_reshape_to_norm=lambda x: x,
            weight_reshape_from_norm=lambda x: x
    ):
        """
        apply (2, 1, 1) norm to weights (or biases) in weight_list,
        :param weight_list: [weight], weight.shape=(topic_dim * task_dim, ...)
        :param learning_rate: latest learning rate
        :param regularization_lambda: regularization strength
        :param reciprocal_stable_factor: avoid a/0 unstable
        :param weight_reshape_to_norm: reshape weight to (topic_dim, task_dim, ...)
        :param weight_reshape_from_norm: reshape weight back to (topic_dim * task_dim) for tf.assign
        :return:
        """
        eta = learning_rate * regularization_lambda
        epsilon = eta * reciprocal_stable_factor
        weight_update_ops = []
        weight_shapes = []
        weight_reshaped_list = []
        weight_reshaped_shapes = []
        for weight_origin in weight_list:
            weight = weight_reshape_to_norm(weight_origin)
            weight_shape = list(map(
                lambda x: x.value,
                weight.shape
            ))
            weight_shapes.append(weight_shape)
            weight_reshaped = tf.reshape(
                weight,
                shape=(weight_shape[0], weight_shape[1], -1)
            )
            weight_reshaped_list.append(weight_reshaped)
            weight_reshaped_shapes.append(
                list(map(lambda x: x.value, weight_reshaped.shape))
            )
        weight_reshaped_combined = tf.concat(
            values=weight_reshaped_list,
            axis=-1
        )
        # proximal update #
        weight_new_reshaped_combined = self.proximal_operator(
            weight=weight_reshaped_combined,
            eta=eta,
            epsilon=epsilon
        )

        weight_new_reshaped_list = tf.split(
            value=weight_new_reshaped_combined,
            num_or_size_splits=list(map(lambda x: x[-1], weight_reshaped_shapes)),
            axis=-1
        )
        for i in range(len(weight_new_reshaped_list)):
            weight_new_reshaped = weight_new_reshaped_list[i]
            weight_shape = weight_shapes[i]
            weight_origin = weight_list[i]
            weight_new = tf.reshape(
                weight_new_reshaped,
                shape=weight_shape,
            )
            weight_origin_new = weight_reshape_from_norm(weight_new)
            weight_update_op = weight_origin.assign(weight_origin_new)
            weight_update_ops.append(weight_update_op)
        return tf.group(*weight_update_ops)

    @staticmethod
    def dim_unflatten(
            weight,
            axis=0,
            dims=[]
    ):
        dim_total = 1
        for dim in dims:
            dim_total *= dim
        weight_shape = list(map(
            lambda x: x.value if x.value is not None else -1,
            weight.shape
        ))
        assert dim_total == weight_shape[axis]
        weight_shape_new = weight_shape[:axis] + dims + weight_shape[axis + 1:]
        weight_reshaped = tf.reshape(
            weight,
            shape=weight_shape_new
        )
        return weight_reshaped

    @staticmethod
    def dims_flatten(
            weight,
            axes=(0, 1)
    ):
        """
        :param weight:
        :param axes: range of axes to be flatten
        """
        weight_shape = list(map(
            lambda x: x.value if x.value is not None else -1,
            weight.shape
        ))
        new_dim = 1
        for axis in range(axes[0], axes[1]):
            new_dim *= weight_shape[axis]
        weight_shape_new = weight_shape[:axes[0]] + [new_dim] + weight_shape[axes[1]:]
        weight_reshaped = tf.reshape(
            weight,
            shape=weight_shape_new
        )
        return weight_reshaped

    def _op_epoch_verbose(self):
        return [
            self.weights_norm,
            self.weights_norm_l0,
            self.weights_norm_global
        ]
        ## for logging memory issue ##
        # return None


def proximal_operator_element_sparse(
        weight,
        eta,
        epsilon,
        **kwargs
):
    """
    calculate proximal operator for element sparse regularization
    :param weight: shape=[..., inner_dim]
    :param eta: learning_rate * regularization_lambda
    :param epsilon: stabilisation term
    :param kwargs:
    :return:
    """
    weight_inner_norm = tf.norm(
        weight,
        ord='euclidean',
        axis=-1,
        keepdims=True
    )
    weight_inner_norm_stabled = weight_inner_norm + epsilon
    weight_new = tf.clip_by_value(
        1.0 - eta / weight_inner_norm_stabled,
        clip_value_min=0.0,
        clip_value_max=1.0  # no effect
    ) * weight
    return weight_new
