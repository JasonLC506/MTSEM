"""
topic task sparse network
optimizer structure inspired by
  https://github.com/jaehong-yoon93/CGES/blob/master/main.py
"""
import tensorflow as tf
import warnings

from models import SharedBottom
from common import StochasticGradientDescentOptimizer


class TopicTaskSparse(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(TopicTaskSparse, self).__init__(**kwargs)

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
            topic_task_logtis_, topic_task_weights, topic_task_biases, topic_task_saver = \
                self._setup_task_specific_block(
                    feature=feature,
                    task_dim=self.model_spec["topic_dim"] * self.task_dim,
                    model_spec=self.model_spec,
                    out_dim=self.label_dim,
                    scope="topic_task_block_"
                )
            topic_task_logits = logits_topic_task_unflatten(topic_task_logtis_)

            # global block #
            global_logits, global_weights, global_biases, global_saver = self._setup_task_specific_block(
                feature=feature,
                task_dim=1,
                model_spec=self.model_spec,
                out_dim=self.label_dim,
                scope="global_block_"
            )           # with task_dim = 1

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

            # gated_average #
            average_topic_task_logits = tf.einsum(
                "ijkl,ij->ikl",
                topic_task_logits,
                gate
            )    # shape=(batch_size, task_dim, label_dim)
            logits = average_topic_task_logits + self.model_spec["global_block_weight"] * global_logits

            self.topic_task_weight_list = topic_task_weights
            self.topic_task_bias_list = topic_task_biases
            weights = topic_task_weights + global_weights + gate_weights
            biases = topic_task_biases + global_biases + gate_biases

            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
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

    def partial_restore(
            self,
            **kwargs
    ):
        super(TopicTaskSparse, self).partial_restore(**kwargs)
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

    def _train_full_split_saver(self):
        op_savers = [self.saver, self.bottom_saver, self.task_specific_top_saver, self.optim_saver]
        save_path_prefixs = list(map(
            lambda x: self.model_name + x,
            ["", "_bottom", "_task_specific_top", "_optim"]
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
    def _setup_proximal_operator(
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
        for weight_origin in weight_list:
            weight = weight_reshape_to_norm(weight_origin)
            weight_shape = list(map(
                lambda x: x.value,
                weight.shape
            ))
            weight_reshaped = tf.reshape(
                weight,
                shape=(weight_shape[0], weight_shape[1], -1)
            )
            weight_inner_norm = tf.norm(
                weight_reshaped,
                ord='euclidean',
                axis=-1,
                keepdims=True
            )
            weight_inner_norm_stabled = weight_inner_norm + epsilon
            weight_new_reshaped = tf.clip_by_value(
                1.0 - eta / weight_inner_norm_stabled,
                clip_value_min=0.0,
                clip_value_max=1.0             # no effect
            ) * weight_reshaped
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
