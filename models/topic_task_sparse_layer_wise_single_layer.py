"""
topic task sparse network
optimizer structure inspired by
  https://github.com/jaehong-yoon93/CGES/blob/master/main.py
"""
import tensorflow as tf
import warnings

from models import TopicTaskSparseLayerWise
from common import StochasticGradientDescentOptimizer


class TopicTaskSparseLayerWiseSingleLayer(TopicTaskSparseLayerWise):
    def __init__(
            self,
            **kwargs
    ):
        super(TopicTaskSparseLayerWiseSingleLayer, self).__init__(**kwargs)

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
            topic_task_dim = self.model_spec["topic_dim"] * self.task_dim
            feature_topic_task_expanded = tf.einsum(
                "ik,j->ijk",
                feature,
                tf.ones(shape=[topic_task_dim])
            )
            _topic_task_hidden_, topic_task_weights, topic_task_biases = self.tf_task_specific_dense(
                inputs=feature_topic_task_expanded,
                units=self.model_spec["task_hidden_a_dimension"],
                task_dim=self.model_spec["topic_dim"] * self.task_dim,
                activation=self.model_spec["activation"],
                name="topic_task_hidden_a"
            )
            # topic_task_logits = logits_topic_task_unflatten(topic_task_logits_)

            # global block #
            _global_hidden, global_weights, global_biases = self.tf_task_specific_dense(
                inputs=tf.expand_dims(
                    feature,
                    axis=1
                ),
                units=self.model_spec["task_hidden_a_dimension"],
                task_dim=1,
                activation=self.model_spec["activation"],
                name="global_hidden_a"
            )

            topic_task_hidden_ = self._combine_task_specific_block(
                feature=feature,
                weights=[topic_task_weights, global_weights],
                biases=[topic_task_biases, global_biases],
                model_spec=self.model_spec,
                scope="combine_topic_task_block"
            )
            topic_task_hidden = logits_topic_task_unflatten(topic_task_hidden_)

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
            average_topic_task_hidden = tf.einsum(
                "ijkl,ij->ikl",
                topic_task_hidden,
                gate
            )    # shape=(batch_size, task_dim, hidden_dim)
            hidden_a = average_topic_task_hidden
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

            self.topic_task_weight_list = topic_task_weights
            self.topic_task_bias_list = topic_task_biases
            self.global_weight_list = global_weights
            self.global_bias_list = global_biases
            weights = topic_task_weights + global_weights + gate_weights + logits_weights
            biases = topic_task_biases + global_biases + gate_biases + logits_biases

            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
            self._setup_variable_verbose()
        return logits, weights, biases, saver

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
        return hidden_a_dropout
