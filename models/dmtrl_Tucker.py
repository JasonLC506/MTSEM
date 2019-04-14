"""
Deep Multi-Task Representation Learning with Tucker Decomposition
@article{yang2016deep,
  title={Deep multi-task representation learning: A tensor factorisation approach},
  author={Yang, Yongxin and Hospedales, Timothy},
  journal={arXiv preprint arXiv:1605.06391},
  year={2016}
}
integrating the interactive training into end-to-end
"""
import tensorflow as tf
import re
import numpy as np
import warnings

from models import SharedBottom
from model_dependency import tensor_producer


class DmtrlTucker(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        self.primary_model = SharedBottom(
            **kwargs
        )
        tf.reset_default_graph()               # not sure needed but to be safe for now
        self.primary_model.initialization()
        self.primary_model.restore(
            save_path=kwargs['model_spec']['primary_model_ckpt']
        )

        super(DmtrlTucker, self).__init__(**kwargs)

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
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
            hidden_a, hidden_a_weights, hidden_a_biases = self.tf_task_specific_dense_using_primary_model(
                primary_model=self.primary_model,
                inputs=feature_task_expanded,
                units=self.model_spec["task_hidden_a_dimension"],
                task_dim=self.task_dim,
                activation=self.model_spec['activation'],
                weight_name_pattern=r'.*task_specific_top/hidden_a_weight',
                tensor_decomposition_method=self.model_spec['tensor_decomposition_method'],
                tensor_decomposition_eps_or_k=self.model_spec["tensor_decomposition_eps_or_k"]
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

    @staticmethod
    def tf_task_specific_dense_using_primary_model(
            primary_model,
            inputs,
            units,
            task_dim,
            activation=None,
            use_bias=True,
            kernel_initial_value=0.01,
            bias_initial_value=0.0001,
            name=None,
            trainable=True,
            weight_name_pattern=r".*task_specific_top/hidden_a_weight",
            tensor_decomposition_method="Tucker",
            tensor_decomposition_eps_or_k=0.01
    ):
        """
        weight as Tucker decomposition defined and initialized by primary model
        """
        if not trainable:
            raise NotImplementedError("variable initialized with primary model must be trainable")
        warnings.warn(
            "in tf_task_specific_dense_using_primary_model(), args: kernel_initial_value, name are ineffective"
        )
        # hidden layers #
        #     factorized hidden layer
        weight_primary_model = None
        for weight_variable_primary_model in primary_model.task_weight_list:
            if re.match(weight_name_pattern, weight_variable_primary_model.name):
                weight_primary_model = weight_variable_primary_model
                break
        if weight_primary_model is None:
            raise RuntimeError(
                "weights_primary_model not found in variable list: %s with name pattern '%s'" % (
                    "\n".join(
                        list(map(
                            lambda x: x.name,
                            primary_model.task_weight_list
                        ))
                    ),
                    str(weight_name_pattern)
                )
            )
        weight_primary_model_value = primary_model.sess.run(
            weight_primary_model
        )

        weight_transposed, weight_factors = tensor_producer(
            np.transpose(
                weight_primary_model_value,
                axes=(1, 2, 0)
            ),          # according to the demo in original version, decomposition takes task as final dimension
            method=tensor_decomposition_method,
            eps_or_k=tensor_decomposition_eps_or_k,
            datatype=np.float32,
            return_true_var=True
        )
        weight_true_factors = weight_factors["U"] + [weight_factors["S"]]
        for w in weight_true_factors:
            print("w: '%s' with shape %s" % (w.name, w.shape))

        weight = tf.transpose(
            weight_transposed,
            perm=(2, 0, 1)
        )
        bias = tf.Variable(
            initial_value=tf.random.normal(
                shape=(task_dim, units),
                mean=0.0,
                stddev=bias_initial_value
            )
        )
        hidden = tf.einsum(
            "ijk,jlk->ijl",
            inputs,
            weight
        )
        if use_bias:
            hidden = hidden + bias
        if activation:
            hidden = activation(hidden)
        return hidden, weight_true_factors, [bias]
