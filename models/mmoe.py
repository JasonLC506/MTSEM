"""
multi-gate mixture-of-expert network
@inproceedings{ma2018modeling,
  title={Modeling task relationships in multi-task learning with multi-gate mixture-of-experts},
  author={Ma, Jiaqi and Zhao, Zhe and Yi, Xinyang and Chen, Jilin and Hong, Lichan and Chi, Ed H},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1930--1939},
  year={2018},
  organization={ACM}
}
Extended from original work~\cite{ma2018modeling}:
    1. shared-bottom structure is used for minimum comparison
"""
import tensorflow as tf
import warnings

from models import SharedBottom


class MMoE(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(MMoE, self).__init__(**kwargs)

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
        weights, biases = [], []
        with tf.variable_scope(scope):
            # experts #
            feature_expert_expanded = self._feature_task_expand(
                feature=feature,
                task_dim=self.model_spec["n_experts"]
            )
            experts, experts_weights, experts_biases = self.tf_task_specific_dense(
                inputs=feature_expert_expanded,
                units=self.model_spec["expert_dimension"],
                task_dim=self.model_spec["n_experts"],
                activation=self.model_spec["activation"],
                name="experts"
            )
            weights += experts_weights
            biases += experts_biases

            # gates #
            feature_task_expanded = self._feature_task_expand(
                feature=feature,
                task_dim=self.task_dim
            )
            gates, gates_weights, gates_biases = self.tf_task_specific_dense(
                inputs=feature_task_expanded,
                units=self.model_spec["n_experts"],
                task_dim=self.task_dim,
                activation=None,
                name="gates"
            )
            gates_normalized = tf.nn.softmax(
                gates,
                axis=-1
            )
            weights += gates_weights
            biases += gates_biases

            # task_hidden_a #
            # (batch_size, n_experts, expert_dimension),(batch_size, task_dim, n_experts)->
            # (batch_size, task_dim, expert_dim)
            hidden_a = tf.einsum(
                "ijk,ilj->ilk",
                experts,
                gates_normalized,
                name="hidden_a"
            )
            if hidden_a.shape[-1].value != self.model_spec["task_hidden_a_dimension"]:
                warnings.warn(
                    "hidden_a shape != 'task_hidden_a_dimension' in config, %d!=%d" % (
                        hidden_a.shape[-1].value,
                        self.model_spec["task_hidden_a_dimension"]
                    )
                )
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
    def _feature_task_expand(
            feature,
            task_dim
    ):
        task_ones = tf.ones(shape=[task_dim])
        feature_task_expanded = tf.einsum(
            "ik,j->ijk",
            feature,
            task_ones
        )
        return feature_task_expanded
