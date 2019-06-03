"""
separate model for each task
"""
import tensorflow as tf

from models import SharedBottom

class Separate(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(Separate, self).__init__(**kwargs)

    def _setup_net(self):
        feature_extracted_tasks = []
        for t in range(self.task_dim):
            feature_extracted_per, _ = self._setup_bottom(
                feature=self.feature,
                model_spec=self.model_spec["bottom"],
                scope="bottom_%d" % t
            )
            feature_extracted_tasks.append(feature_extracted_per)
        feature_extracted = tf.stack(feature_extracted_tasks, axis=1)
        self.bottom_saver = None

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

    @staticmethod
    def _setup_task_specific_block(
            feature,
            task_dim,
            model_spec,
            out_dim,
            out_activation=None,
            scope="task_specific_block"
    ):
        with tf.variable_scope(scope):
            # # expand shared feature to task_dim #
            # task_ones = tf.ones(shape=[task_dim])
            # feature_task_expanded = tf.einsum(
            #     "ik,j->ijk",
            #     feature,
            #     task_ones
            # )
            feature_task_expanded = feature  # already expanded

            weights, biases = [], []
            # hidden layers #
            hidden_a, hidden_a_weights, hidden_a_biases = SharedBottom.tf_task_specific_dense(
                inputs=feature_task_expanded,
                units=model_spec["task_hidden_a_dimension"],
                task_dim=task_dim,
                activation=model_spec['activation'],
                name="hidden_a"
            )
            weights += hidden_a_weights
            biases += hidden_a_biases
            hidden_a_dropout = tf.layers.dropout(
                hidden_a,
                rate=model_spec["dropout_task_hidden_a"],
                name="task_hidden_a_dropout"
            )
            # logit layer #
            logits, logits_weights, logits_biases = SharedBottom.tf_task_specific_dense(
                inputs=hidden_a_dropout,
                units=out_dim,
                task_dim=task_dim,
                activation=out_activation,
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

