"""
cross-stitch network
@inproceedings{misra2016cross,
  title={Cross-stitch networks for multi-task learning},
  author={Misra, Ishan and Shrivastava, Abhinav and Gupta, Abhinav and Hebert, Martial},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3994--4003},
  year={2016}
}
simplest version:
    1. Different from the implementation
     https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning
      (c07edb758aad7e0a2eb8da82e63105eae2ef77a4)
    where cross-stitch units are shared for different layers, which enforce the equal dimensions over different layers,
    it is simply removed, that is, here each cross-stitch unit is independent
    2. without learning rates scaling for cross-stitch units
    3. initialization with task-specific models is optional
"""
import tensorflow as tf

from models import SharedBottom


class CrossStitch(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(CrossStitch, self).__init__(**kwargs)

    def _setup_task_specific_top(
            self,
            feature,
            scope="task_specific_top"
    ):
        weights, biases, cross_stitch_units = [], [], []
        with tf.variable_scope(scope):
            # expand shared feature to task_dim #
            task_ones = tf.ones(shape=[self.task_dim])
            feature_task_expanded = tf.einsum(
                "ik,j->ijk",
                feature,
                task_ones
            )

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

        with tf.variable_scope(scope + "_cross_stitch_units"):
            stitch_a, cross_stitch_units_a = self.apply_cross_stitch(
                inputs=hidden_a_dropout,
                name="cross_stitch_units_a"
            )
            cross_stitch_units += cross_stitch_units_a

        # logit layer #
        with tf.variable_scope(scope):
            logits, logits_weights, logits_biases = self.tf_task_specific_dense(
                inputs=stitch_a,
                units=self.label_dim,
                task_dim=self.task_dim,
                activation=None,
                name="logits"
            )
            weights += logits_weights
            biases += logits_biases

        with tf.variable_scope(scope):
            saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
        with tf.variable_scope(scope + "_cross_stitch_units"):
            cross_stitch_units_saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
            self.cross_stitch_units_saver = cross_stitch_units_saver
            self.cross_stitch_units = cross_stitch_units
        return logits, weights, biases, saver

    @staticmethod
    def apply_cross_stitch(
            inputs,
            name="cross_stitch_units"
    ):
        inputs_shape = list(map(
            lambda x: x.value,
            inputs.shape
        ))                        # shape=(batch_size, task_dim, ...)
        inputs_flat = tf.contrib.layers.flatten(
            inputs=inputs,
        )

        # initialize with identity matrix
        cross_stitch_units = tf.Variable(
            initial_value=tf.eye(
                num_rows=inputs_flat.shape[1].value
            ),
            name=name,
        )
        output_flat = tf.matmul(inputs_flat, cross_stitch_units)

        # need to call .value to convert Dimension objects to normal value
        output = tf.reshape(
            output_flat,
            shape=[-1] + inputs_shape[1:]
        )
        return output, [cross_stitch_units]

    def _train_full_split_saver(self):
        op_savers = [self.saver, self.bottom_saver, self.task_specific_top_saver, self.cross_stitch_units_saver]
        save_path_prefixs = list(map(
            lambda x: self.model_name + x,
            ["", "_bottom", "_task_specific_top", "_cross_stitch_units"]
        ))
        return op_savers, save_path_prefixs

    def partial_restore(
            self,
            save_path_bottom=None,
            save_path_task_specific_top=None,
            save_path_cross_stitch_units=None
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
        if save_path_cross_stitch_units is not None:
            self.cross_stitch_units_saver.restore(
                sess=self.sess,
                save_path=save_path_cross_stitch_units
            )
            print("cross_stitch_units restored from %s" % save_path_cross_stitch_units)
