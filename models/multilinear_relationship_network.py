"""
multilinear relationship network
@incollection{NIPS2017_6757,
    title = {Learning Multiple Tasks with Multilinear Relationship Networks},
    author = {Long, Mingsheng and CAO, ZHANGJIE and Wang, Jianmin and Yu, Philip S},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {1593--1602},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/6757-learning-multiple-tasks-with-multilinear-relationship-networks.pdf}
}
all task specific weights are multilinear regularized
"""
import tensorflow as tf

from models import SharedBottom


class MultilinearRelationshipNetwork(SharedBottom):
    def __init__(
            self,
            **kwargs
    ):
        super(MultilinearRelationshipNetwork, self).__init__(**kwargs)

    def partial_restore(
            self,
            **kwargs
    ):
        super(MultilinearRelationshipNetwork, self).partial_restore(**kwargs)
        save_path_regularization = kwargs["save_path_regularization"]
        if save_path_regularization is not None:
            self.regularization_saver.restore(
                sess=self.sess,
                save_path=save_path_regularization
            )
            print("regularization retored from %s" % save_path_regularization)

    def _train_full_split_saver(self):
        op_savers = [self.bottom_saver, self.task_specific_top_saver, self.regularization_saver, self.saver]
        save_path_prefixs = list(map(
            lambda x: self.model_name + x,
            ["_bottom", "_task_specific_top", "_regularization", ""]
        ))
        return op_savers, save_path_prefixs

    def _fn_op_update_train(self, steps):
        """
        update the regularization variables
        """
        if (steps + 1) % self.model_spec["regularization_variable_update_frequency"] == 0:
            return self.op_regularization_variable_update
        else:
            return None

    def _setup_regularization(
            self,
            scope="multilinear_norm"
    ):
        # original l2 regularization #
        regularization_loss = super(MultilinearRelationshipNetwork, self)._setup_regularization()

        # multilinear norm #
        with tf.variable_scope(scope):
            norm_losses, cov_invs, op_cov_inv_updates = 0.0, {}, []
            for weight in self.task_weight_list:
                norm_loss, covariance_invs, op_covariance_inv_updates = self._multilinear_regularization(
                    weight=weight
                )
                norm_losses += norm_loss
                cov_invs[weight.name] = covariance_invs
                op_cov_inv_updates += op_covariance_inv_updates
            regularization_loss += self.model_spec["regularization_multilinear_norm"] * norm_losses
            self.cov_invs = cov_invs
            self.op_regularization_variable_update = tf.group(*op_cov_inv_updates)
            self.regularization_saver = tf.train.Saver(
                var_list=tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name
                ),
                max_to_keep=1000
            )
        return regularization_loss

    def _multilinear_regularization(
            self,
            weight
    ):
        weight_shape = list(map(lambda x: x.value, weight.shape))
        weight_dim = len(weight_shape)
        covariance_invs = [
            tf.Variable(
                tf.eye(num_rows=weight_shape[i]),
                trainable=False
            ) for i in range(weight_dim)
        ]

        # regularization loss for weight #
        regularization_loss = self.multilinear_norm(
            weight=weight,
            covariance_invs=covariance_invs,
            weight_shape=weight_shape
        )

        # updates for covariance_invs #
        covariance_temps = [
            self.multilinear_partial_norm(
                weight=weight,
                covariance_invs=covariance_invs,
                axis=i,
                epsilon=self.model_spec['covariance_epsilon'],
                weight_shape=weight_shape
            ) for i in range(weight_dim)
        ]
        covariance_inv_temps = [
            self.matrix_inverse_generalized(
                matrix=covariance_temps[i],
                singular_value_threshold=self.model_spec['matrix_inverse_singular_value_threshold'],
                trace_threshold=self.model_spec['matrix_inverse_trace_threshold']
            ) for i in range(weight_dim)
        ]
        op_covariance_inv_updates = [
            covariance_invs[i].assign(
                value=covariance_inv_temps[i]
            ) for i in range(weight_dim)
        ]
        return regularization_loss, covariance_invs, op_covariance_inv_updates

    @staticmethod
    def multilinear_norm(
            weight,
            covariance_invs,
            weight_shape=None
    ):
        if weight_shape is None:
            weight_shape = list(map(lambda x: x.value, weight.shape))
        weight_dim = len(weight_shape)
        middle_results = [weight]
        for dim in range(weight_dim):
            middle_results.append(
                tf.tensordot(
                    a=covariance_invs[dim],
                    b=middle_results[-1],
                    axes=(1, dim)
                )
            )
        norm = tf.einsum(
            "i,i->",
            tf.reshape(weight, shape=[-1]),
            tf.reshape(middle_results[-1], shape=[-1])
        )
        return norm

    @staticmethod
    def multilinear_partial_norm(
            weight,
            covariance_invs,
            axis,
            epsilon=0.00001,
            weight_shape=None
    ):
        if weight_shape is None:
            weight_shape = list(map(lambda x: x.value, weight.shape))
        weight_dim = len(weight_shape)
        perm = [axis]
        for dim in range(weight_dim):
            if dim == axis:
                continue
            perm.append(dim)
        weight_transposed = tf.transpose(
            weight,
            perm=perm
        )
        covariance_invs_transposed = [covariance_invs[dim] for dim in perm]
        middle_results = [weight_transposed]
        for dim in range(1, weight_dim):
            middle_results.append(
                tf.tensordot(
                    a=covariance_invs_transposed[dim],
                    b=middle_results[-1],
                    axes=(1, dim)
                )
            )
        partial_norm = tf.einsum(
            "ik,jk->ij",
            tf.reshape(weight_transposed, shape=[weight_shape[axis], -1]),
            tf.reshape(middle_results[-1], shape=[weight_shape[axis], -1])
        )
        partial_norm_stable = partial_norm + epsilon * tf.eye(weight_shape[axis])
        return partial_norm_stable

    @staticmethod
    def matrix_inverse_generalized(
            matrix,
            singular_value_threshold=0.1,
            trace_threshold=3000.0,
            output_svd_factors=False
    ):
        """
        inverse 2d matrix using svd, following
        https://github.com/thuml/MTlearn/blob/master/src/model_multi_task.py
        for singular_value < singular_value_threshold to keep numerical stable
        and applying afterward trace cut-off
        :return: ! inverse matrix transposed
        """
        s, u, v = tf.linalg.svd(matrix)
        s_inv = MultilinearRelationshipNetwork.select_func(
            inputs=s,
            threshold=singular_value_threshold
        )
        matrix_inv = tf.matmul(
            u,
            tf.matmul(
                tf.diag(s_inv),
                tf.transpose(v)
            )
        )
        matrix_inv_trace = tf.trace(matrix_inv)
        trace_threshold = tf.constant(trace_threshold)
        matrix_inv_trace_cutoff = tf.cond(
            pred=tf.greater(matrix_inv_trace, trace_threshold),
            true_fn=lambda: matrix_inv / matrix_inv_trace * trace_threshold,
            false_fn=lambda: tf.identity(matrix_inv),
            name="matrix_inv_trace_cutoff"
        )
        if output_svd_factors:
            return matrix_inv_trace_cutoff, [s, u, v]
        else:
            return matrix_inv_trace_cutoff

    @staticmethod
    def select_func(
            inputs,
            threshold
    ):
        return tf.where(
            condition=inputs > threshold,
            x=1.0 / inputs,
            y=inputs
        )
