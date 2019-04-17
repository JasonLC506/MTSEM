"""
MT-LinAdapt
@inproceedings{gong2016modeling,
  title={Modeling social norms evolution for personalized sentiment classification},
  author={Gong, Lin and Al Boni, Mohammad and Wang, Hongning},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={855--865},
  year={2016}
}

specific design for PSEM problem
"""
import tensorflow as tf

from models import NN

Optimizer = tf.train.AdamOptimizer


class MtLinAdapt(NN):
    def __init__(
            self,
            feature_shape,
            feature_dim,
            label_dim,
            task_dim,
            model_spec,
            model_name=None
    ):
        self.feature_shape = feature_shape
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.task_dim = task_dim
        self.model_spec = model_spec
        self.group_dim = model_spec["group_dim"]
        if model_name is None:
            self.model_name = model_spec["name"]
        else:
            self.model_name = model_name

        super(MtLinAdapt, self).__init__(graph=None)

    def initialization(
            self,
            feature_group_file="../data/feature_group",
            w0_file="../data/w0"
    ):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)
        self.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.feature_group_placeholder,
            fixed_input_init=self.feature_group_init,
            fixed_input_file=feature_group_file
        )
        self.initialize_fixed_input(
            sess=self.sess,
            fixed_input_placeholder=self.w0_placeholder,
            fixed_input_init=self.w0_init,
            fixed_input_file=w0_file
        )

    def _setup_placeholder(self):
        with tf.name_scope("placeholder"):
            self.feature_index = tf.placeholder(
                dtype=tf.int32,
                shape=[None, self.feature_dim],
                name="feature_index"
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
            self.task_id = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name="task_id"
            )                              # task represented as id for many tasks

    def _setup_net(self):
        self.feature_group, self.feature_group_placeholder, self.feature_group_init = self.fixed_input_load(
            input_shape=[self.group_dim, self.feature_shape, self.label_dim],
            dtype=tf.float32,
            trainable=False,
            name="feature_group"
        )

        self.w0, self.w0_placeholder, self.w0_init = self.fixed_input_load(
            input_shape=[self.feature_shape, self.label_dim],
            dtype=tf.float32,
            trainable=False,
            name="w0"
        )

        a_s = tf.Variable(
            tf.random_normal(
                shape=[self.feature_shape, self.label_dim],
                mean=1.0,
                stddev=0.001
            ),
            dtype=tf.float32,
            name="scale_shared"
        )

        b_s = tf.Variable(
            tf.random_normal(
                shape=[self.feature_shape, self.label_dim],
                mean=0.0,
                stddev=0.001
            ),
            dtype=tf.float32,
            name="shift_shared"
        )

        w_s = a_s * self.w0 + b_s

        w_s_g = self.feature_group * tf.expand_dims(
            w_s,
            axis=0
        )

        w_s_g_trans = tf.transpose(
            w_s_g,
            perm=[1, 0, 2]
        )                                   # shape=[feature_shape, group_dim, label_dim]

        sum_f_g_looked_up = tf.nn.embedding_lookup(
            params=w_s_g_trans,
            ids=self.feature_index
        )                                   # shape=[batch_size, feature_dim, group_dim, label_dim]

        sum_f_g = tf.reduce_sum(sum_f_g_looked_up, axis=1)        # shape=[batch_size, group_dim, label_dim]

        feature_group_trans = tf.transpose(
            self.feature_group,
            perm=[1, 0, 2]
        )

        sum_g_looked_up = tf.nn.embedding_lookup(
            params=feature_group_trans,
            ids=self.feature_index
        )                                   # shape=[batch_size, feature_dim, group_dim, label_dim]

        sum_g = tf.reduce_sum(sum_g_looked_up, axis=1)             # shape=[batch_size, group_dim, label_dim]

        a = tf.Variable(
            tf.random_normal(
                shape=[self.task_dim, self.group_dim],
                mean=1.0,
                stddev=0.001
            ),
            dtype=tf.float32,
            name="scale"
        )

        b = tf.Variable(
            tf.random_normal(
                shape=[self.task_dim, self.group_dim],
                mean=0.0,
                stddev=0.001
            ),
            dtype=tf.float32,
            name="shift"
        )

        task_scale_group = tf.nn.embedding_lookup(
            params=a,
            ids=self.task_id
        )

        task_shift_group = tf.nn.embedding_lookup(
            params=b,
            ids=self.task_id
        )

        self.logits_scale = tf.einsum(
            "ij,ijk->ik",
            task_scale_group,
            sum_f_g
        )

        self.logits_shift = tf.einsum(
            "ij,ijk->ik",
            task_shift_group,
            sum_g
        )

        self.logits = self.logits_scale + self.logits_shift

        self.label_pred = tf.nn.softmax(
            logits=self.logits,
            axis=-1,
            name="softmax"
        )

        self.l2_a_s = tf.pow(
            tf.norm(a_s - tf.constant(1.0), ord=2),
            2.0
        )

        self.l2_b_s = tf.pow(
            tf.norm(b_s, ord=2),
            2.0
        )

        self.l2_a = tf.pow(
            tf.norm(a - tf.constant(1.0), ord=2),
            2.0
        )

        self.l2_b = tf.pow(
            tf.norm(b, ord=2),
            2.0
        )

    def _setup_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.logits,
            name="cross_entropy"
        )
        # TODO: for many tasks, task-specific loss is not calculated; otherwise notimplemented
        self.loss_cross_entropy = tf.tensordot(
            cross_entropy,
            self.weight,
            axes=[0, 0]
        )
        self.loss_cross_entropy_mean = self.loss_cross_entropy / tf.reduce_sum(self.weight)

        self.regularization_loss = self.model_spec["eta_a"] * self.l2_a + \
            self.model_spec["eta_b"] * self.l2_b + \
            self.model_spec["eta_a_s"] + self.l2_a_s + \
            self.model_spec["eta_b_s"] + self.l2_b_s
        self.loss = self.loss_cross_entropy + self.regularization_loss
        self.loss_mean = self.loss_cross_entropy_mean + self.regularization_loss

    def _setup_optim(self):
        # TODO:: using BERT optimizer with warm-up and weight decay and correct l2 regularization
        self.optimizer = Optimizer(
            learning_rate=self.model_spec["learning_rate"],
            epsilon=1e-06,
            name="optimizer"
        ).minimize(self.loss_mean)

    def train(
            self,
            data_generator,
            data_generator_valid=None
    ):
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss, self.loss_cross_entropy],
            session=self.sess,
            op_data_size=None,
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
            self.feature_index: data["feature_index"][batch_index],
            self.label: data["label"][batch_index],
            self.weight: data["weight"][batch_index],
            self.task_id: data["task"][batch_index]
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
            self.feature_index: data["feature_index"][batch_index],
            self.weight: data["weight"][batch_index],
            self.task_id: data["task"][batch_index]
        }
        return feed_dict
