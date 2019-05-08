from models.base_nn import NN
from models.mt_lin_adapt import MtLinAdapt
from models.shared_bottom import SharedBottom
from models.inter_task_l2 import InterTaskL2
from models.dmtrl_Tucker import DmtrlTucker
from models.cross_stitch import CrossStitch
from models.mmoe import MMoE
from models.multilinear_relationship_network import MultilinearRelationshipNetwork
from models.topic_task_sparse import TopicTaskSparse
from models.topic_task_sparse_layer_wise import TopicTaskSparseLayerWise
from models.topic_task_sparse_layer_wise_single_layer import TopicTaskSparseLayerWiseSingleLayer
from models.single import FC
from models.topic_task_sparse_layer_wise_exclusive import TopicTaskSparseLayerWiseExclusive, \
    proximal_operator_exclusive_sparse
from models.topic_task_sparse_layer_wise_single_layer_exclusive import TopicTaskSparseLayerWiseSingleLayerExclusive


Models = {
    'fc': FC,
    'shared_bottom': SharedBottom,
    'inter_task_l2': InterTaskL2,
    'dmtrl_Tucker': DmtrlTucker,
    'cross_stitch': CrossStitch,
    "mmoe": MMoE,
    "multilinear_relationship_network": MultilinearRelationshipNetwork,
    "topic_task_sparse": TopicTaskSparse,
    "topic_task_sparse_layer_wise": TopicTaskSparseLayerWise,
    "topic_task_sparse_layer_wise_single_layer": TopicTaskSparseLayerWiseSingleLayer,
    "topic_task_sparse_layer_wise_exclusive": TopicTaskSparseLayerWiseExclusive,
    "topic_task_sparse_layer_wise_single_layer_exclusive": TopicTaskSparseLayerWiseSingleLayerExclusive
}
