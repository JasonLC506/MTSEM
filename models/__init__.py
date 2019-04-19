from models.base_nn import NN
from models.single import FC
from models.mt_lin_adapt import MtLinAdapt
from models.shared_bottom import SharedBottom
from models.inter_task_l2 import InterTaskL2
from models.dmtrl_Tucker import DmtrlTucker
from models.cross_stitch import CrossStitch
from models.mmoe import MMoE
from models.multilinear_relationship_network import MultilinearRelationshipNetwork
from models.topic_task_sparse import TopicTaskSparse


Models = {
    'fc': FC,
    'shared_bottom': SharedBottom,
    'inter_task_l2': InterTaskL2,
    'dmtrl_Tucker': DmtrlTucker,
    'cross_stitch': CrossStitch,
    "mmoe": MMoE,
    "multilinear_relationship_network": MultilinearRelationshipNetwork,
    "topic_task_sparse": TopicTaskSparse
}
