"""
topic task sparse network with exclusive sparsity regularization
"""
from models import proximal_operator_exclusive_sparse, TopicTaskSparseLayerWiseSingleLayer


class TopicTaskSparseLayerWiseSingleLayerExclusive(TopicTaskSparseLayerWiseSingleLayer):
    @staticmethod
    def proximal_operator(
            weight,
            **kwargs
    ):
        return proximal_operator_exclusive_sparse(
            weight=weight,
            **kwargs
        )
