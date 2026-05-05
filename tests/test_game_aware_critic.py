"""Tests for game-aware critic features."""

from src.game_pricing.types import PriceVector
from src.mec_model.state import EdgeNodeState, QueueSnapshot, SystemState, UserState
from src.mec_model.types import NodeId, UserId
from src.rl_algorithms.game_aware.critic_features import (
    build_critic_features,
    mask_unavailable_edges,
)


def test_critic_features_include_queue_and_price() -> None:
    state = SystemState(
        users={UserId("u"): UserState(UserId("u"))},
        edge_nodes={NodeId("e"): EdgeNodeState(NodeId("e"), queue=QueueSnapshot(1, 2, 1))},
    )

    features = build_critic_features(state, PriceVector((1.0,)), {"queue_penalty": 1.0})
    masked = mask_unavailable_edges(features, [False])

    assert features.queue_pressure
    assert features.price_vector == (1.0,)
    assert masked.price_vector == (0.0,)

