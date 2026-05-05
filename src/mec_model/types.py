"""Shared type aliases for the mainline-A MEC model."""

from __future__ import annotations

from typing import Literal, NewType


TaskId = NewType("TaskId", str)
NodeId = NewType("NodeId", str)
UserId = NewType("UserId", str)

ChannelModelType = Literal["analytic", "3gpp_lite", "rayleigh", "pathloss_only"]
QueueModelType = Literal["mm1", "mmc", "parallel", "finite_capacity"]
MigrationPolicyType = Literal["no_migration", "nearest_edge", "cooperative_migration"]
NodeType = Literal["BS", "RSU", "UAV", "PEER_DEVICE", "CLOUD"]

