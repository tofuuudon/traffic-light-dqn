"""Generic typings for the project."""

from collections import namedtuple
from dataclasses import dataclass

from traci._trafficlight import TrafficLightDomain


@dataclass
class TrafficLightSystem:
    """Dataclass for TLS containing information about its states."""

    tls_id: str
    lane_ids: tuple[str]
    phases: tuple[TrafficLightDomain.Phase]


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    alpha: float
    epsilon: float
    epsilon_max: float
    epsilon_min: float
    epsilon_decay: int
    gamma: float
    batch_size: int
    replay_size: int
    sync_rate: int


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))
