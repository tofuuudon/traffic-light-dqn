"""Generic typings for the project."""

from dataclasses import dataclass

from traci._trafficlight import TrafficLightDomain


@dataclass
class TrafficLightSystem:
    """Dataclass for TLS containing information about its states."""

    tls_id: str
    lane_ids: tuple[str]
    phases: tuple[TrafficLightDomain.Phase]
