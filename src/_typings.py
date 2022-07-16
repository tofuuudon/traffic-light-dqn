"""Generic typings for the project."""

from dataclasses import dataclass


@dataclass
class TrafficLightSystem:
    """Dataclass for TLS containing information about its states."""

    tls_id: str
    lane_ids: list[str]
