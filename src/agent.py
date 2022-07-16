"""Traffic light system (TLS) agent."""

from numpy import float64
from numpy._typing import NDArray


class Agent:
    """Simple TLS agent."""

    def __init__(
        self,
        obs_space: NDArray[float64],
        n_action: int,
    ) -> None:

        # Environment
        self.obs_size = obs_space
        self.action_size = n_action

        # Metrics
        self.total_reward: float = 0
