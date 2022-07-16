"""Traffic light system (TLS) agent."""


class Agent:
    """Simple TLS agent."""

    def __init__(
        self,
        obs_space: tuple[int, ...],
        action_space: tuple[int, ...],
    ) -> None:

        # Environment
        self.obs_size = obs_space
        self.action_size = action_space

        # Metrics
        self.total_reward: float = 0
