"""Creates CLI for interacting with SUMO and the DQN."""

from argparse import ArgumentParser, Namespace


class CLI:
    """CLI instance for simulation configurations."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(description="CLI for SUMO RL")
        self.parser.add_argument(
            "-e",
            "--episodes",
            metavar="\b",
            type=int,
            help="The number of episodes to run.",
            default=10,
        )
        self.parser.add_argument(
            "-m",
            "--max_step",
            metavar="\b",
            type=int,
            help="The maximum number of steps in each episode.",
            default=3600,
        )
        self.parser.add_argument(
            "-g",
            "--gui",
            action="store_true",
            help="Sets SUMO to launch with GUI.",
        )

    def get_args(self) -> Namespace:
        """Get arguments from CLI.

        Returns:
            Namespace: All arguments from the CLI.
        """
        return self.parser.parse_args()
