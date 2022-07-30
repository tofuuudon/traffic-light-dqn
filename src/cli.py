"""Creates CLI for interacting with SUMO and the DQN."""

from argparse import ArgumentParser, Namespace


class CLI:
    """CLI instance for simulation configurations."""

    def __init__(self) -> None:
        self.parser = ArgumentParser(description="CLI for SUMO RL")

        # Simulation
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

        # Agent
        self.parser.add_argument(
            "--epsilon-max",
            metavar="\b",
            type=float,
            help="Starting/maximum value of epsilon.",
            default=0.99,
        )
        self.parser.add_argument(
            "--epsilon-min",
            metavar="\b",
            type=float,
            help="Minimum value of epsilon.",
            default=0.05,
        )
        self.parser.add_argument(
            "--epsilon-decay",
            metavar="\b",
            type=int,
            help="The proportional decay of epsilon in steps.",
            default=1_800,
        )
        self.parser.add_argument(
            "--gamma",
            metavar="\b",
            type=float,
            help="The discount rate.",
            default=0.99,
        )
        self.parser.add_argument(
            "--batch-size",
            metavar="\b",
            type=int,
            help="The number of replay memory experiences to draw from.",
            default=32,
        )
        self.parser.add_argument(
            "--replay-size",
            metavar="\b",
            type=int,
            help="The size of the replay memory.",
            default=10_000,
        )
        self.parser.add_argument(
            "--sync-rate",
            metavar="\b",
            type=int,
            help="The frequency of syncing the target net in steps.",
            default=10,
        )

    def get_args(self) -> Namespace:
        """Get arguments from CLI.

        Returns:
            Namespace: All arguments from the CLI.
        """
        return self.parser.parse_args()
