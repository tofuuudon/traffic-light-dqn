"""Instantiates SUMO with pre-defined network and configuration for DQN."""

import os
import sys

import traci
from sumolib import checkBinary
from torch.utils.tensorboard.writer import SummaryWriter

from agent import Agent
from cli import CLI
from _typings import TrafficLightSystem, Experience, AgentConfig

# Args parser
args = CLI().get_args()

print("\n========== Starting Simulation ==========")
print(f"Mode: {'GUI' if args.gui else 'No GUI'}")
print(f"Number of episodes: {args.episodes}")
print(f"Maximum of steps: {args.max_step}\n")

# Checks for SUMO_HOME enviroment
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Setup
sumoBinary = checkBinary("sumo-gui" if args.gui else "sumo")
sumoCmd = [sumoBinary, "-W", "-c", "data/train-network/osm.sumocfg"]

traci.start(sumoCmd)

# Create a new starting state file if not existing
START_STATE_PATH = "data/train-network/start.state.xml"
if not os.path.exists(START_STATE_PATH):
    traci.simulation.saveState(START_STATE_PATH)

agent_config = AgentConfig(
    alpha=1e-2,
    epsilon=0.99,
    epsilon_max=0.99,
    epsilon_min=0.05,
    epsilon_decay=1_800,
    gamma=0.99,
    batch_size=32,
    replay_size=10_000,
    sync_rate=10,
)

# All TLS agents
TLS_AGENTS: tuple[Agent, ...] = tuple(
    Agent(
        TrafficLightSystem(
            tls_id,
            traci.trafficlight.getControlledLanes(tls_id),
            traci.trafficlight.getAllProgramLogics(tls_id)[0].phases,
        ),
        agent_config,
    )
    for tls_id in traci.trafficlight.getIDList()
)

# Tensorboard logger
writter = SummaryWriter()

# Main simulation loop
for ep in range(args.episodes):
    EPS_REWARD: float = 0

    # Episode simulation stepper
    for step in range(args.max_step):

        # Prepares action for each TLS agent
        sa_pairs = [agent.prepare_step(step) for agent in TLS_AGENTS]

        traci.simulationStep()

        # In the next step, evaluate the action taken
        for idx, agent in enumerate(TLS_AGENTS):
            state, action = sa_pairs[idx]
            next_state, reward = agent.evaluate_step(state)

            # Saves the experience
            agent.memory.push(Experience(state, action, next_state, reward))

            # Performs training with epsilon greedy
            agent.train(step)

            # Updates this episode's reward
            EPS_REWARD += reward.reshape(-1)[0].item()

    # Saves data to tensorboard
    writter.add_scalar("Episode reward", EPS_REWARD, ep)

    # Resets simulation after each episode
    traci.simulation.loadState(START_STATE_PATH)


writter.close()
traci.close()
