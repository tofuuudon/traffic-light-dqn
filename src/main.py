"""Instantiates SUMO with pre-defined network and configuration for DQN."""

import os
import sys
from datetime import datetime

import torch
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


print("\n========== Agent Configuration ==========")
print(f"Max epsilon: {args.epsilon_max}")
print(f"Min epsilon: {args.epsilon_min}")
print(f"Epsilon decay: {args.epsilon_decay}")
print(f"Gamma: {args.gamma}")
print(f"Batch size: {args.batch_size}")
print(f"Replay size: {args.replay_size}")
print(f"Sync rate: {args.sync_rate}\n")

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

# Configuration for all agents
agent_config = AgentConfig(
    epsilon=args.epsilon_max,
    epsilon_max=args.epsilon_max,
    epsilon_min=args.epsilon_min,
    epsilon_decay=args.epsilon_decay,
    gamma=args.gamma,
    batch_size=args.batch_size,
    replay_size=args.replay_size,
    sync_rate=args.sync_rate,
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
writter = SummaryWriter() if args.log else None

# Main simulation loop
for ep in range(args.episodes):
    eps_reward: float = 0

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
            eps_reward += reward.reshape(-1)[0].item()

    # Gets simulation data
    vehicle_ids: list[str] = traci.vehicle.getIDList()
    waiting_times = map(traci.vehicle.getWaitingTime, vehicle_ids)
    time_loss = map(traci.vehicle.getTimeLoss, vehicle_ids)
    avg_waiting_time = sum(waiting_times) / len(vehicle_ids)
    avg_time_loss = sum(time_loss) / len(vehicle_ids)

    if isinstance(writter, SummaryWriter):
        # Saves data to tensorboard
        writter.add_scalar("Episode reward", eps_reward, ep)
        writter.add_scalar("Vehicle count", len(vehicle_ids), ep)
        writter.add_scalar("Avg. waiting time", avg_waiting_time, ep)
        writter.add_scalar("Avg. time loss", avg_time_loss, ep)

    print(f"\n========= Episode {ep + 1} =========")
    print(f"Episode Reward: {eps_reward}")
    print(f"Vehicle count: {len(vehicle_ids)}")
    print(f"Avg. waiting time: {avg_waiting_time}")
    print(f"Avg. time loss: {avg_time_loss}")

    # Resets simulation after each episode
    traci.simulation.loadState(START_STATE_PATH)
    print()

NOW = datetime.now().strftime("%d-%m-%YT%H:%M:%S")

if not os.path.exists(f"models/{NOW}"):
    os.mkdir(f"models/{NOW}")

if args.save_models:
    for agent in TLS_AGENTS:
        tls_id = agent.tls_node.tls_id
        if not os.path.exists(f"models/{NOW}/{tls_id}"):
            os.mkdir(f"models/{NOW}/{tls_id}")
        torch.save(
            agent.net.state_dict(),
            f"models/{NOW}/{tls_id}/policy-net.pt",
        )
        torch.save(
            agent.target_net.state_dict(),
            f"models/{NOW}/{tls_id}/policy-target-net.pt",
        )

if isinstance(writter, SummaryWriter):
    writter.close()

traci.close()
