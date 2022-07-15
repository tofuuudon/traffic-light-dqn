"""Instantiates SUMO with pre-defined network and config for DQN."""

import os
import sys

# pylint disable: import-error

import traci
from sumolib import checkBinary

# Checks for SUMO_HOME enviroment
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

sumoBinary = checkBinary("sumo")
sumoCmd = [sumoBinary, "-W", "-c", "data/train-network/osm.sumocfg"]

STEP = 0
N_STEP = 3600

traci.start(sumoCmd)

# Gets all 4-way intersections with 16 states
TLS_IDS: list[int] = [
    id
    for id in traci.trafficlight.getIDList()
    if len(traci.trafficlight.getRedYellowGreenState(id)) == 16
]

# Runtime
while STEP <= N_STEP:
    traci.simulationStep()

    STEP += 1

traci.close()
