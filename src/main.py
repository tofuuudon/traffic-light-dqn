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
sumoCmd = [
    sumoBinary,
    "-W",
    "-c",
    "data/train-network/osm.sumocfg",
    "--statistic-output",
    "report/statistic.xml",
]

traci.start(sumoCmd)
STEP = 0

while STEP <= 3600:
    traci.simulationStep()
    STEP += 1

traci.close()
