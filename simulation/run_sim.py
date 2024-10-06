from src.Simulation import Simulation
import json
import os


config = json.load(open("config/default_sim_config.json", "r"))
sim = Simulation(config)
sim.run()
