import numpy as np
from gymnasium import Env, spaces
from SwarmAgent import SwarmAgent

class SwarmEnvironment(Env):
    def __init__(self, energy_field, n_agents=10, max_steps=1000):
        super().__init__()

        self.energy_field = energy_field
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.current_step = 0

        # Init agents with random position
        self.agents = [
            SwarmAgent(
                initial_theta=np.random.uniform(0, np.pi),
                initial_phi=np.random.uniform(0, 2*np.pi)
            ) for _ in range(n_agents)
        ]
        