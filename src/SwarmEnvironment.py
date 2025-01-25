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

        # action / observation spaces
        self.action_space = spaces.Discrete(5)

        obs_size = 3 * (1 + 2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] * (1 + 2)),
            high=np.array([np.pi, 2*np.pi, 1] * (1 + 2)),
            dtype=np.float32
        )

        # Performance optimization
        self._action_to_direction = {
            0: (1, 0),    # Move toward pole
            1: (-1, 0),   # Move toward equator
            2: (0, 1),    # Move east
            3: (0, -1),   # Move west
            4: (0, 0)     # Stay
        }