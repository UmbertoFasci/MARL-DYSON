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

    def _get_state(self, agent_idx):
        agent = self.agents[agent_idx]
        agent_pos = np.array(agent.get_position())

        # locate 2 nearest neighbors using vectorized operations
        positions = np.array([a.get_position() for a in self.agents])
        distances = np.sum((positions - agent_pos)**2, axis=1)
        distances[agent_idx] = np.inf
        neighbor_indices = np.argpartition(distances, 2)[:2]

        # state vector
        state = np.zeros(9, dtype=np.float32)

        # self
        state[0:2] = agent_pos
        state[2] = self.energy_field.get_energy_at_position(*agent_pos)

        # neighbors
        for i, idx in enumerate(neighbor_indices):
            pos = positions[idx]
            state[3+i*3:5+i*3] = pos
            state[5+i*3] = self.energy_field.get_energy_at_position(*pos)
        
        return state