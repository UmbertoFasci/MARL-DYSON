import numpy as np

class SwarmAgent:
    def __init__(self, initial_theta, initial_phi, step_size=0.1):
        self.theta = initial_theta  # Latitude [0, pi]
        self.phi = initial_phi      # Longitude [0, 2pi]
        self.step_size = step_size
        self.energy_collected = 0.0
        self.position_history = [(initial_theta, initial_phi)]
    
    def move(self, direction):
        d_theta, d_phi = direction
        
        # Update position with boundary checking
        new_theta = np.clip(self.theta + d_theta * self.step_size, 0, np.pi)
        new_phi = (self.phi + d_phi * self.step_size) % (2 * np.pi)
        
        self.theta = new_theta
        self.phi = new_phi
        self.position_history.append((new_theta, new_phi))
    
    def get_position(self):
        return (self.theta, self.phi)
    
    def collect_energy(self, energy_field):
        energy = energy_field.get_energy_at_position(self.theta, self.phi)
        self.energy_collected += energy
        return energy
    
    def get_possible_moves(self):
        return [
            (1, 0),   # Move toward pole
            (-1, 0),  # Move toward equator
            (0, 1),   # Move east
            (0, -1),  # Move west
            (0, 0),   # Stay in place
        ]

    def get_cartesian_coords(self, radius=1.0):
        x = radius * np.sin(self.theta) * np.cos(self.phi)
        y = radius * np.sin(self.theta) * np.sin(self.phi)
        z = radius * np.cos(self.theta)
        return (x, y, z)