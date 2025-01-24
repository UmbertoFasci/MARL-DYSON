import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class StarEnergyField:
    def __init__(self, radius=1.0, resolution=50, smoothing=2.0):
        self.radius = radius
        self.resolution = resolution
        self.smoothing = smoothing
        
        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        self.theta, self.phi = np.meshgrid(theta, phi)
        
        self.energy_field = self.generate_energy_field()
    
    def generate_energy_field(self):
        random_field = np.random.rand(self.resolution, self.resolution)
        smoothed_field = gaussian_filter(random_field, sigma=self.smoothing)
        return (smoothed_field - smoothed_field.min()) / (smoothed_field.max() - smoothed_field.min())
    
    def get_energy_at_position(self, theta, phi):
        theta_idx = np.argmin(np.abs(np.linspace(0, np.pi, self.resolution) - theta))
        phi_idx = np.argmin(np.abs(np.linspace(0, 2*np.pi, self.resolution) - phi))
        return self.energy_field[phi_idx, theta_idx]
    
    def visualize_energy_field(self):
        x = self.radius * np.sin(self.theta) * np.cos(self.phi)
        y = self.radius * np.sin(self.theta) * np.sin(self.phi)
        z = self.radius * np.cos(self.theta)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        norm = plt.Normalize(self.energy_field.min(), self.energy_field.max())
        colors = plt.cm.plasma(norm(self.energy_field))
        surf = ax.plot_surface(x, y, z, facecolors=colors, alpha=0.7)
        
        m = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
        fig.colorbar(m, ax=ax, label='Energy Level')
        
        ax.set_title('Energy Field Distribution')
        plt.show()