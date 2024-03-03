import random
import math
import matplotlib.pyplot as plt

# Define the world (grid) where the robot and particles exist
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height

# Define a particle with a position (x, y) and a weight
class Particle:
    def __init__(self, x, y, weight=1.0):
        self.x = x
        self.y = y
        self.weight = weight
    
    def __repr__(self):
        return f'Particle(x={self.x}, y={self.y}, weight={self.weight})'

# Define a simple robot with a position (x, y) and noisy sensors
class Robot:
    def __init__(self, world):
        self.world = world
        self.x = random.uniform(0, world.width)
        self.y = random.uniform(0, world.height)

    def move(self, delta_x, delta_y):
        # Simulate robot movement with noise
        self.x += delta_x + random.gauss(0, 0.1)
        self.y += delta_y + random.gauss(0, 0.1)

    def sense(self):
        # Simulate noisy sensor readings
        return random.gauss(0, 1)

# Particle filter simulator
class ParticleFilterSimulator:
    def __init__(self, world, particle_count):
        self.world = world
        self.particles = [Particle(random.uniform(0, world.width), random.uniform(0, world.height)) for _ in range(particle_count)]
        self.robot = Robot(world)

    def update_particles(self, delta_x, delta_y, sensor_reading):
        # Move particles based on robot's movement
        for particle in self.particles:
            particle.x += delta_x + random.gauss(0, 1.0)
            particle.y += delta_y + random.gauss(0, 1.0)

        # Update particle weights based on sensor readings
        for particle in self.particles:
            particle_distance = math.sqrt((particle.x - self.robot.x)**2 + (particle.y - self.robot.y)**2)
            particle.weight = math.exp(-0.5 * (particle_distance - sensor_reading)**2)

        # Normalize particle weights
        total_weight = sum(particle.weight for particle in self.particles)
        for particle in self.particles:
            particle.weight /= total_weight

    def resample_particles(self):
        # Resample particles based on their weights
        new_particles = []
        cumulative_weight = 0
        for _ in range(len(self.particles)):
            choice = random.uniform(0, 1)
            for particle in self.particles:
                cumulative_weight += particle.weight
                if cumulative_weight >= choice:
                    new_particles.append(Particle(particle.x, particle.y))
                    break
            else:
                new_particles.append(Particle(random.uniform(0, self.world.width), random.uniform(0, self.world.height)))
        self.particles = new_particles

    def run_simulation(self, steps):
        for step in range(steps):
            # Robot moves forward
            delta_x = 0.1
            delta_y = 0.0
            self.robot.move(delta_x, delta_y)

            # Robot senses the environment
            sensor_reading = self.robot.sense()

            # Update particles based on robot movement and sensor readings
            self.update_particles(delta_x, delta_y, sensor_reading)

            # Resample particles
            self.resample_particles()

            # Plot the robot and particles
            self.plot(step)

    def plot(self, step):
        plt.figure()
        plt.title(f"Step {step + 1}")
        plt.scatter(self.robot.x, self.robot.y, color='red', label='Robot')

        particle_x = [particle.x for particle in self.particles]
        particle_y = [particle.y for particle in self.particles]
        # weights = [particle.weight * 1000 for particle in self.particles]  # Scale weights for better visualization
        plt.scatter(particle_x, particle_y, cmap='viridis', label='Particles', alpha=0.5)
        plt.colorbar(label='Particle Weight * 1000')
        plt.legend()
        plt.xlim(0, self.world.width + 1)
        plt.ylim(0, self.world.height + 1)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.show()

# Define the world dimensions
world_width = 10
world_height = 5

# Create the world and particle filter simulator
world = World(world_width, world_height)
particle_filter_simulator = ParticleFilterSimulator(world, particle_count=1000)

# Run the simulation for 5 steps
particle_filter_simulator.run_simulation(steps=10)
