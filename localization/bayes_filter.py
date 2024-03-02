import numpy as np
import abc
import matplotlib.pyplot as plt

class BayesFilter(object):
    def __init__(self, initial_belief, motion_model, sensor_model):
        self.belief = initial_belief
        self.transition_model = motion_model
        self.sensor_model = sensor_model

    def predict(self, control):
        self.belief = self.transition_model.predict(self.belief, control)
    
    def update(self, measurement):
        self.belief = self.sensor_model.update(self.belief, measurement)

    def localize(self, control, measurement):
        self.predict(control)
        self.update(measurement)
        return self.belief

class SensorModel(abc.ABC):
    
        @abc.abstractmethod
        def __init__(self, measurement_noise=0.1):
            ...
    
        @abc.abstractmethod
        def generate_measurement(self, true_state):
            ...
    
        @abc.abstractmethod
        def update(self, belief, measurement):
            ...


class MotionModel(abc.ABC):

    @abc.abstractmethod
    def __init__(self, motion_noise=0.1):
        ...

    @abc.abstractmethod
    def predict(self, belief, control):
        ...


class CarMotionModel(MotionModel):
    def __init__(self, motion_noise=0.0):
        self.motion_noise = motion_noise

    def predict(self, belief, control_input):
        """
        Predicts the next state based on the control input using the kinematic equations.
        
        Parameters:
        - belief: Current belief about the car's state [x, y, theta, v]
        - control_input: Control input [delta_steer, acceleration]
        """
        x, y, theta, v = belief

        delta_steer, acceleration = control_input
        delta_t = 0.1  # Time step for prediction

        # Kinematic equations
        x += v * np.cos(theta) * delta_t
        y += v * np.sin(theta) * delta_t
        theta += (v / 2.0) * np.tan(delta_steer) / 2.5 * delta_t  # Simplified model for steering
        v += acceleration * delta_t

        # Add motion noise
        x += np.random.normal(0, self.motion_noise)
        y += np.random.normal(0, self.motion_noise)
        theta += np.random.normal(0, self.motion_noise)
        v += np.random.normal(0, self.motion_noise)

        return np.array([x, y, theta, v])
    
class SimpleCarSensorModel(SensorModel):
    def __init__(self, measurement_noise=0.001):
        self.measurement_noise = measurement_noise

    def generate_measurement(self, true_state):
        """
        Generates a noisy measurement based on the true state of the car.

        Parameters:
        - true_state: True state of the car [x, y, theta, v]

        Returns:
        - Noisy measurement [x_measurement, y_measurement]
        """
        x, y, _, _ = true_state

        # Add measurement noise
        x_measurement = x + np.random.normal(0, self.measurement_noise)
        y_measurement = y + np.random.normal(0, self.measurement_noise)

        return np.array([x_measurement, y_measurement])

    def update(self, belief, measurement):
        """
        Updates the belief based on the sensor measurement.

        Parameters:
        - belief: Current belief about the car's state [x, y, theta, v]
        - measurement: Sensor measurement [x_measurement, y_measurement]

        Returns:
        - Updated belief
        """
        # For simplicity, assume a simple Gaussian noise model without the complex likelihood calculation
        updated_belief = belief + np.random.normal(0, self.measurement_noise, size=len(belief))

        # Normalize to make it a probability distribution
        updated_belief /= np.sum(updated_belief)

        return updated_belief
    
if __name__ == "__main__":
    initial_belief = np.array([0, 0, 0, 0])  # Initial belief about the car's state [x, y, theta, v]
    motion_model = CarMotionModel()
    sensor_model = SimpleCarSensorModel()
    bayes_filter = BayesFilter(initial_belief, motion_model, sensor_model)

    num_steps = 100  # Number of simulation steps

    # Lists to store the true and estimated states for plotting
    true_states = [initial_belief[:2]]
    estimated_states = [initial_belief[:2]]

    for step in range(num_steps):
        # Simulate control input
        control_input = np.array([0.5, 0.1])

        # Simulate motion and update the belief
        bayes_filter.predict(control_input)

        # Simulate sensor measurement
        true_state = motion_model.predict(bayes_filter.belief, control_input)
        measurement = sensor_model.generate_measurement(true_state)

        # Update the belief based on the sensor measurement
        bayes_filter.update(measurement)

        # Append the true and estimated states for plotting
        true_states.append(true_state[:2])
        estimated_states.append(bayes_filter.belief[:2])

    true_states = np.array(true_states)
    estimated_states = np.array(estimated_states)

    # Plotting
    plt.plot(true_states[:, 0], true_states[:, 1], label="True Trajectory", linestyle="--", marker='o')
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], label="Estimated Trajectory", linestyle="-", marker='x')
    plt.title("Simulated Car Localization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()