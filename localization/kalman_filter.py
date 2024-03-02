import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, initial_state, initial_covariance, process_variance, measurement_variance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # Define state transition matrix (A) for the 2D car model
        self.A = np.array([
            [1, 0, 0.1, 0],
            [0, 1, 0, 0.1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Define control input matrix (B) for the 2D car model
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0.1, 0],
            [0, 0.1]
        ])

        # Define measurement matrix (H) for the 2D car model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def predict(self, control_input):
        # Prediction step
        # State prediction
        predicted_state = np.dot(self.A, self.state) + np.dot(self.B, control_input)

        # Covariance prediction
        predicted_covariance = np.dot(np.dot(self.A, self.covariance), self.A.T) + self.process_variance

        # Update state and covariance
        self.state = predicted_state
        self.covariance = predicted_covariance

        return predicted_state, predicted_covariance

    def update(self, measurement):
        # Update step
        # Kalman gain calculation
        kalman_gain = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.covariance), self.H.T) + self.measurement_variance))

        # Update state and covariance
        self.state = self.state + np.dot(kalman_gain, measurement - np.dot(self.H, self.state))
        self.covariance = np.dot((np.eye(len(self.state)) - np.dot(kalman_gain, self.H)), self.covariance)

        return self.state, self.covariance
    

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # True state evolution (simulated)
    true_states = np.zeros((100, 4))
    true_states[0] = np.array([0, 0, 0, 0])

    # Add process and measurement noise
    process_noise = np.random.normal(0, 0.01, (len(true_states), 2))
    measurement_noise = np.random.normal(0, 0.001, (len(true_states), 2))

    # Simulate noisy measurements
    measurements = true_states[:, :2] + measurement_noise

    # Initialize Kalman Filter
    initial_state = np.array([0, 0, 0, 0])
    initial_covariance = np.eye(4)
    process_variance = 0.01
    measurement_variance = 0.001

    kalman_filter = KalmanFilter2D(initial_state, initial_covariance, process_variance, measurement_variance)

    # Kalman Filter loop
    filtered_states = []
    for i in range(len(true_states)):
        # Prediction step
        predicted_state, _ = kalman_filter.predict(process_noise[i])

        # Update step
        filtered_state, _ = kalman_filter.update(measurements[i])

        filtered_states.append(filtered_state)
    # Plot the results
    plt.plot(true_states[:, 0], true_states[:, 1], label='True Trajectory', linestyle='--', marker='o')
    plt.plot(measurements[:, 0], measurements[:, 1], label='Measurements', linestyle='None', marker='x')
    plt.plot(np.array(filtered_states)[:, 0], np.array(filtered_states)[:, 1], label='Filtered Trajectory', linestyle='-', marker='.')
    plt.title('2D Kalman Filter Example for Car Model')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()
