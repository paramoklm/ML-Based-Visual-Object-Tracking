import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        # Define parameters
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.x_pred = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[0.25 * dt**4, 0, 0.5 * dt**3, 0],
                           [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                           [0.5 * dt**3, 0, dt**2, 0],
                           [0, 0.5 * dt**3, 0, dt**2]]) * std_acc**2
        self.R = np.diag([x_sdt_meas**2, y_sdt_meas**2])
        self.P = np.eye(4)

    def predict(self):
        # Predict the state estimate and error covariance
        self.A, self.B
        self.x_pred = np.dot(self.A, self.x_pred) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update the state estimate and error covariance based on measurements
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x_pred = self.x_pred + np.dot(K, (z - np.dot(self.H, self.x_pred)))
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

# Test the KalmanFilter class
if __name__ == "__main__":
    # Example usage
    dt = 0.1
    u_x, u_y = 1, 1
    std_acc = 1
    x_sdt_meas, y_sdt_meas = 0.1, 0.1
    kalman_filter = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)
    kalman_filter.x_pred = np.array([[0], [0], [0], [0]])
    kalman_filter.predict()
    z_measurement = np.array([[10], [10]])
    kalman_filter.update(z_measurement)
    print("Predicted State Estimate:", kalman_filter.x_pred)

