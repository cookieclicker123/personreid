import numpy as np


class KalmanFilter:
    def __init__(self, measurement, dt=0.1):
        # Noise
        R_std = 0.35
        Q_std = 0.04
        self.R = np.eye(2) * R_std ** 2
        self.Q = np.eye(4) * Q_std ** 2

        # Initiliza State
        # 4D state with initial position and initial velocity assigned
        self.X = np.array([[measurement[0]], [measurement[1]], [0.0], [0.0]])

        # self.Bu = np.zeros((4,4))
        # self.U = np.array([[0.0], [0.0], [0.0], [0.0]])

        # Initilize Error covariance Matrix
        gamma = 10
        self.P = gamma * np.eye(4)

        # State Transition matrix
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.B = np.eye(4)
        self.C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # Obtain prior estimates and error covariance from initial values

    def predict(self):
        # prior state estimate
        self.X = self.A @ self.X
        # prior-error covariance matrix
        self.P = self.A @ self.P @ self.A.T + self.B @ self.Q @ self.B.T

    def update(self, measurement: tuple):
        # kalman gain
        self.G = (
            self.P @ self.C.T @ np.linalg.inv((self.C @ self.P @ self.C.T + self.R))
        )
        # new state estimate
        self.X = self.X + self.G @ (
            np.array(measurement).reshape(-1, 1) - self.C @ self.X
        )
        # new-error covariance matrix
        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

"""
This script defines a class for implementing a basic Kalman Filter, a popular algorithm for tracking and prediction in various applications like object tracking in video, navigation systems, and more.

The Kalman Filter operates in two steps: Prediction and Update. Here's an overview of how each component and method within the KalmanFilter class contributes to the tracking process:

1. **Initialization (__init__)**:
   - **Noise Parameters**:
     - `R_std` and `Q_std` define the standard deviations for measurement noise and process noise, respectively. These values are crucial as they influence how much trust the filter places in the measurements versus its own predictions.
     - `self.R`: Measurement noise covariance matrix. It represents the expected uncertainty in the measurements. In this case, the measurement noise is assumed to be independent across different measurements with a variance of R_std^2.
     - `self.Q`: Process noise covariance matrix. It describes the expected uncertainty in the system dynamics, accounting for unknown influences on the state variables (e.g., unexpected movements).
   
   - **State Initialization**:
     - The state vector `self.X` is initialized with the position from the initial measurement and zero velocity. It's a four-dimensional vector representing position and velocity in two dimensions ([x, y, vx, vy]).
   
   - **Error Covariance Matrix (self.P)**:
     - Initialized to a scaled identity matrix, where `gamma` controls the initial certainty about the state estimates. A higher `gamma` implies less certainty about the initial state.
   
   - **State Transition Matrix (self.A)**:
     - Defines how the state variables are expected to evolve from one timestep to the next, without considering the control inputs or process noise. Here, it's set to account for simple linear motion (x += vx*dt, y += vy*dt).
   
   - **Measurement Matrix (self.C)**:
     - Maps the state vector into the measurement domain. It's used to convert the predicted state estimate into a form that can be directly compared to the measurement.

2. **Prediction (predict method)**:
   - Uses the state transition matrix `self.A` to predict the next state based on the current state estimate.
   - Updates the error covariance matrix `self.P` to reflect the increased uncertainty after moving the state estimate forward. This step incorporates the process noise through the matrix `self.Q`.

3. **Update (update method)**:
   - Calculates the Kalman Gain `self.G`, which determines the weighting given to the new measurement versus the predicted state.
   - Updates the state estimate `self.X` using the new measurement, adjusting the state towards the measurement based on the Kalman Gain.
   - Updates the error covariance matrix `self.P` to reflect the decreased uncertainty after incorporating the new measurement.

The Kalman Filter's strength lies in its ability to provide a statistically optimal estimate under the assumption that the noises are Gaussian. It effectively balances between the predictions and the measurements, adjusting its estimates based on the observed errors and the expected noise levels.

**Implementation Notes for YOLOv9**:
If integrating a YOLOv9 model for object detection, this Kalman Filter can be used to smooth the trajectories of detected objects. The measurement updates would come from the new detections (object positions) provided by YOLOv9, potentially improving tracking accuracy and robustness in environments with frequent scene changes or occlusions.
"""