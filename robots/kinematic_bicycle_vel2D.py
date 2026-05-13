import numpy as np
import casadi as ca

"""
Created on May 13th, 2026
@author: Akshunn

@description:
Velocity-controlled kinematic bicycle model for Safety Gymnasium Racecar.

3-state model that matches the Racecar's action space [V, δ] exactly.
    State:   [x, y, θ]
    Control: [V, δ]  (velocity command, steering angle)

Dynamics:
    ẋ = V · cos(θ)
    ẏ = V · sin(θ)
    θ̇ = V · tan(δ) / L

Where L is the wheelbase (rear axle to front axle distance).
This differs from KinematicBicycle2D which uses [a, β] (acceleration, slip angle).

Reference: Safety Gymnasium racecar.xml
    - Wheelbase L = 0.325m
    - Velocity actuator: ctrlrange [-20, 20]
    - Steering actuator: ctrlrange [-0.785, 0.785] rad (±45 deg)
"""


def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class KinematicBicycleVel2D:
    """3-state kinematic bicycle with velocity control.

    State:   X = [x, y, θ]
    Control: U = [V, δ]  (velocity, steering angle)

    Dynamics:
        ẋ = V · cos(θ)
        ẏ = V · sin(θ)
        θ̇ = V · tan(δ) / L

    This model matches the Safety Gymnasium Racecar action space directly:
        action[0] = V  (rear-axle velocity command, range [-20, 20])
        action[1] = δ  (front steering angle, range [-0.785, 0.785])
    """

    def __init__(self, dt, robot_spec):
        self.dt = dt
        self.robot_spec = robot_spec

        self.robot_spec.setdefault('model', 'KinematicBicycleVel2D')

        # Racecar geometry (from racecar.xml)
        self.robot_spec.setdefault('wheel_base', 0.325)
        self.robot_spec.setdefault('body_width', 0.2)
        self.robot_spec.setdefault('radius', 0.16)  # bounding circle for collision

        # Control limits (from racecar.xml actuator ranges)
        self.robot_spec.setdefault('v_max', 5.0)
        self.robot_spec.setdefault('delta_max', 0.785)  # ~45 deg

    def f(self, X, casadi=False):
        """Drift term — zero (no autonomous motion without control)."""
        if casadi:
            return ca.vertcat(0, 0, 0)
        else:
            return np.array([0.0, 0.0, 0.0]).reshape(-1, 1)

    def g(self, X, casadi=False):
        """Input matrix. Maps [V, δ] → state derivatives.

        Note: θ̇ = V·tan(δ)/L couples both controls, so g is not purely
        a function of X. For step() we integrate directly. This g() is
        provided for interface compatibility and uses a linearized form
        where δ is treated as small (tan(δ) ≈ δ).
        """
        theta = X[2, 0]
        L = self.robot_spec['wheel_base']

        if casadi:
            g = ca.SX.zeros(3, 2)
            g[0, 0] = ca.cos(theta)
            g[1, 0] = ca.sin(theta)
            g[2, 0] = 1.0 / L  # approximate: tan(δ)/L ≈ δ/L for small δ
            return g
        else:
            return np.array([
                [np.cos(theta), 0],
                [np.sin(theta), 0],
                [1.0 / L,       0],
            ])

    def step(self, X, U, casadi=False):
        """Euler integration of dynamics.

        Args:
            X: State [x, y, θ] as (3,1) array
            U: Control [V, δ] as (2,1) array

        Returns:
            X_next: Next state (3,1) array
        """
        if casadi:
            V = U[0, 0]
            delta = U[1, 0]
            theta = X[2, 0]
            L = self.robot_spec['wheel_base']

            x_dot = V * ca.cos(theta)
            y_dot = V * ca.sin(theta)
            theta_dot = V * ca.tan(delta) / L

            X_next = X + self.dt * ca.vertcat(x_dot, y_dot, theta_dot)
            X_next[2, 0] = angle_normalize(X_next[2, 0])
            return X_next
        else:
            V = U[0, 0]
            delta = U[1, 0]
            theta = X[2, 0]
            L = self.robot_spec['wheel_base']

            # Clip controls to limits
            v_max = self.robot_spec['v_max']
            delta_max = self.robot_spec['delta_max']
            V = np.clip(V, -v_max, v_max)
            delta = np.clip(delta, -delta_max, delta_max)

            x_dot = V * np.cos(theta)
            y_dot = V * np.sin(theta)
            theta_dot = V * np.tan(delta) / L

            X_next = X + self.dt * np.array([x_dot, y_dot, theta_dot]).reshape(-1, 1)
            X_next[2, 0] = angle_normalize(X_next[2, 0])
            return X_next

    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_theta=2.0):
        """Go-to-goal controller. Outputs [V, δ] directly.

        Args:
            X: State [x, y, θ] as (3,1) array
            G: Goal [gx, gy] as (2,1) array
            d_min: Deadzone distance
            k_v: Velocity proportional gain
            k_theta: Steering proportional gain

        Returns:
            U: Control [V, δ] as (2,1) array
        """
        G = np.copy(G.reshape(-1, 1))
        v_max = self.robot_spec['v_max']
        delta_max = self.robot_spec['delta_max']

        # Distance and bearing to goal
        distance = max(np.linalg.norm(X[0:2, 0] - G[0:2, 0]) - d_min, 0.05)
        theta_des = np.arctan2(G[1, 0] - X[1, 0], G[0, 0] - X[0, 0])
        heading_error = angle_normalize(theta_des - X[2, 0])

        # Steering: P on heading error
        delta = np.clip(k_theta * heading_error, -delta_max, delta_max)

        # Speed: proportional to distance, reduced when not facing goal
        heading_scale = max(0.0, np.cos(heading_error))
        V = np.clip(k_v * distance * heading_scale, 0, v_max)

        return np.array([V, delta]).reshape(-1, 1)

    def stop(self, X):
        """Backup controller: zero velocity, zero steering."""
        return np.array([0.0, 0.0]).reshape(-1, 1)

    def has_stopped(self, X, U=None, tol=0.05):
        """Check if the robot has stopped.

        Since V is a control input (not a state), we check the last
        commanded velocity. If no U is provided, returns True (no motion).
        """
        if U is None:
            return True
        return abs(U[0, 0]) < tol

    def agent_barrier(self, X, obs, robot_radius, beta=1.1):
        """Continuous-time CBF for circular obstacle avoidance.

        h(x) = ||x - x_obs||^2 - β * d_min^2

        Note: For this velocity-controlled model, the relative degree
        structure differs from acceleration-controlled models. The CBF
        has relative degree 1 w.r.t. the velocity input V.

        Args:
            X: State [x, y, θ] as (3,1)
            obs: Obstacle [ox, oy, r_obs] as (3,) array
            robot_radius: Robot collision radius
            beta: CBF tightening parameter

        Returns:
            h: CBF value (scalar)
            h_dot: Time derivative of h (1, 1)
            dh_dot_dx: Jacobian of h_dot w.r.t. X (1, 3)
        """
        obsX = obs[0:2].reshape(2, 1)
        d_min = obs[2] + robot_radius

        h = np.linalg.norm(X[0:2] - obsX[0:2])**2 - beta * d_min**2

        # h_dot = 2*(x-ox)*ẋ + 2*(y-oy)*ẏ
        #       = 2*(x-ox)*V*cos(θ) + 2*(y-oy)*V*sin(θ)
        # Since V is a control input, h_dot depends on U.
        # For now return h_dot assuming V=0 (drift only).
        h_dot = np.array([[0.0]])

        # dh/dx = [2*(x-ox), 2*(y-oy), 0]
        dh_dx = np.array([
            2 * (X[0, 0] - obsX[0, 0]),
            2 * (X[1, 0] - obsX[1, 0]),
            0.0
        ]).reshape(1, -1)

        return h, h_dot, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.1):
        """Discrete-time CBF for circular obstacle avoidance.

        Args:
            x_k: Current state [x, y, θ] as (3,1) CasADi or numpy
            u_k: Current control [V, δ] as (2,1) CasADi or numpy
            obs: Obstacle parameters
            robot_radius: Robot collision radius
            beta: CBF tightening parameter

        Returns:
            h_k: CBF at current step
            d_h: First difference (h_{k+1} - h_k)
            dd_h: Second difference (h_{k+2} - 2*h_{k+1} + h_k)
        """
        x_k1 = self.step(x_k, u_k, casadi=True)
        x_k2 = self.step(x_k1, u_k, casadi=True)

        def h(x, obs, robot_radius, beta=1.1):
            x_obs = obs[0]
            y_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs
            return (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 - beta * d_min**2

        h_k = h(x_k, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k2 = h(x_k2, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k

        return h_k, d_h, dd_h
