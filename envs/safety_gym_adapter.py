"""
Created on May 13th, 2026
@author: Akshunn

@description:
Adapter that bridges Safety Gymnasium environment internals to the
Gatekeeper/MPS collision checking interface.

Gatekeeper expects an environment object with:
    - check_collision(position, robot_radius) → bool
    - check_obstacle_collision(position, robot_radius) → (bool, idx or None)

Safety Gymnasium exposes hazard positions via task.hazards.pos and agent
state via task.agent.pos / task.data.

This adapter extracts that information and presents it in the format
Gatekeeper expects.

@required-scripts: safety_gymnasium
"""

import numpy as np


class SafetyGymAdapter:
    """Bridges Safety Gymnasium environment → Gatekeeper collision checking.

    Wraps a Safety Gymnasium env (Builder instance) to provide:
        - check_collision(): arena boundary check
        - check_obstacle_collision(): hazard proximity check
        - get_agent_state(): extract [x, y, θ] from MuJoCo data
        - get_goal_position(): extract goal [x, y]
        - get_hazard_info(): list of (x, y, radius) for all hazards
    """

    def __init__(self, env, arena_extents=None):
        """
        Args:
            env: Safety Gymnasium Builder instance
                 (the object returned by safety_gymnasium.make())
            arena_extents: Arena bounds [x_min, y_min, x_max, y_max].
                           Defaults to [-1.5, -1.5, 1.5, 1.5] for GoalLevel1.
        """
        self.env = env
        self.task = env.task

        # Arena boundaries (GoalLevel1 uses [-1.5, 1.5])
        if arena_extents is None:
            arena_extents = [-1.5, -1.5, 1.5, 1.5]
        self.arena_extents = arena_extents

    # ------------------------------------------------------------------
    # State extraction (from MuJoCo internals)
    # ------------------------------------------------------------------

    def get_agent_state(self):
        """Extract agent state [x, y, θ] from MuJoCo.

        Returns:
            np.ndarray: State vector [x, y, θ] as (3, 1) column vector.
        """
        # Position: from MuJoCo body
        pos = self.task.agent.pos  # [x, y, z]
        x, y = pos[0], pos[1]

        # Heading: from the agent's rotation matrix
        # agent.mat is a 3x3 rotation matrix. For a 2D heading on the
        # ground plane, θ = atan2(mat[1,0], mat[0,0]).
        mat = self.task.agent.mat  # 3x3 rotation matrix
        theta = np.arctan2(mat[1, 0], mat[0, 0])

        return np.array([x, y, theta]).reshape(-1, 1)

    def get_agent_velocity(self):
        """Extract agent velocity from MuJoCo.

        Returns:
            float: Forward speed (scalar).
        """
        vel = self.task.agent.vel  # [vx, vy, vz] in world frame
        pos = self.task.agent.pos
        mat = self.task.agent.mat

        # Project world velocity onto body forward direction
        theta = np.arctan2(mat[1, 0], mat[0, 0])
        v_forward = vel[0] * np.cos(theta) + vel[1] * np.sin(theta)
        return v_forward

    def get_goal_position(self):
        """Get goal [x, y] position.

        Returns:
            np.ndarray: Goal position as (2, 1) column vector.
        """
        goal_pos = self.task.goal.pos  # [x, y, z]
        return np.array([goal_pos[0], goal_pos[1]]).reshape(-1, 1)

    def get_hazard_info(self):
        """Get all hazard positions and radii.

        Returns:
            list of np.ndarray: Each element is [x, y, radius].
        """
        hazards = []
        hazard_obj = self.task.hazards
        hazard_radius = hazard_obj.size  # 0.2 for GoalLevel1

        for h_pos in hazard_obj.pos:
            hazards.append(np.array([h_pos[0], h_pos[1], hazard_radius]))

        return hazards

    # ------------------------------------------------------------------
    # Gatekeeper collision interface
    # ------------------------------------------------------------------

    def check_collision(self, position, robot_radius=0.0):
        """Check if position is outside the arena boundary.

        Gatekeeper calls this to check boundary violations.

        Args:
            position: [x, y] array-like
            robot_radius: Robot collision radius

        Returns:
            bool: True if collision with boundary.
        """
        x, y = position[0], position[1]
        x_min, y_min, x_max, y_max = self.arena_extents

        if (x - robot_radius < x_min or x + robot_radius > x_max or
                y - robot_radius < y_min or y + robot_radius > y_max):
            return True
        return False

    def check_obstacle_collision(self, position, robot_radius=0.0):
        """Check if position collides with any hazard.

        Gatekeeper calls this to check static obstacle violations.
        Uses circle-circle intersection: dist < hazard_radius + robot_radius.

        Args:
            position: [x, y] array-like
            robot_radius: Robot collision radius

        Returns:
            tuple: (bool, int or None) — collision flag and hazard index.
        """
        x, y = position[0], position[1]
        hazards = self.get_hazard_info()

        for i, hazard in enumerate(hazards):
            hx, hy, h_radius = hazard[0], hazard[1], hazard[2]
            dist = np.sqrt((x - hx)**2 + (y - hy)**2)
            if dist < (h_radius + robot_radius):
                return True, i

        return False, None
