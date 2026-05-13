"""
Created on May 13th, 2026
@author: Akshunn

@description:
Test script for Gatekeeper and MPS safety shielding in the Safety Gymnasium
SafetyRacecarGoal1-v0 environment.

The racecar navigates to a goal position while avoiding 8 hazards.
Three modes:
    - none:       Unshielded PD go-to-goal (baseline, expect hazard violations)
    - gatekeeper: Forward simulation + backward search for max nominal horizon
    - mps:        One-step nominal horizon (more conservative)

Usage:
    uv run python examples/safety_gymnasium/test_goal.py --algo gatekeeper
    uv run python examples/safety_gymnasium/test_goal.py --algo mps
    uv run python examples/safety_gymnasium/test_goal.py --algo none
    uv run python examples/safety_gymnasium/test_goal.py --algo gatekeeper --render

@required-scripts:
    safe_control/robots/kinematic_bicycle_vel2D.py
    safe_control/envs/safety_gym_adapter.py
    safe_control/shielding/gatekeeper.py
    safe_control/shielding/mps.py
"""

import argparse
import numpy as np

import safety_gymnasium

from safe_control.robots.kinematic_bicycle_vel2D import KinematicBicycleVel2D
from safe_control.envs.safety_gym_adapter import SafetyGymAdapter
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS
from safe_control.position_control.backup_controller import BackupController


# =============================================================================
# Racecar Backup Controller (Stop)
# =============================================================================

class RacecarStopController(BackupController):
    """Backup controller for velocity-controlled racecar: command V=0, δ=0."""

    def __init__(self, robot_spec, dt):
        super().__init__(robot_spec, dt)

    def compute_control(self, state, target=None):
        """Return [V=0, δ=0] — stop immediately.

        Args:
            state: Current state [x, y, θ] as (3,1)
            target: Not used

        Returns:
            Control [0, 0] as (2,1)
        """
        return np.array([[0.0], [0.0]])

    def simulate_trajectory(self, initial_state, target, horizon, friction=1.0):
        """Forward simulate stopping trajectory (robot stays in place).

        Args:
            initial_state: [x, y, θ] as (3,1)
            target: Not used
            horizon: Number of steps
            friction: Not used

        Returns:
            trajectory: (3 x horizon+1) array
        """
        state = np.array(initial_state).flatten()
        trajectory = np.zeros((3, horizon + 1))
        trajectory[:, 0] = state
        # V=0 → robot doesn't move, all future states = initial state
        for i in range(horizon):
            trajectory[:, i + 1] = state
        return trajectory


# =============================================================================
# Configuration
# =============================================================================

ROBOT_SPEC = {
    'model': 'KinematicBicycleVel2D',
    'wheel_base': 0.325,
    'radius': 0.16,      # Bounding circle for collision checking
    'v_max': 5.0,
    'delta_max': 0.785,   # ~45 deg
}

SIM_CONFIG = {
    'dt': 0.04,              # Safety Gymnasium timestep (see builder.py)
    'max_steps': 1000,       # Max steps per episode
    'backup_horizon': 2.0,   # seconds
    'nominal_horizon': 2.0,  # seconds (for Gatekeeper)
    'event_offset': 0.04,    # re-evaluation interval (= dt)
    'safety_margin': 0.02,   # extra buffer for collision checking
}


# =============================================================================
# Main Simulation
# =============================================================================

def run_episode(algo='gatekeeper', render=False, seed=None):
    """Run a single episode with the specified shielding algorithm.

    Args:
        algo: 'gatekeeper', 'mps', or 'none'
        render: Whether to render the 3D environment
        seed: Random seed for environment reset

    Returns:
        dict: Episode statistics
    """
    # --- Create environment ---
    render_mode = 'human' if render else None
    env = safety_gymnasium.make('SafetyRacecarGoal1-v0', render_mode=render_mode)

    # --- Reset ---
    obs, info = env.reset(seed=seed)

    # --- Create adapter ---
    adapter = SafetyGymAdapter(env.unwrapped)

    # --- Create robot dynamics model ---
    dt = SIM_CONFIG['dt']
    robot = KinematicBicycleVel2D(dt, ROBOT_SPEC.copy())

    # --- Setup shielding ---
    if algo in ('gatekeeper', 'mps'):
        if algo == 'gatekeeper':
            shielding = Gatekeeper(
                robot=robot,
                robot_spec=robot.robot_spec,
                dt=dt,
                backup_horizon=SIM_CONFIG['backup_horizon'],
                event_offset=SIM_CONFIG['event_offset'],
                nominal_horizon=SIM_CONFIG['nominal_horizon'],
                safety_margin=SIM_CONFIG['safety_margin'],
            )
            print(f"Using GATEKEEPER shielding")
        else:
            shielding = MPS(
                robot=robot,
                robot_spec=robot.robot_spec,
                dt=dt,
                backup_horizon=SIM_CONFIG['backup_horizon'],
                event_offset=SIM_CONFIG['event_offset'],
                safety_margin=SIM_CONFIG['safety_margin'],
            )
            print(f"Using MPS shielding")

        # Set backup controller (stop)
        backup = RacecarStopController(robot.robot_spec, dt)
        shielding.set_backup_controller(backup, target=None)

        # Set environment for collision checking
        shielding.set_environment(adapter)

        # Set nominal controller (go-to-goal via forward propagation)
        goal_pos = adapter.get_goal_position()
        shielding.nominal_controller = lambda state: robot.nominal_input(state, goal_pos)
    else:
        shielding = None
        print(f"Using NO shielding (baseline)")

    # --- Simulation loop ---
    total_cost = 0.0
    total_reward = 0.0
    nominal_steps = 0
    backup_steps = 0

    print(f"\nRunning episode (max {SIM_CONFIG['max_steps']} steps)...")
    print(f"{'Step':>6} {'x':>7} {'y':>7} {'θ':>7} {'V_cmd':>7} {'δ_cmd':>7} {'Cost':>6} {'Mode':>10}")
    print("-" * 70)

    for step in range(SIM_CONFIG['max_steps']):
        # Extract state from MuJoCo
        state = adapter.get_agent_state()  # [x, y, θ] as (3,1)
        goal = adapter.get_goal_position()  # [gx, gy] as (2,1)

        # Compute nominal control
        u_nominal = robot.nominal_input(state, goal)  # [V, δ] as (2,1)

        if shielding is not None:
            # Update goal for nominal controller (goal may change after reaching one)
            shielding.nominal_controller = lambda s, g=goal: robot.nominal_input(s, g)

            # Forward propagation mode: let Gatekeeper generate nominal trajectory
            # and validate it. We set the nominal controller above.
            u_safe = shielding.solve_control_problem(state.flatten())
            u_safe = np.array(u_safe).flatten()

            if shielding.is_using_backup():
                backup_steps += 1
            else:
                nominal_steps += 1
        else:
            u_safe = u_nominal.flatten()
            nominal_steps += 1

        # Map to Safety Gymnasium action: [velocity, steering_angle]
        action = np.array([u_safe[0], u_safe[1]])

        # Step environment
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_cost += cost
        total_reward += reward

        # Print status
        if step % 50 == 0 or cost > 0:
            x, y, theta = state.flatten()
            mode = "BACKUP" if (shielding and shielding.is_using_backup()) else "NOMINAL"
            cost_marker = " ***" if cost > 0 else ""
            print(f"{step:6d} {x:7.3f} {y:7.3f} {theta:7.3f} {u_safe[0]:7.3f} {u_safe[1]:7.3f} {cost:6.1f} {mode:>10}{cost_marker}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step} ({'terminated' if terminated else 'truncated'})")
            break

    # --- Summary ---
    print("\n" + "=" * 50)
    print(f"Algorithm:    {algo.upper()}")
    print(f"Total Steps:  {step + 1}")
    print(f"Total Cost:   {total_cost:.1f}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Nominal Steps: {nominal_steps}")
    print(f"Backup Steps:  {backup_steps}")
    if nominal_steps + backup_steps > 0:
        print(f"Nominal Ratio: {nominal_steps / (nominal_steps + backup_steps):.1%}")
    print("=" * 50)

    env.close()

    return {
        'algo': algo,
        'total_cost': total_cost,
        'total_reward': total_reward,
        'total_steps': step + 1,
        'nominal_steps': nominal_steps,
        'backup_steps': backup_steps,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Gatekeeper/MPS on Safety Gymnasium Racecar')
    parser.add_argument('--algo', type=str, default='gatekeeper',
                        choices=['gatekeeper', 'mps', 'none'],
                        help='Shielding algorithm to use')
    parser.add_argument('--render', action='store_true',
                        help='Enable 3D rendering')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    results = run_episode(algo=args.algo, render=args.render, seed=args.seed)
