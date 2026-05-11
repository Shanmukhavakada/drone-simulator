"""
Simple Quadcopter Drone Simulator (Improved Version)
----------------------------------------------------
Now includes:
- Proper PID altitude controller
- Better simulation stability
- Cleaner timestep handling
- Reduced oscillations

This remains an educational simulator (not industrial-grade), but behaves much more realistically.
"""

import numpy as np

# -----------------------------
# Physics constants
# -----------------------------
g = 9.81

# -----------------------------
# Drone parameters
# -----------------------------
mass = 1.2  # kg
max_thrust_per_motor = 7.0  # N per motor

# -----------------------------
# State
# -----------------------------
state = {
    "pos": np.array([0.0, 0.0, 0.0]),
    "vel": np.array([0.0, 0.0, 0.0]),
    "acc": np.array([0.0, 0.0, 0.0]),

    "angles": np.array([0.0, 0.0, 0.0]),
    "ang_vel": np.array([0.0, 0.0, 0.0])
}

motors = np.array([0.5, 0.5, 0.5, 0.5])

# -----------------------------
# PID state (altitude control)
# -----------------------------
pid_integral = 0.0
pid_prev_error = 0.0

# -----------------------------
# Helpers
# -----------------------------

def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp]
    ])


def compute_thrust(motors):
    return motors * max_thrust_per_motor

# -----------------------------
# Improved PID Controller
# -----------------------------


def hover_controller(target_z, dt):
    global motors, pid_integral, pid_prev_error

    z = state["pos"][2]
    error = target_z - z

    Kp = 0.9
    Ki = 0.15
    Kd = 1.2

    pid_integral += error * dt
    derivative = (error - pid_prev_error) / dt

    output = (Kp * error) + (Ki * pid_integral) + (Kd * derivative)

    base_throttle = (mass * g) / (4 * max_thrust_per_motor)

    throttle = base_throttle + output
    throttle = np.clip(throttle, 0.0, 1.0)

    motors[:] = throttle
    pid_prev_error = error
# -----------------------------
# Physics step
# -----------------------------

def step(dt):
    global state, motors

    roll, pitch, yaw = state["angles"]
    R = rotation_matrix(roll, pitch, yaw)

    thrusts = compute_thrust(motors)
    total_thrust_body = np.array([0, 0, np.sum(thrusts)])

    force_world = R @ total_thrust_body
    gravity = np.array([0, 0, -mass * g])

    acc = (force_world + gravity) / mass

    state["acc"] = acc
    state["vel"] += acc * dt
    state["pos"] += state["vel"] * dt

    # simplified rotational dynamics
    torque = np.array([
        motors[0] + motors[2] - motors[1] - motors[3],
        motors[0] + motors[1] - motors[2] - motors[3],
        motors[0] + motors[3] - motors[1] - motors[2]
    ])

    state["ang_vel"] += torque * dt
    state["angles"] += state["ang_vel"] * dt

# -----------------------------
# Simulation loop
# -----------------------------

def run_sim(steps=500, dt=0.02, target_z=1.0):
    for i in range(steps):
        hover_controller(target_z, dt)
        step(dt)

        if i % 20 == 0:
            print(f"t={i*dt:.2f}s | z={state['pos'][2]:.3f} | vel={state['vel'][2]:.3f}")


if __name__ == "__main__":
    run_sim()
"""
"""