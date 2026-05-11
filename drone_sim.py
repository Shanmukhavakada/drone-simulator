import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

# =====================================================
# CONSTANTS
# =====================================================

g = 9.81
mass = 1.2

max_thrust_per_motor = 8.0

dt = 0.02

# =====================================================
# DRONE STATE
# =====================================================

state = {
    "pos": np.array([0.0, 0.0, 0.0]),
    "vel": np.array([0.0, 0.0, 0.0]),

    "angles": np.array([0.0, 0.0, 0.0]),
    "ang_vel": np.array([0.0, 0.0, 0.0])
}

# motor outputs
motors = np.array([0.5, 0.5, 0.5, 0.5])

# smooth motor simulation
motor_targets = np.array([0.5, 0.5, 0.5, 0.5])

# =====================================================
# KEYBOARD INPUT
# =====================================================

keys = set()

def on_press(key):
    try:
        keys.add(key.char)
    except:
        keys.add(key)

def on_release(key):
    try:
        keys.discard(key.char)
    except:
        keys.discard(key)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()

# =====================================================
# PID MEMORY
# =====================================================

pid_z_i = 0.0
pid_z_prev = 0.0

# =====================================================
# VISUALIZATION
# =====================================================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

trail_x = []
trail_y = []
trail_z = []

# =====================================================
# INPUT SYSTEM
# =====================================================

def get_inputs():

    vx_cmd = 0.0
    vy_cmd = 0.0

    yaw_cmd = 0.0
    throttle_cmd = 0.0

    move_speed = 1.5

    if 'w' in keys:
        vx_cmd += move_speed

    if 's' in keys:
        vx_cmd -= move_speed

    if 'd' in keys:
        vy_cmd += move_speed

    if 'a' in keys:
        vy_cmd -= move_speed

    if 'q' in keys:
        yaw_cmd = -0.3

    if 'e' in keys:
        yaw_cmd = 0.3

    if keyboard.Key.up in keys:
        throttle_cmd = 0.4

    if keyboard.Key.down in keys:
        throttle_cmd = -0.4

    return vx_cmd, vy_cmd, yaw_cmd, throttle_cmd

# =====================================================
# ROTATION MATRIX
# =====================================================

def rotation_matrix(r, p, y):

    cr = np.cos(r)
    sr = np.sin(r)

    cp = np.cos(p)
    sp = np.sin(p)

    cy = np.cos(y)
    sy = np.sin(y)

    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp, sr*cp, cr*cp]
    ])

# =====================================================
# FLIGHT CONTROLLER
# =====================================================

def controller():

    global pid_z_i
    global pid_z_prev
    global motor_targets

    # ------------------------------------------
    # INPUTS
    # ------------------------------------------

    vx_cmd, vy_cmd, yaw_cmd, throttle_cmd = get_inputs()

    # ------------------------------------------
    # CURRENT STATE
    # ------------------------------------------

    x, y, z = state["pos"]

    vx, vy, vz = state["vel"]

    roll, pitch, yaw = state["angles"]

    # =================================================
    # VELOCITY CONTROL
    # =================================================

    K_vel = 0.25

    target_pitch = np.clip(
        K_vel * vx_cmd,
        -0.25,
        0.25
    )

    target_roll = np.clip(
        -K_vel * vy_cmd,
        -0.25,
        0.25
    )

    # =================================================
    # ALTITUDE HOLD PID
    # =================================================

    target_z = 1.5

    error_z = target_z - z

    Kp = 1.4
    Ki = 0.15
    Kd = 0.9

    pid_z_i += error_z * dt
    pid_z_i = np.clip(pid_z_i, -2, 2)

    derivative = (error_z - pid_z_prev) / dt
    pid_z_prev = error_z

    altitude_output = (
        Kp * error_z +
        Ki * pid_z_i +
        Kd * derivative
    )

    # =================================================
    # ATTITUDE STABILIZATION
    # =================================================

    K_att = 4.5

    roll_error = target_roll - roll
    pitch_error = target_pitch - pitch

    roll_output = K_att * roll_error
    pitch_output = K_att * pitch_error

    # =================================================
    # BASE HOVER THRUST
    # =================================================

    hover = (mass * g) / (4 * max_thrust_per_motor)

    # =================================================
    # MOTOR MIXING
    # =================================================

    motor_targets[0] = (
        hover +
        altitude_output +
        pitch_output +
        roll_output -
        yaw_cmd +
        throttle_cmd
    )

    motor_targets[1] = (
        hover +
        altitude_output +
        pitch_output -
        roll_output +
        yaw_cmd +
        throttle_cmd
    )

    motor_targets[2] = (
        hover +
        altitude_output -
        pitch_output +
        roll_output +
        yaw_cmd +
        throttle_cmd
    )

    motor_targets[3] = (
        hover +
        altitude_output -
        pitch_output -
        roll_output -
        yaw_cmd +
        throttle_cmd
    )

    motor_targets[:] = np.clip(motor_targets, 0, 1)

# =====================================================
# PHYSICS ENGINE
# =====================================================

def physics_step():

    global motors

    # =================================================
    # MOTOR INERTIA (REALISTIC)
    # =================================================

    motor_response = 0.08

    motors += (motor_targets - motors) * motor_response

    # =================================================
    # ROTATION
    # =================================================

    r, p, y = state["angles"]

    R = rotation_matrix(r, p, y)

    # =================================================
    # THRUST
    # =================================================

    total_thrust = np.sum(motors) * max_thrust_per_motor

    thrust_vector = np.array([0, 0, total_thrust])

    world_force = R @ thrust_vector

    gravity_force = np.array([0, 0, -mass * g])

    # =================================================
    # ACCELERATION
    # =================================================

    acceleration = (world_force + gravity_force) / mass

    # =================================================
    # DRAG / AIR RESISTANCE
    # =================================================

    drag = np.array([
        -1.8 * state["vel"][0],
        -1.8 * state["vel"][1],
        -0.9 * state["vel"][2]
    ])

    acceleration += drag

    # =================================================
    # UPDATE LINEAR MOTION
    # =================================================

    state["vel"] += acceleration * dt
    state["pos"] += state["vel"] * dt

    # =================================================
    # ANGULAR PHYSICS
    # =================================================

    torque = np.array([

        motors[0] + motors[2]
        - motors[1] - motors[3],

        motors[0] + motors[1]
        - motors[2] - motors[3],

        motors[0] + motors[3]
        - motors[1] - motors[2]
    ])

    # angular damping
    torque += -2.5 * state["ang_vel"]

    state["ang_vel"] += torque * dt

    state["angles"] += state["ang_vel"] * dt

# =====================================================
# VISUALIZATION
# =====================================================

def update_plot():

    ax.clear()

    x, y, z = state["pos"]

    trail_x.append(x)
    trail_y.append(y)
    trail_z.append(z)

    ax.plot(
        trail_x,
        trail_y,
        trail_z,
        linewidth=2
    )

    ax.scatter(
        x,
        y,
        z,
        s=80
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 4)

    ax.set_title("Advanced Drone Simulator V2")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.pause(0.001)

# =====================================================
# MAIN LOOP
# =====================================================

def run():

    steps = 3000

    for _ in range(steps):

        controller()

        physics_step()

        update_plot()

    plt.show()

# =====================================================
# START
# =====================================================

if __name__ == "__main__":

    run()