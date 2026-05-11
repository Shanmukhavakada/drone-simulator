import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONSTANTS
# =====================================================

g = 9.81
mass = 1.2
dt = 0.02

max_thrust_per_motor = 8.0

# =====================================================
# STATE
# =====================================================

state = {
    "pos": np.array([0.0, 0.0, 0.0]),
    "vel": np.array([0.0, 0.0, 0.0]),

    "angles": np.array([0.0, 0.0, 0.0]),
    "ang_vel": np.array([0.0, 0.0, 0.0])
}

motors = np.array([0.5, 0.5, 0.5, 0.5])
motor_targets = np.array([0.5, 0.5, 0.5, 0.5])

# =====================================================
# WAYPOINTS
# =====================================================

waypoints = [
    np.array([0, 0, 1.5]),
    np.array([2, 0, 2]),
    np.array([2, 2, 2]),
    np.array([-2, 2, 1.5]),
    np.array([0, 0, 1.5])
]

current_waypoint = 0

# =====================================================
# PID MEMORY
# =====================================================

pid_z_i = 0
pid_z_prev = 0

# =====================================================
# VISUALIZATION
# =====================================================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

trail_x = []
trail_y = []
trail_z = []

# =====================================================
# ROTATION MATRIX
# =====================================================

def rotation_matrix(r, p, y):

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp, sr*cp, cr*cp]
    ])

# =====================================================
# AUTOPILOT CONTROLLER
# =====================================================

def controller():

    global current_waypoint
    global pid_z_i
    global pid_z_prev
    global motor_targets

    # ----------------------------------------
    # CURRENT WAYPOINT
    # ----------------------------------------

    target = waypoints[current_waypoint]

    tx, ty, tz = target

    x, y, z = state["pos"]
    vx, vy, vz = state["vel"]

    roll, pitch, yaw = state["angles"]

    # =================================================
    # WAYPOINT REACHED?
    # =================================================

    distance = np.linalg.norm(target - state["pos"])

    if distance < 0.3:

        current_waypoint += 1

        if current_waypoint >= len(waypoints):
            current_waypoint = 0

    # =================================================
    # POSITION CONTROL
    # =================================================

    Kp_pos = 0.35
    Kd_pos = 0.25

    x_error = tx - x
    y_error = ty - y

    vx_target = Kp_pos * x_error
    vy_target = Kp_pos * y_error

    vx_error = vx_target - vx
    vy_error = vy_target - vy

    target_pitch = np.clip(
        0.4 * vx_error,
        -0.25,
        0.25
    )

    target_roll = np.clip(
        -0.4 * vy_error,
        -0.25,
        0.25
    )

    # =================================================
    # ALTITUDE PID
    # =================================================

    z_error = tz - z

    Kp = 1.4
    Ki = 0.15
    Kd = 0.9

    pid_z_i += z_error * dt
    pid_z_i = np.clip(pid_z_i, -2, 2)

    derivative = (z_error - pid_z_prev) / dt
    pid_z_prev = z_error

    altitude_output = (
        Kp * z_error +
        Ki * pid_z_i +
        Kd * derivative
    )

    # =================================================
    # ATTITUDE CONTROL
    # =================================================

    K_att = 4.5

    roll_error = target_roll - roll
    pitch_error = target_pitch - pitch

    roll_output = K_att * roll_error
    pitch_output = K_att * pitch_error

    # =================================================
    # HOVER THRUST
    # =================================================

    hover = (mass * g) / (4 * max_thrust_per_motor)

    # =================================================
    # MOTOR MIXING
    # =================================================

    motor_targets[0] = (
        hover +
        altitude_output +
        pitch_output +
        roll_output
    )

    motor_targets[1] = (
        hover +
        altitude_output +
        pitch_output -
        roll_output
    )

    motor_targets[2] = (
        hover +
        altitude_output -
        pitch_output +
        roll_output
    )

    motor_targets[3] = (
        hover +
        altitude_output -
        pitch_output -
        roll_output
    )

    motor_targets[:] = np.clip(motor_targets, 0, 1)

# =====================================================
# PHYSICS ENGINE
# =====================================================

def physics_step():

    global motors

    # smooth motor response
    motors += (motor_targets - motors) * 0.08

    r, p, y = state["angles"]

    R = rotation_matrix(r, p, y)

    total_thrust = np.sum(motors) * max_thrust_per_motor

    thrust = np.array([0, 0, total_thrust])

    world_force = R @ thrust

    gravity_force = np.array([0, 0, -mass * g])

    acceleration = (
        world_force + gravity_force
    ) / mass

    # drag
    drag = np.array([
        -1.8 * state["vel"][0],
        -1.8 * state["vel"][1],
        -0.9 * state["vel"][2]
    ])

    acceleration += drag

    # update motion
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

    # drone trail
    ax.plot(
        trail_x,
        trail_y,
        trail_z,
        linewidth=2
    )

    # drone
    ax.scatter(
        x,
        y,
        z,
        s=100
    )

    # current waypoint
    wp = waypoints[current_waypoint]

    ax.scatter(
        wp[0],
        wp[1],
        wp[2],
        s=200,
        marker='X'
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 4)

    ax.set_title("Autonomous Drone Simulator")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.pause(0.001)

# =====================================================
# MAIN LOOP
# =====================================================

def run():

    for _ in range(5000):

        controller()

        physics_step()

        update_plot()

    plt.show()

# =====================================================
# START
# =====================================================

if __name__ == "__main__":

    run()
