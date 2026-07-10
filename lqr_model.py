"""
Ishara -- wheeled-inverted-pendulum plant model + LQR gain solver (Stage 1).

Computes the LQR gain vector K = [k_theta, k_theta_dot, k_x, k_x_dot] from the
robot's PHYSICAL parameters (mass, CoM height, wheel radius, ...) plus the Q/R
weights. Runs on the Pi (has numpy/scipy); the resulting K is pushed to the
STM32 via the LQRK serial command, where u = -K.x runs at the fast loop rate.

Fixed-height model for Stage 1: the pendulum length = CoM height, treated as a
constant. Stage 2 will recompute K as the servo/body height changes.

Design choices / honesty:
  * This is the STANDARD linearized cart-pole (wheeled inverted pendulum) model
    about the upright equilibrium. It is an APPROXIMATION -- motor dynamics,
    wheel inertia (until measured), and friction are omitted or lumped.
  * Wheel inertia defaults to 0 (unknown); when measured, pass it in and the
    model uses it.
  * K is only as good as the parameters + Q/R. It is a STARTING POINT for
    hardware tuning / PSO, not a guaranteed-optimal controller. The firmware
    keeps the PID running alongside, so a poor K does not mean the robot has no
    controller.

Requires numpy + scipy. If scipy is unavailable, solve_lqr raises ImportError
with a clear message rather than returning a bogus K.
"""

import math

G = 9.81


def build_state_space(mass_kg, com_h_m, wheel_r_m, wheel_i_kgm2=0.0):
    """
    Linearized wheeled-inverted-pendulum state space about upright.

    State x = [theta, theta_dot, x, x_dot]
        theta   : body tilt from vertical (rad)
        theta_dot
        x       : wheel/base position (m)
        x_dot
    Input u = wheel force / drive effort (N, mapped to PWM downstream).

    Returns (A, B) as nested lists (numpy not required by callers that only
    forward them). Uses the standard small-angle cart-pole linearization:

        theta_ddot = (g/L) * theta - (1/L) * (u/M_eff)      [pendulum]
        x_ddot     = (u / M_eff)                            [base]

    where L = com_h_m, M_eff folds in wheel inertia as an effective mass at the
    wheel: M_eff = mass + I_wheel / r^2 (0 when inertia unknown).
    """
    L = max(com_h_m, 1e-3)                       # guard divide-by-zero
    r = max(wheel_r_m, 1e-3)
    M_eff = mass_kg + (wheel_i_kgm2 / (r * r) if wheel_i_kgm2 > 0 else 0.0)

    A = [
        [0.0, 1.0, 0.0, 0.0],
        [G / L, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    B = [
        [0.0],
        [-1.0 / (L * M_eff)],
        [0.0],
        [1.0 / M_eff],
    ]
    return A, B


def solve_lqr(mass_kg, com_h_m, wheel_r_m, wheel_i_kgm2=0.0,
              q_theta=1.0, q_theta_dot=1.0, q_x=1.0, q_x_dot=1.0, r=1.0):
    """
    Solve the continuous-time LQR for the plant defined by the physical params
    and the Q (diagonal) / R weights. Returns K = [k_theta, k_theta_dot, k_x,
    k_x_dot] in the SAME state units as the firmware (theta in DEGREES, since
    the estimator reports degrees; we convert the radian-domain gains here).

    Raises ImportError if numpy/scipy are missing (no silent bogus fallback).
    """
    try:
        import numpy as np
        from scipy.linalg import solve_continuous_are
    except ImportError as e:
        raise ImportError(
            "solve_lqr needs numpy + scipy (pip install numpy scipy). "
            "Original: %s" % e)

    A_l, B_l = build_state_space(mass_kg, com_h_m, wheel_r_m, wheel_i_kgm2)
    A = np.array(A_l, dtype=float)
    B = np.array(B_l, dtype=float)
    Q = np.diag([float(q_theta), float(q_theta_dot), float(q_x), float(q_x_dot)])
    R = np.array([[float(r)]])

    P = solve_continuous_are(A, B, Q, R)
    K_rad = np.linalg.inv(R) @ (B.T @ P)      # 1x4, gains for theta in RADIANS
    K_rad = K_rad.flatten()

    # firmware reports theta/theta_dot in DEGREES -> divide the angle-domain
    # gains by (180/pi) so u is unchanged: k_deg * deg = k_rad * rad.
    deg = math.pi / 180.0
    k_theta     = float(K_rad[0] * deg)
    k_theta_dot = float(K_rad[1] * deg)
    k_x         = float(K_rad[2])
    k_x_dot     = float(K_rad[3])
    return [k_theta, k_theta_dot, k_x, k_x_dot]
