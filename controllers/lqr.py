from . import BaseController
import numpy as np
from tinyphysics import FPS

GAINS_PATH = "./lqr_gains.npz"

V_SLOW = 10
V_FAST = 25

class Controller(BaseController):
    """
    LQR kontroler s gain schedulingem podle vEgo.
    Koeficienty K jsou identifikované z reálných dat.
    """
    def __init__(self):
        gains = np.load(GAINS_PATH)
        self.K_slow = gains['K_slow']  # (1, 3)
        self.K_mid  = gains['K_mid']
        self.K_fast = gains['K_fast']

        self.error_integral = 0.0
        self.prev_error     = 0.0

        # Feedforward – stejný jako v PID
        self.ff_weight  = 0.366
        self.ff_horizon = 4
        self.roll_comp  = 0.621

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        dt  = 1 / FPS
        v   = state[1]

        # Výběr K podle rychlosti
        if v < V_SLOW:
            K = self.K_slow
        elif v < V_FAST:
            K = self.K_mid
        else:
            K = self.K_fast

        # Výpočet stavového vektoru
        error = target_lataccel - current_lataccel
        self.error_integral += error * dt
        self.error_integral  = np.clip(self.error_integral, -1, 1)
        error_diff = (error - self.prev_error) / dt
        self.prev_error = error

        x = np.array([error, self.error_integral, error_diff])

        # LQR akce: u = -K @ x
        lqr_action = float(-K @ x)

        # Feedforward z future_plan
        feedforward = 0.0
        if len(future_plan[0]) > 0:
            horizon = min(self.ff_horizon, len(future_plan[0]))
            weights = np.exp(-np.arange(horizon) * 0.1)
            weights /= weights.sum()
            ff_target   = np.dot(future_plan[0][:horizon], weights)
            feedforward = self.ff_weight * ff_target

        action = lqr_action + feedforward - state[0] * self.roll_comp
        return float(np.clip(action, -1, 1))