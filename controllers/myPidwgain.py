from . import BaseController
import numpy as np
from tinyphysics import FPS

class Controller(BaseController):
  """
  PID controller with feedforward and gain scheduling based on vEgo.
  Optimized in Bayesian optimizer.
  """
  def __init__(self):
    # Pomalá jízda (v < 10 m/s ~ 36 km/h)
    self.p_slow = 0.402
    self.i_slow = 0.100
    self.d_slow = -0.003

    # Střední rychlost (10 <= v < 25 m/s ~ 90 km/h)
    self.p_mid = 0.361
    self.i_mid = 0.833
    self.d_mid = -0.001

    # Vysoká rychlost (v >= 25 m/s ~ 90+ km/h)
    self.p_fast = 0.156
    self.i_fast = 1.134
    self.d_fast = -0.011

    self.error_integral = 0
    self.prev_error = 0

    self.ff_weight = 0.410
    self.ff_horizon = 6
    self.roll_comp = 0.444

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    dt = 1 / FPS
    v = state[1]  # vEgo v m/s

    # Gain scheduling – vyber koeficienty podle rychlosti
    if v < 10:
      p, i, d = self.p_slow, self.i_slow, self.d_slow
    elif v < 25:
      p, i, d = self.p_mid, self.i_mid, self.d_mid
    else:
      p, i, d = self.p_fast, self.i_fast, self.d_fast

    error = target_lataccel - current_lataccel
    self.error_integral += error * dt
    self.error_integral = np.clip(self.error_integral, -1, 1)
    error_diff = (error - self.prev_error) / dt
    self.prev_error = error

    pid = p * error + i * self.error_integral + d * error_diff

    feedforward = 0.0
    if len(future_plan[0]) > 0:
      horizon = min(self.ff_horizon, len(future_plan[0]))
      weights = np.exp(-np.arange(horizon) * 0.1)
      weights /= weights.sum()
      ff_target = np.dot(future_plan[0][:horizon], weights)
      feedforward = self.ff_weight * ff_target

    action = pid + feedforward - state[0] * self.roll_comp
    return np.clip(action, -1, 1)