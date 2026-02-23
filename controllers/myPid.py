from . import BaseController
import numpy as np
from tinyphysics import FPS

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.279
    self.i = 0.785
    self.d = -0.013
    self.error_integral = 0
    self.prev_error = 0
    
    self.ff_weight = 0.366
    self.ff_horizon = 4

  # state = one line of -> [0] roll_lataccel, [1] vEgo, [2] aEgo
  # future_plan = next 50 lines of -> target_lataccel [0], roll_lataccel [1], vEgo [2], aEgo [3]
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    #average = 0
    #if(len(future_plan[0][:20]) > 0):
    #  average = (np.mean(future_plan[0][:20]))
    dt = 1/FPS
    error = (target_lataccel - current_lataccel)
    self.error_integral += error * dt
    self.error_integral = np.clip(self.error_integral, -1, 1)
    error_diff = (error - self.prev_error)/dt
    self.prev_error = error
    pid = self.p * error + self.i * self.error_integral + self.d * error_diff
    
    feedforward = 0.0
    if len(future_plan[0]) > 0:
      horizon = min(self.ff_horizon, len(future_plan[0]))
      # Weighted average - closer steps have bigger impact
      weights = np.exp(-np.arange(horizon) * 0.1)
      weights /= weights.sum()
      ff_target = np.dot(future_plan[0][:horizon], weights)
      feedforward = self.ff_weight * ff_target

    action = pid + feedforward
    action = np.clip(action, -1, 1)
    return action