from . import BaseController
import numpy as np
import copy
from tinyphysics import TinyPhysicsModel, FPS, CONTEXT_LENGTH, MAX_ACC_DELTA
from tinyphysics import State
from collections import namedtuple

MODEL_PATH = "./models/tinyphysics.onnx"

class Controller(BaseController):
  """
  PID controller with feedforward weigthed future lat_acc is used for computing actions.
  Other actions are created with random noise and subsequently a MPC model is used to 
  compute future impact chosen actions have on system. 
  """
  def __init__(self,):
    self.p = 0.279
    self.i = 0.785
    self.d = -0.013
    self.error_integral = 0
    self.prev_error = 0
    
    self.ff_weight = 0.366
    self.ff_horizon = 4
    
    self.mpc_horizon = 3
    self.mpc_samples = 3
    self.mpc_noise   = 0.3
    
    self.physics_model = TinyPhysicsModel(MODEL_PATH, debug=False)

    # history is maintained in cotroller
    self.state_history   = []
    self.action_history  = []
    self.lataccel_history = []
    self.future_ground_truth = [] # future_plan of mpc_horizon length

  # state = one line of -> [0] roll_lataccel, [1] vEgo, [2] aEgo
  # future_plan = next 50 lines of -> target_lataccel [0], roll_lataccel [1], vEgo [2], aEgo [3]
  
  def count_action(self, target_lataccel, current_lataccel, state, future_plan):
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

    action = pid + feedforward - state[0] * 0.53
    action = np.clip(action, -1, 1)
    #print(f"error: {error}, action: {action}")
    return action
  
  def predict_forward(self, state_history, action_history, lataccel_history, first_action,
                      current_lataccel, future_plan, horizon):
    total_cost = 0
    predicted_lat_acc = []
    
    
    if len(action_history) < CONTEXT_LENGTH:
      return float('inf')
      
    action_history.append(first_action)
    
    for i in range(horizon):
      predicted_lat_acc.append(self.physics_model.get_current_lataccel(
                sim_states=state_history[-CONTEXT_LENGTH:],
                actions=action_history[-CONTEXT_LENGTH:],
                past_preds=lataccel_history[-CONTEXT_LENGTH:]
            ))
      state = State(
          roll_lataccel=future_plan[1][i],
          v_ego=future_plan[2][i],
          a_ego=future_plan[3][i]
      )
      state_history.append(state)
      lataccel_history.append(predicted_lat_acc[-1])
      if (i == horizon - 1):
        break
      action_history.append(self.count_action(self.future_ground_truth[i][0], predicted_lat_acc[i], state, [row[i:] for row in self.future_ground_truth]))
   
    target = np.array(self.future_ground_truth[0])[:self.mpc_horizon]
    pred = np.array(predicted_lat_acc)[:self.mpc_horizon]  
    #lat_accel_cost = np.mean((self.future_ground_truth[0][:self.mpc_horizon] - predicted_lat_acc)**2) * 100
    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(predicted_lat_acc) * FPS)**2) * 100
    total_cost = lat_accel_cost * 50 + jerk_cost
    
    print(total_cost)
    
    return total_cost
    
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.state_history.append(state)
    self.lataccel_history.append(current_lataccel) 
    self.future_ground_truth = [sublist[:(self.mpc_horizon+self.ff_horizon)] for sublist in future_plan]
    
    pid_action = self.count_action(target_lataccel, current_lataccel, state, future_plan)
    noise = pid_action * np.random.uniform(-0.3, 0.3)
    
    if len(self.state_history) < CONTEXT_LENGTH:
      self.action_history.append(pid_action)
      return float(pid_action)
    
    ActionsTuple = namedtuple("Actions", ["action", "act_w_noise1", "act_w_noise2"])
    Actions = ActionsTuple(pid_action, pid_action + noise, pid_action - noise) 
    
    best_action = pid_action
    best_cost = float('inf')
    
    if len(future_plan[0]) > 0:
      self.mpc_horizon = min(self.mpc_horizon, len(future_plan[0]))
      for action in Actions:
        cost = self.predict_forward(self.state_history, self.action_history, self.lataccel_history,
                                    action, current_lataccel, future_plan, self.mpc_horizon)
        if cost < best_cost:
          best_cost = cost
          best_action = action
      
    print("\n")
    self.action_history.append(best_action)
    return float(best_action)