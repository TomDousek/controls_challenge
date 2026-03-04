from . import BaseController
import numpy as np
from tinyphysics import TinyPhysicsModel, FPS, CONTEXT_LENGTH, MAX_ACC_DELTA
from tinyphysics import State
from collections import namedtuple

MODEL_PATH = "./models/tinyphysics.onnx"

class Controller(BaseController):
  """
  PID controller with feedforward weighted future lat_acc is used for computing actions.
  Other actions are created with random noise and subsequently a MPC model is used to
  compute future impact chosen actions have on system.
  """
  def __init__(self):
    self.p = 0.279
    self.i = 0.785
    self.d = -0.013
    self.error_integral = 0
    self.prev_error = 0
    self.count = 0

    self.ff_weight = 0.366
    self.ff_horizon = 4

    self.mpc_horizon = 3
    self.mpc_noise   = 0.3

    self.physics_model = TinyPhysicsModel(MODEL_PATH, debug=False)

    # History is maintained in controller
    self.state_history    = []
    self.action_history   = []
    self.lataccel_history = []

  # state       = one line of   -> [0] roll_lataccel, [1] vEgo, [2] aEgo
  # future_plan = next 50 lines -> [0] target_lataccel, [1] roll_lataccel, [2] vEgo, [3] aEgo

  def get_pid_snapshot(self):
    return {
      'error_integral': self.error_integral,
      'prev_error':     self.prev_error
    }

  def restore_pid_snapshot(self, snapshot):
    self.error_integral = snapshot['error_integral']
    self.prev_error     = snapshot['prev_error']

  def count_action(self, target_lataccel, current_lataccel, state, future_plan):
    dt = 1 / FPS
    error = target_lataccel - current_lataccel
    self.error_integral += error * dt
    self.error_integral = np.clip(self.error_integral, -1, 1)
    error_diff = (error - self.prev_error) / dt
    self.prev_error = error

    pid = self.p * error + self.i * self.error_integral + self.d * error_diff

    feedforward = 0.0
    if len(future_plan[0]) > 0:
      horizon = min(self.ff_horizon, len(future_plan[0]))
      weights = np.exp(-np.arange(horizon) * 0.1)
      weights /= weights.sum()
      ff_target = np.dot(future_plan[0][:horizon], weights)
      feedforward = self.ff_weight * ff_target

    action = pid + feedforward - state[0] * 0.53
    return np.clip(action, -1, 1)

  def predict_forward(self, state_history, action_history, lataccel_history,
                      first_action, current_lataccel, future_plan, horizon):
    if len(action_history) < CONTEXT_LENGTH:
      return float('inf')

    # Kopie historií – nikdy neměň skutečný stav
    s_hist = list(state_history)
    a_hist = list(action_history)
    l_hist = list(lataccel_history)

    # Přidej první kandidátní akci
    a_hist.append(first_action)

    predicted = []
    prev_lat = current_lataccel

    for i in range(horizon):
      # Predikuj lat_acc pomocí fyzikálního modelu
      pred = self.physics_model.get_current_lataccel(
        sim_states=s_hist[-CONTEXT_LENGTH:],
        actions=a_hist[-CONTEXT_LENGTH:],
        past_preds=l_hist[-CONTEXT_LENGTH:]
      )
      pred = np.clip(pred, prev_lat - MAX_ACC_DELTA, prev_lat + MAX_ACC_DELTA)
      predicted.append(pred)
      l_hist.append(pred)

      if i < horizon - 1:
        # Budoucí stav z future_plan
        next_state = State(
          roll_lataccel=future_plan[1][i] if i < len(future_plan[1]) else 0.0,
          v_ego=future_plan[2][i]         if i < len(future_plan[2]) else 0.0,
          a_ego=future_plan[3][i]         if i < len(future_plan[3]) else 0.0
        )
        s_hist.append(next_state)

        # Spočítej další akci pomocí skutečného PID ale obnov stav po výpočtu
        pid_snap = self.get_pid_snapshot()
        next_target = future_plan[0][i] if i < len(future_plan[0]) else 0.0
        next_action = self.count_action(
          next_target, pred, next_state,
          [row[i:] for row in future_plan]
        )
        self.restore_pid_snapshot(pid_snap)

        a_hist.append(next_action)

      prev_lat = pred

    # Výpočet cost stejně jako v tinyphysics.compute_cost
    target_arr = np.array([
      future_plan[0][i] for i in range(min(horizon, len(future_plan[0])))
    ])
    pred_arr = np.array(predicted[:len(target_arr)])
    lat_cost  = np.mean((target_arr - pred_arr) ** 2) * 100
    jerk_cost = np.mean((np.diff(predicted) * FPS) ** 2) * 100 if len(predicted) > 1 else 0.0

    return lat_cost * 50 + jerk_cost

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # Aktualizuj skutečnou historii
    self.state_history.append(state)
    self.lataccel_history.append(current_lataccel)

    # Spočítej PID akci – tím se aktualizuje error_integral a prev_error
    pid_action = self.count_action(target_lataccel, current_lataccel, state, future_plan)

    # Warm-up: čekej na dostatek historie
    if len(self.state_history) < CONTEXT_LENGTH:
      self.action_history.append(pid_action)
      return float(pid_action)

    # Kandidátní akce
    candidates = np.linspace(pid_action * 0.7, pid_action * 1.3, 5)

    best_action = pid_action
    best_cost   = float('inf')
    count = 0

    if len(future_plan[0]) > 0:
      horizon = min(self.mpc_horizon, len(future_plan[0]))

      # Ulož PID snapshot před smyčkou – všichni kandidáti startují ze stejného bodu
      pid_snapshot = self.get_pid_snapshot()

      for action in candidates:
        # Obnov PID stav před každým kandidátem
        self.restore_pid_snapshot(pid_snapshot)

        cost = self.predict_forward(
          self.state_history, self.action_history, self.lataccel_history,
          action, current_lataccel, future_plan, horizon
        )
        if cost < best_cost:
          best_cost   = cost
          best_action = action

      # Obnov PID stav na původní hodnotu po MPC hodnocení
      self.restore_pid_snapshot(pid_snapshot)
    
    self.action_history.append(best_action)
    return float(best_action)