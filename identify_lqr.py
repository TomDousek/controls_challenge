import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import solve_discrete_are
from tinyphysics import FPS, CONTROL_START_IDX, TinyPhysicsModel, TinyPhysicsSimulator
from controllers import myPidwgain

DATA_DIR   = Path("data/")
data_files = sorted(DATA_DIR.glob("*.csv"))[:200]

MODEL_PATH = "models/tinyphysics.onnx"
DT    = 1 / FPS
V_SLOW = 10
V_FAST = 25

data_slow, data_mid, data_fast = [], [], []

model = TinyPhysicsModel(MODEL_PATH, debug=False)

for i, path in enumerate(data_files):
    print(f"Zpracovávám {i+1}/{len(data_files)}: {path.name}")

    controller = myPidwgain.Controller()
    sim = TinyPhysicsSimulator(model, str(path), controller=controller, debug=False)
    sim.rollout()

    # Vezmi data PO control start idx kde je skutečný error
    targets  = np.array(sim.target_lataccel_history[CONTROL_START_IDX:])
    currents = np.array(sim.current_lataccel_history[CONTROL_START_IDX:])
    actions  = np.array(sim.action_history[CONTROL_START_IDX:])
    states   = sim.state_history[CONTROL_START_IDX:]

    # Rekonstruuj stavový vektor [error, integral, diff]
    error_integral = 0.0
    prev_error     = 0.0
    x_list = []

    for k in range(len(targets)):
        error = targets[k] - currents[k]
        error_integral += error * DT
        error_integral  = np.clip(error_integral, -1, 1)
        error_diff = (error - prev_error) / DT
        prev_error = error
        x_list.append(np.array([error, error_integral, error_diff]))

    # Sestavení trojic (x_k, u_k, x_k+1)
    for k in range(len(x_list) - 1):
        x_k  = x_list[k]
        u_k  = actions[k] if k < len(actions) else 0.0
        x_k1 = x_list[k + 1]
        v    = states[k].v_ego

        row = (x_k, float(u_k), x_k1)
        if v < V_SLOW:
            data_slow.append(row)
        elif v < V_FAST:
            data_mid.append(row)
        else:
            data_fast.append(row)

print(f"\nNasbíráno: slow={len(data_slow)}, mid={len(data_mid)}, fast={len(data_fast)}")


def identify_system(data, label):
    X  = np.array([d[0] for d in data])
    U  = np.array([d[1] for d in data]).reshape(-1, 1)
    X1 = np.array([d[2] for d in data])

    XU = np.hstack([X, U])
    Theta, _, _, _ = np.linalg.lstsq(XU, X1, rcond=None)
    Theta = Theta.T

    A = Theta[:, :3]
    B = Theta[:, 3:]

    # Ověř kvalitu fitu
    X1_pred = (Theta @ XU.T).T
    r2 = 1 - np.sum((X1 - X1_pred)**2) / np.sum((X1 - np.mean(X1, axis=0))**2)

    print(f"\n[{label}] {len(data)} vzorků, R²={r2:.4f}")
    print(f"  A =\n{np.round(A, 4)}")
    print(f"  B =\n{np.round(B, 4)}")
    return A, B


def compute_lqr(A, B, Q, R, label):
    try:
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        print(f"  [{label}] K = {np.round(K, 4)}")
        return K
    except Exception as e:
        print(f"  [{label}] Riccati selhala: {e}, používám pseudoinverzi")
        K = np.linalg.pinv(B) @ A
        return K.reshape(1, -1)


Q = np.diag([50.0, 1.0, 0.1])
R = np.array([[0.1]])

A_slow, B_slow = identify_system(data_slow, "SLOW")
A_mid,  B_mid  = identify_system(data_mid,  "MID")
A_fast, B_fast = identify_system(data_fast, "FAST")

K_slow = compute_lqr(A_slow, B_slow, Q, R, "SLOW")
K_mid  = compute_lqr(A_mid,  B_mid,  Q, R, "MID")
K_fast = compute_lqr(A_fast, B_fast, Q, R, "FAST")

np.savez("lqr_gains.npz", K_slow=K_slow, K_mid=K_mid, K_fast=K_fast)
print("\nK matice uloženy do lqr_gains.npz")