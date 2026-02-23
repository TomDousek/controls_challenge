import optuna
import numpy as np
from pathlib import Path
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX
from controllers import myPid

MODEL_PATH = "models/tinyphysics.onnx"
# Vezmi jen část dat pro rychlost – třeba prvních 100 souborů
DATA_DIR = Path("data/")
data_files = sorted(DATA_DIR.glob("*.csv"))[:100]

model = TinyPhysicsModel(MODEL_PATH, debug=False)

def objective(trial):
    # Prohledávaný prostor
    p        = trial.suggest_float("p",          0.05, 0.8)
    i        = trial.suggest_float("i",          0.05, 0.8)
    d        = trial.suggest_float("d",        -0.05, 0.05)
    ff_weight = trial.suggest_float("ff_weight",  0.1, 0.7)
    ff_horizon = trial.suggest_int("ff_horizon",    2,  15)
    
    costs = []
    for data_path in data_files:
        controller = myPid.Controller()
        controller.p          = p
        controller.i          = i
        controller.d          = d
        controller.ff_weight  = ff_weight
        controller.ff_horizon = ff_horizon

        sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
        result = sim.rollout()
        costs.append(result['total_cost'])

    return np.mean(costs)

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Aktuální nejlepší bod je startovní – Optuna z něj bude vycházet
    study.enqueue_trial({
        "p": 0.3, "i": 0.47, "d": -0.01,
        "ff_weight": 0.4, "ff_horizon": 4
    })
    
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n=== Nejlepší parametry ===")
    print(study.best_params)
    print(f"Cost: {study.best_value:.4f}")