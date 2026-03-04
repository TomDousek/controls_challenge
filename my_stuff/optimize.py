import optuna
import numpy as np
from pathlib import Path
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
from controllers import myPidwgain

MODEL_PATH = "models/tinyphysics.onnx"
DATA_DIR   = Path("data/")
data_files = sorted(DATA_DIR.glob("*.csv"))[:100]

model = TinyPhysicsModel(MODEL_PATH, debug=False)


def objective(trial):
    p_slow     = trial.suggest_float("p_slow",     0.05, 0.8)
    i_slow     = trial.suggest_float("i_slow",     0.05, 1.2)
    d_slow     = trial.suggest_float("d_slow",    -0.05, 0.0)
    p_mid      = trial.suggest_float("p_mid",      0.05, 0.8)
    i_mid      = trial.suggest_float("i_mid",      0.05, 1.2)
    d_mid      = trial.suggest_float("d_mid",     -0.05, 0.0)
    p_fast     = trial.suggest_float("p_fast",     0.05, 0.8)
    i_fast     = trial.suggest_float("i_fast",     0.05, 1.2)
    d_fast     = trial.suggest_float("d_fast",    -0.05, 0.0)
    ff_weight  = trial.suggest_float("ff_weight",  0.1,  0.7)
    ff_horizon = trial.suggest_int(  "ff_horizon", 2,    10)
    roll_comp  = trial.suggest_float("roll_comp",  0.0,  1.0)

    controller = myPidwgain.Controller()
    controller.p_slow     = p_slow
    controller.i_slow     = i_slow
    controller.d_slow     = d_slow
    controller.p_mid      = p_mid
    controller.i_mid      = i_mid
    controller.d_mid      = d_mid
    controller.p_fast     = p_fast
    controller.i_fast     = i_fast
    controller.d_fast     = d_fast
    controller.ff_weight  = ff_weight
    controller.ff_horizon = ff_horizon
    controller.roll_comp  = roll_comp

    costs = []
    for data_file in data_files:
        sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
        costs.append(sim.rollout()['total_cost'])
    return np.mean(costs)


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage="sqlite:///optuna_pid.db",
        study_name="pid_gain_scheduling",
        load_if_exists=True
    )

    # Startovní bod – původní hodnoty před gain schedulingem
    study.enqueue_trial({
        "p_slow": 0.279, "i_slow": 0.785, "d_slow": -0.013,
        "p_mid":  0.279, "i_mid":  0.785, "d_mid":  -0.013,
        "p_fast": 0.279, "i_fast": 0.785, "d_fast": -0.013,
        "ff_weight": 0.366, "ff_horizon": 4, "roll_comp": 0.53
    })

    study.optimize(objective, n_trials=200, show_progress_bar=True)

    # Výsledky
    print("\n=== Nejlepší parametry ===")
    print(study.best_params)
    print(f"Cost: {study.best_value:.4f}")

    # Vizualizace
    import optuna.visualization as vis

    vis.plot_optimization_history(study).write_html("opt_history.html")
    vis.plot_param_importances(study).write_html("opt_importances.html")
    vis.plot_slice(study).write_html("opt_slice.html")
    vis.plot_contour(study, params=["p_slow", "p_fast"]).write_html("opt_contour_p.html")
    vis.plot_contour(study, params=["p_mid",  "ff_weight"]).write_html("opt_contour_ff.html")

    print("\nVizualizace uloženy jako HTML soubory.")