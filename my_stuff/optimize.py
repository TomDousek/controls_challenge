import optuna
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# ── Cesty – uprav podle prostředí ─────────────────────────────────────────────
if os.path.exists("/content/drive"):
    BASE    = Path("/content/drive/MyDrive/controls_challenge")
    DB_PATH = "sqlite:////content/drive/MyDrive/controls_challenge/optuna_pid2.db"
else:
    BASE    = Path(".")
    DB_PATH = "sqlite:///optuna_pid2.db"

MODEL_PATH = str(BASE / "models/tinyphysics.onnx")
DATA_DIR   = BASE / "data"

all_files = sorted(DATA_DIR.glob("*.csv"))
print(f"Celkem souborů: {len(all_files)}")

SAMPLE_SIZE = 600
N_TRIALS    = 300


# ── Top-level funkce pro multiprocessing ──────────────────────────────────────
def evaluate_single(args):
    data_file, params, model_path = args
    try:
        import onnxruntime as ort

        # GPU patch
        original = ort.InferenceSession
        def patched(model, options=None, providers=None):
            available = ort.get_available_providers()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                        if 'CUDAExecutionProvider' in available \
                        else ['CPUExecutionProvider']
            return original(model, options, providers)
        ort.InferenceSession = patched

        from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
        from controllers import myPidwgain

        model = TinyPhysicsModel(model_path, debug=False)
        controller = myPidwgain.Controller()
        for k, v in params.items():
            setattr(controller, k, v)

        sim = TinyPhysicsSimulator(model, str(data_file), controller=controller, debug=False)
        return sim.rollout()['total_cost']
    except Exception as e:
        return float('inf')


# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial):
    params = {
        'p_slow':     trial.suggest_float("p_slow",     0.05, 0.8),
        'i_slow':     trial.suggest_float("i_slow",     0.05, 1.2),
        'd_slow':     trial.suggest_float("d_slow",    -0.05, 0.0),
        'p_mid':      trial.suggest_float("p_mid",      0.05, 0.8),
        'i_mid':      trial.suggest_float("i_mid",      0.05, 1.2),
        'd_mid':      trial.suggest_float("d_mid",     -0.05, 0.0),
        'p_fast':     trial.suggest_float("p_fast",     0.05, 0.8),
        'i_fast':     trial.suggest_float("i_fast",     0.05, 1.2),
        'd_fast':     trial.suggest_float("d_fast",    -0.05, 0.0),
        'ff_weight':  trial.suggest_float("ff_weight",  0.1,  0.7),
        'ff_horizon': trial.suggest_int(  "ff_horizon", 2,    10),
        'roll_comp':  trial.suggest_float("roll_comp",  0.0,  1.0),
    }

    sampled = np.random.choice(all_files, size=min(SAMPLE_SIZE, len(all_files)), replace=False)
    args = [(f, params, MODEL_PATH) for f in sampled]

    n_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        costs = list(executor.map(evaluate_single, args))

    costs = [c for c in costs if c != float('inf')]
    return float(np.mean(costs)) if costs else float('inf')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=None),
        storage=DB_PATH,
        study_name="pid_gain_scheduling_v2",
        load_if_exists=True
    )

    # Přidej startovní bod jen pokud studie nemá žádné trialy
    if len(study.trials) == 0:
        study.enqueue_trial({
            "p_slow": 0.148, "i_slow": 0.949, "d_slow": -0.021,
            "p_mid":  0.149, "i_mid":  0.902, "d_mid":  -0.010,
            "p_fast": 0.152, "i_fast": 1.076, "d_fast": -0.001,
            "ff_weight": 0.408, "ff_horizon": 5, "roll_comp": 0.621
        })
        print("Nová studie – přidán startovní bod z předchozí optimalizace.")
    else:
        print(f"Navazuji na existující studii ({len(study.trials)} dokončených trialů).")

    print(f"Spouštím optimalizaci: {N_TRIALS} trialů × {SAMPLE_SIZE} souborů")
    print(f"Databáze: {DB_PATH}")
    print(f"CPU jádra: {multiprocessing.cpu_count()}\n")

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n=== Nejlepší parametry ===")
    print(study.best_params)
    print(f"Cost: {study.best_value:.4f}")

    # Vizualizace
    import optuna.visualization as vis

    out = BASE if os.path.exists("/content/drive") else Path(".")
    vis.plot_optimization_history(study).write_html(str(out / "opt_history.html"))
    vis.plot_param_importances(study).write_html(str(out / "opt_importances.html"))
    vis.plot_slice(study).write_html(str(out / "opt_slice.html"))
    vis.plot_contour(study, params=["p_slow", "p_fast"]).write_html(str(out / "opt_contour_p.html"))
    vis.plot_contour(study, params=["p_mid", "ff_weight"]).write_html(str(out / "opt_contour_ff.html"))

    print("\nVizualizace uloženy.")