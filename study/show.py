import optuna
import optuna.visualization as vis

study = optuna.load_study(
    study_name="pid_gain_scheduling",
    storage="sqlite:///optuna_pid.db"
)

print(f"Počet dokončených trialů: {len(study.trials)}")
print(f"\n=== Nejlepší parametry ===")
print(study.best_params)
print(f"Cost: {study.best_value:.4f}")

vis.plot_optimization_history(study).write_html("opt_history.html")
vis.plot_param_importances(study).write_html("opt_importances.html")
vis.plot_slice(study).write_html("opt_slice.html")
vis.plot_contour(study, params=["p_slow", "p_fast"]).write_html("opt_contour_p.html")
vis.plot_contour(study, params=["p_mid", "ff_weight"]).write_html("opt_contour_ff.html")

print("\nHTML soubory uloženy.")