"""
pso_from_cache.py

Run PSO model selection using cached predictions from the
five already-trained CNN base learners.

Pipeline:

    Cached CNN predictions
            ↓
        PSO selection
            ↓
      Best 3 of 5 models
            ↓
     Save selected models

The test set is loaded only for verification/reporting.
It is NEVER used by PSO fitness.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pso_model_selection import (
    PSOModelSelector,
    MODEL_NAMES
)


# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = "pso_cache"
OUTPUT_DIR = "PSO_Result"

SWARM_SIZE = 20
ITERATIONS = 30
RESTARTS = 5
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CACHE
# ============================================================

print("=" * 70)
print("PSO MODEL SELECTION")
print("=" * 70)

print("\nLoading cached predictions...")

val_probs = np.load(
    os.path.join(
        CACHE_DIR,
        "val_probs.npy"
    )
)

test_probs = np.load(
    os.path.join(
        CACHE_DIR,
        "test_probs.npy"
    )
)

y_val = np.load(
    os.path.join(
        CACHE_DIR,
        "val_labels.npy"
    )
)

y_test = np.load(
    os.path.join(
        CACHE_DIR,
        "test_labels.npy"
    )
)


# ============================================================
# VALIDATE CACHE
# ============================================================

if val_probs.shape[0] != 5:
    raise ValueError(
        "val_probs.npy must contain predictions "
        "for exactly 5 CNN models."
    )

if test_probs.shape[0] != 5:
    raise ValueError(
        "test_probs.npy must contain predictions "
        "for exactly 5 CNN models."
    )

if val_probs.shape[1] != len(y_val):
    raise ValueError(
        "Validation prediction count does not "
        "match validation labels."
    )

if test_probs.shape[1] != len(y_test):
    raise ValueError(
        "Test prediction count does not "
        "match test labels."
    )


print(f"\nValidation predictions: {val_probs.shape}")
print(f"Test predictions:       {test_probs.shape}")

print(
    f"Validation labels: "
    f"{len(y_val)}"
)

print(
    f"Test labels: "
    f"{len(y_test)}"
)

print("\nModel order:")

for i, model in enumerate(MODEL_NAMES):
    print(f"  {i}: {model}")


# ============================================================
# RUN PSO MULTIPLE TIMES
# ============================================================

all_results = []

print("\n" + "=" * 70)
print("RUNNING PSO")
print("=" * 70)

for restart in range(RESTARTS):

    seed = RANDOM_SEED + restart

    print(
        f"\n--- PSO Restart "
        f"{restart + 1}/{RESTARTS} "
        f"(seed={seed}) ---"
    )

    selector = PSOModelSelector(
        n_models=5,
        n_select=3,
        swarm_size=SWARM_SIZE,
        iterations=ITERATIONS,
        inertia_start=0.9,
        inertia_end=0.4,
        cognitive=2.0,
        social=2.0,
        seed=seed,
    )

    result = selector.optimize(
        val_probs,
        y_val
    )

    result["restart"] = restart + 1
    result["seed"] = seed

    all_results.append(result)

    print(
        f"\nRestart {restart + 1} result:"
    )

    print(
        f"  Selected models: "
        f"{result['selected_models']}"
    )

    print(
        f"  Fitness: "
        f"{result['fitness']:.6f}"
    )

    print(
        f"  Accuracy: "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"  F1: "
        f"{result['f1']:.6f}"
    )

    print(
        f"  AUC: "
        f"{result['auc']:.6f}"
    )


# ============================================================
# SELECT BEST PSO RUN
# ============================================================

best_result = max(
    all_results,
    key=lambda x: x["fitness"]
)


print("\n" + "=" * 70)
print("FINAL PSO RESULT")
print("=" * 70)

print(
    f"\nBest restart: "
    f"{best_result['restart']}"
)

print(
    f"Selected 3 models:"
)

for model in best_result["selected_models"]:
    print(f"  - {model}")

print(
    f"\nValidation Fitness: "
    f"{best_result['fitness']:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{best_result['accuracy']:.6f}"
)

print(
    f"Validation F1: "
    f"{best_result['f1']:.6f}"
)

print(
    f"Validation AUC: "
    f"{best_result['auc']:.6f}"
)


# ============================================================
# EXHAUSTIVE SEARCH CHECK
# ============================================================

"""
Because 5 choose 3 = 10, we can independently evaluate all
10 combinations.

This is NOT used by PSO.

It is a verification experiment to determine whether PSO found
the globally best combination according to the same fitness
function.
"""

from itertools import combinations
from pso_model_selection import (
    PSOModelSelector
)

print("\n" + "=" * 70)
print("EXHAUSTIVE 3-OF-5 VERIFICATION")
print("=" * 70)

exhaustive_results = []

for combination in combinations(range(5), 3):

    metrics = PSOModelSelector.calculate_fitness(
        np.array(combination),
        val_probs,
        y_val
    )

    models = [
        MODEL_NAMES[i]
        for i in combination
    ]

    exhaustive_results.append({
        "indices": combination,
        "models": models,
        **metrics
    })


exhaustive_best = max(
    exhaustive_results,
    key=lambda x: x["fitness"]
)

print(
    "\nExhaustive best combination:"
)

print(
    exhaustive_best["models"]
)

print(
    f"Fitness: "
    f"{exhaustive_best['fitness']:.6f}"
)

print(
    f"Accuracy: "
    f"{exhaustive_best['accuracy']:.6f}"
)

print(
    f"F1: "
    f"{exhaustive_best['f1']:.6f}"
)

print(
    f"AUC: "
    f"{exhaustive_best['auc']:.6f}"
)

pso_indices = tuple(
    sorted(
        best_result["selected_indices"]
    )
)

exhaustive_indices = tuple(
    sorted(
        exhaustive_best["indices"]
    )
)

if pso_indices == exhaustive_indices:

    print(
        "\n✓ PSO found the same optimal "
        "3-model combination as exhaustive search."
    )

    pso_matches_exhaustive = True

else:

    print(
        "\n⚠ PSO did not find the exhaustive "
        "global optimum."
    )

    print(
        f"PSO:        {best_result['selected_models']}"
    )

    print(
        f"Exhaustive: {exhaustive_best['models']}"
    )

    pso_matches_exhaustive = False


# ============================================================
# SAVE SELECTION
# ============================================================

selection_data = {

    "model_order": MODEL_NAMES,

    "selected_models":
        best_result["selected_models"],

    "selected_indices":
        [
            int(i)
            for i in best_result["selected_indices"]
        ],

    "dropped_models":
        [
            model
            for model in MODEL_NAMES
            if model not in best_result["selected_models"]
        ],

    "fitness": best_result["fitness"],

    "validation_metrics": {
        "accuracy":
            best_result["accuracy"],
        "f1":
            best_result["f1"],
        "auc":
            best_result["auc"],
    },

    "fitness_definition":
        "(Accuracy + F1 + AUC) / 3",

    "pso_parameters": {
        "swarm_size": SWARM_SIZE,
        "iterations": ITERATIONS,
        "restarts": RESTARTS,
        "inertia_start": 0.9,
        "inertia_end": 0.4,
        "cognitive_coefficient": 2.0,
        "social_coefficient": 2.0,
        "random_seed": RANDOM_SEED,
    },

    "exhaustive_verification": {
        "best_models":
            exhaustive_best["models"],
        "best_fitness":
            exhaustive_best["fitness"],
        "pso_matches_exhaustive":
            pso_matches_exhaustive,
    }
}


selection_path = os.path.join(
    OUTPUT_DIR,
    "pso_selected_models.json"
)

with open(
    selection_path,
    "w"
) as f:

    json.dump(
        selection_data,
        f,
        indent=4
    )


# ============================================================
# SAVE ALL COMBINATIONS
# ============================================================

combination_path = os.path.join(
    OUTPUT_DIR,
    "all_3of5_combinations.json"
)

with open(
    combination_path,
    "w"
) as f:

    json.dump(
        exhaustive_results,
        f,
        indent=4
    )


# ============================================================
# SAVE CONVERGENCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for result in all_results:

    plt.plot(
        range(
            len(result["history"])
        ),
        result["history"],
        label=f"Restart {result['restart']}"
    )

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Best Validation Fitness"
)

plt.title(
    "PSO Convergence"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "pso_convergence.png"
)

plt.savefig(
    plot_path,
    dpi=150
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "pso_selection_metrics.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "PSO-BASED CNN MODEL SELECTION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Candidate models:\n"
    )

    for model in MODEL_NAMES:
        f.write(
            f"  - {model}\n"
        )

    f.write(
        "\nSelection objective:\n"
    )

    f.write(
        "  Select exactly 3 of the 5 CNN models.\n"
    )

    f.write(
        "\nFitness function:\n"
    )

    f.write(
        "  Fitness = (Accuracy + F1 + AUC) / 3\n"
    )

    f.write(
        "\nBest PSO selection:\n"
    )

    for model in best_result["selected_models"]:
        f.write(
            f"  - {model}\n"
        )

    f.write(
        "\nValidation metrics:\n"
    )

    f.write(
        f"  Fitness:  {best_result['fitness']:.6f}\n"
    )

    f.write(
        f"  Accuracy: {best_result['accuracy']:.6f}\n"
    )

    f.write(
        f"  F1:       {best_result['f1']:.6f}\n"
    )

    f.write(
        f"  AUC:      {best_result['auc']:.6f}\n"
    )

    f.write(
        "\nExhaustive-search verification:\n"
    )

    f.write(
        f"  Best combination: "
        f"{', '.join(exhaustive_best['models'])}\n"
    )

    f.write(
        f"  Best fitness: "
        f"{exhaustive_best['fitness']:.6f}\n"
    )

    f.write(
        f"  PSO matches exhaustive optimum: "
        f"{pso_matches_exhaustive}\n"
    )


print("\n" + "=" * 70)
print("PSO SELECTION COMPLETE")
print("=" * 70)

print(
    f"\nSelection saved to:\n"
    f"{selection_path}"
)

print(
    f"\nNext step:"
    f"\nRun stacking_ensemble.py"
)