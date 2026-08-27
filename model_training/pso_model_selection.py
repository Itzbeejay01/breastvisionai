"""
pso_model_selection.py

PSO-based selection of the optimal 3 CNN base learners from 5 candidates.

Candidate models:
    1. EfficientNetB0
    2. DenseNet121
    3. ResNet50
    4. VGG16
    5. Xception

Fitness:
    Composite of:
        - Accuracy
        - F1-score
        - AUC-ROC

The PSO searches for the subset of exactly 3 models that maximizes:

    Fitness = (Accuracy + F1 + AUC) / 3

IMPORTANT:
    The test set must NEVER be used during PSO optimization.
    Fitness is calculated exclusively on the validation set.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


MODEL_NAMES = [
    "EfficientNet",
    "DenseNet",
    "ResNet",
    "VGG16",
    "Xception",
]


class PSOModelSelector:
    """
    Particle Swarm Optimization for selecting exactly K models.

    Each particle has N continuous position values.
    The K models with the highest position values are selected.
    """

    def __init__(
        self,
        n_models=5,
        n_select=3,
        swarm_size=20,
        iterations=30,
        inertia_start=0.9,
        inertia_end=0.4,
        cognitive=2.0,
        social=2.0,
        seed=42,
    ):
        self.n_models = n_models
        self.n_select = n_select
        self.swarm_size = swarm_size
        self.iterations = iterations

        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.cognitive = cognitive
        self.social = social

        self.rng = np.random.RandomState(seed)

    # ---------------------------------------------------------
    # Decode particle position into exactly K selected models
    # ---------------------------------------------------------

    def decode_particle(self, position):
        """
        Select exactly n_select models using the highest particle
        position values.
        """

        selected_indices = np.argsort(position)[-self.n_select:]
        selected_indices = np.sort(selected_indices)

        return selected_indices

    # ---------------------------------------------------------
    # Fitness function
    # ---------------------------------------------------------

    @staticmethod
    def calculate_fitness(
        selected_indices,
        val_probs,
        y_val,
    ):
        """
        Calculate fitness using:

            Accuracy
            F1-score
            AUC-ROC

        A simple equal-weight composite score is used:

            Fitness = (Accuracy + F1 + AUC) / 3
        """

        selected_probs = val_probs[selected_indices]

        # Simple probability averaging for evaluating
        # the candidate subset.
        ensemble_probability = np.mean(
            selected_probs,
            axis=0
        )

        ensemble_prediction = (
            ensemble_probability >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_val,
            ensemble_prediction
        )

        f1 = f1_score(
            y_val,
            ensemble_prediction,
            zero_division=0
        )

        auc = roc_auc_score(
            y_val,
            ensemble_probability
        )

        fitness = (
            accuracy +
            f1 +
            auc
        ) / 3.0

        return {
            "fitness": float(fitness),
            "accuracy": float(accuracy),
            "f1": float(f1),
            "auc": float(auc),
        }

    # ---------------------------------------------------------
    # Run PSO
    # ---------------------------------------------------------

    def optimize(self, val_probs, y_val):
        """
        Run PSO and return the best 3-model subset.
        """

        if val_probs.shape[0] != self.n_models:
            raise ValueError(
                f"Expected {self.n_models} models, "
                f"but received {val_probs.shape[0]}."
            )

        # -----------------------------------------------------
        # Initialize particle positions
        # -----------------------------------------------------

        positions = self.rng.uniform(
            0.0,
            1.0,
            size=(self.swarm_size, self.n_models)
        )

        velocities = self.rng.uniform(
            -0.1,
            0.1,
            size=(self.swarm_size, self.n_models)
        )

        # -----------------------------------------------------
        # Evaluate initial particles
        # -----------------------------------------------------

        particle_fitness = np.zeros(self.swarm_size)

        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(
            self.swarm_size,
            -np.inf
        )

        global_best_position = None
        global_best_fitness = -np.inf
        global_best_metrics = None
        global_best_indices = None

        history = []

        for i in range(self.swarm_size):

            selected = self.decode_particle(
                positions[i]
            )

            metrics = self.calculate_fitness(
                selected,
                val_probs,
                y_val
            )

            particle_fitness[i] = metrics["fitness"]

            personal_best_fitness[i] = metrics["fitness"]

            if metrics["fitness"] > global_best_fitness:

                global_best_fitness = metrics["fitness"]
                global_best_position = positions[i].copy()
                global_best_indices = selected.copy()
                global_best_metrics = metrics.copy()

        history.append(global_best_fitness)

        # -----------------------------------------------------
        # Main PSO loop
        # -----------------------------------------------------

        for iteration in range(self.iterations):

            inertia = (
                self.inertia_start
                -
                (
                    (self.inertia_start - self.inertia_end)
                    * iteration
                    / max(self.iterations - 1, 1)
                )
            )

            for i in range(self.swarm_size):

                r1 = self.rng.random(self.n_models)
                r2 = self.rng.random(self.n_models)

                # Velocity update
                velocities[i] = (
                    inertia * velocities[i]
                    +
                    self.cognitive
                    * r1
                    * (
                        personal_best_positions[i]
                        - positions[i]
                    )
                    +
                    self.social
                    * r2
                    * (
                        global_best_position
                        - positions[i]
                    )
                )

                # Position update
                positions[i] += velocities[i]

                # Keep positions bounded
                positions[i] = np.clip(
                    positions[i],
                    0.0,
                    1.0
                )

                # Decode exactly 3 models
                selected = self.decode_particle(
                    positions[i]
                )

                # Evaluate
                metrics = self.calculate_fitness(
                    selected,
                    val_probs,
                    y_val
                )

                particle_fitness[i] = metrics["fitness"]

                # Personal best
                if (
                    metrics["fitness"]
                    >
                    personal_best_fitness[i]
                ):
                    personal_best_fitness[i] = (
                        metrics["fitness"]
                    )

                    personal_best_positions[i] = (
                        positions[i].copy()
                    )

                # Global best
                if (
                    metrics["fitness"]
                    >
                    global_best_fitness
                ):
                    global_best_fitness = (
                        metrics["fitness"]
                    )

                    global_best_position = (
                        positions[i].copy()
                    )

                    global_best_indices = (
                        selected.copy()
                    )

                    global_best_metrics = (
                        metrics.copy()
                    )

            history.append(global_best_fitness)

            print(
                f"Iteration "
                f"{iteration + 1:03d}/{self.iterations} | "
                f"Best Fitness = "
                f"{global_best_fitness:.6f} | "
                f"Accuracy = "
                f"{global_best_metrics['accuracy']:.6f} | "
                f"F1 = "
                f"{global_best_metrics['f1']:.6f} | "
                f"AUC = "
                f"{global_best_metrics['auc']:.6f} | "
                f"Selected = "
                f"{[MODEL_NAMES[i] for i in global_best_indices]}"
            )

        return {
            "selected_indices": global_best_indices,
            "selected_models": [
                MODEL_NAMES[i]
                for i in global_best_indices
            ],
            "fitness": global_best_fitness,
            "accuracy": global_best_metrics["accuracy"],
            "f1": global_best_metrics["f1"],
            "auc": global_best_metrics["auc"],
            "history": history,
        }