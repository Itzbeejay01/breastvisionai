"""
Model registry for PSO-selected CNN models and the GradientBoosting meta-learner.

Loads all 3 PSO-selected Keras models and the GB meta-learner once at startup
and caches them in memory. Does NOT reload models per request.
"""

import os
import json

import numpy as np
import joblib

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "breastvisionai.settings")

import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(BASE_DIR, "models")
PSO_RESULT_DIR = os.path.join(BASE_DIR, "PSO_Result")
STACKING_RESULT_DIR = os.path.join(BASE_DIR, "Stacking_Result")

MODEL_FILES = {
    "EfficientNet": "efficientnet_final_model.keras",
    "ResNet": "resnet_final_model.keras",
    "VGG16": "vgg16_final_model.keras",
}

MODEL_NAMES_ALL = [
    "EfficientNet",
    "DenseNet",
    "ResNet",
    "VGG16",
    "Xception",
]

IMAGE_SIZE = (224, 224)


class PSOModelRegistry:
    """
    Singleton that loads and caches:
      - 3 PSO-selected Keras CNN models
      - GradientBoostingClassifier meta-learner
      - PSO selection configuration (selected models, weights, metrics)
      - Fusion configuration (late fusion weights)
    """

    _instance = None
    _initialized = False

    models = {}          # {model_name: tf.keras.Model}
    meta_learner = None  # joblib GradientBoostingClassifier
    selection = None     # dict from pso_selected_models.json
    fusion_config = None # dict from fusion_configuration.json
    pso_weights = None   # {model_name: weight} for all 5 models (PSO-optimized)

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return cls
        cls._load_selection()
        cls._load_fusion_config()
        cls._load_models()
        cls._load_meta_learner()
        cls._compute_pso_weights()
        cls._initialized = True
        return cls

    @classmethod
    def _load_selection(cls):
        path = os.path.join(PSO_RESULT_DIR, "pso_selected_models.json")
        with open(path, "r") as f:
            cls.selection = json.load(f)

    @classmethod
    def _load_fusion_config(cls):
        path = os.path.join(STACKING_RESULT_DIR, "fusion_configuration.json")
        with open(path, "r") as f:
            cls.fusion_config = json.load(f)

    @classmethod
    def _load_models(cls):
        selected = cls.selection["selected_models"]
        for name in selected:
            model_file = MODEL_FILES.get(name)
            if model_file is None:
                raise FileNotFoundError(f"No model file mapping for {name}")
            model_path = os.path.join(MODEL_DIR, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            cls.models[name] = load_model(model_path)
            print(f"  Loaded model: {name} from {model_path}")

    @classmethod
    def _load_meta_learner(cls):
        path = os.path.join(STACKING_RESULT_DIR, "pso_stacked_gb_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GradientBoosting meta-learner not found at {path}. "
                f"Run 'python model_training/stacking_ensemble.py' first."
            )
        cls.meta_learner = joblib.load(path)
        print(f"  Loaded GB meta-learner from {path}")

    @classmethod
    def _compute_pso_weights(cls):
        """
        Compute equal-weight fusion weights for the selected 3 models
        (matching the stacking_ensemble.py late fusion strategy).

        Also loads PSO-optimized full weights from pso_weights.json if available.
        """
        selected = cls.selection["selected_models"]
        n = len(selected)
        cls.fusion_weights = {name: 1.0 / n for name in selected}

        weights_path = os.path.join(PSO_RESULT_DIR, "pso_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                pso_data = json.load(f)
            cls.pso_weights = pso_data.get("optimal_weights", {})

    @classmethod
    def get_models(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.models

    @classmethod
    def get_meta_learner(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.meta_learner

    @classmethod
    def get_selection(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.selection

    @classmethod
    def get_fusion_config(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.fusion_config

    @classmethod
    def get_fusion_weights(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.fusion_weights

    @classmethod
    def get_pso_weights(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.pso_weights or {}

    @classmethod
    def get_selected_model_names(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.selection["selected_models"]

    @classmethod
    def get_selected_indices(cls):
        if not cls._initialized:
            cls.initialize()
        return cls.selection["selected_indices"]
