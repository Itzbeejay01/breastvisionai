from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import tempfile

from prediction.services import explainability
from prediction.services.class_labels import malignant_column
from prediction.services.model_registry import PSOModelRegistry


class ClassMappingTest(TestCase):
    def test_malignant_probability_column_uses_estimator_labels(self):
        model = SimpleNamespace(classes_=np.asarray(["malignant", "benign"]))
        self.assertEqual(malignant_column(model), 0)

    def test_registry_schema_matches_three_and_four_feature_models(self):
        original_initialized = PSOModelRegistry._initialized
        original_model = PSOModelRegistry.meta_learner
        original_selection = PSOModelRegistry.selection
        try:
            PSOModelRegistry._initialized = True
            PSOModelRegistry.selection = {"selected_models": ["EfficientNet", "ResNet", "VGG16"]}
            PSOModelRegistry.meta_learner = SimpleNamespace(n_features_in_=3)
            self.assertEqual(PSOModelRegistry.get_meta_feature_names(), ["EfficientNet", "ResNet", "VGG16"])
            PSOModelRegistry.meta_learner = SimpleNamespace(n_features_in_=4)
            self.assertEqual(
                PSOModelRegistry.get_meta_feature_names(),
                ["EfficientNet", "ResNet", "VGG16", "Late Fusion"],
            )
        finally:
            PSOModelRegistry._initialized = original_initialized
            PSOModelRegistry.meta_learner = original_model
            PSOModelRegistry.selection = original_selection

    def test_missing_model_artifact_fails_with_a_controlled_error(self):
        from prediction.services import model_registry
        original_selection = model_registry.PSOModelRegistry.selection
        original_models = model_registry.PSOModelRegistry.models
        try:
            model_registry.PSOModelRegistry.selection = {"selected_models": ["EfficientNet"]}
            model_registry.PSOModelRegistry.models = {}
            with tempfile.TemporaryDirectory() as temp_dir, patch.object(
                model_registry, "MODEL_DIR", temp_dir
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Model file not found"):
                    model_registry.PSOModelRegistry._load_models()
        finally:
            model_registry.PSOModelRegistry.selection = original_selection
            model_registry.PSOModelRegistry.models = original_models


class TreeShapTest(TestCase):
    def test_extracts_malignant_output_and_rejects_truncated_features(self):
        explanation = SimpleNamespace(
            values=np.asarray([[[0.1, 0.7], [0.2, 0.8], [0.3, 0.9]]]),
            base_values=np.asarray([[0.4, 0.6]]),
        )
        values, base = explainability._extract_tree_shap(explanation, 3, 1)
        np.testing.assert_allclose(values, [0.7, 0.8, 0.9])
        self.assertAlmostEqual(base, 0.6)
        with self.assertRaises(ValueError):
            explainability._extract_tree_shap(explanation, 4, 1)

    def test_probability_additivity_and_exact_three_feature_order(self):
        model = SimpleNamespace(
            classes_=np.asarray([0, 1]),
            n_features_in_=3,
            predict_proba=lambda x: np.asarray([[0.25, 0.75]]),
        )
        names = ["EfficientNet", "ResNet", "VGG16"]
        explanation = SimpleNamespace(
            values=np.asarray([[0.1, 0.2, 0.15]]),
            base_values=np.asarray([0.3]),
        )

        def tree_explainer(*args, **kwargs):
            return lambda values, **options: explanation

        with (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_meta_feature_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_meta_learner", return_value=model),
            patch.object(explainability, "_background_features", return_value=np.ones((5, 3))),
            patch.object(explainability.shap, "TreeExplainer", side_effect=tree_explainer),
        ):
            result = explainability.explain_meta_learner(
                {"EfficientNet": 0.2, "ResNet": 0.3, "VGG16": 0.4},
                0.3,
                0.75,
            )
            invalid_result = explainability.explain_meta_learner(
                {"EfficientNet": 0.2, "ResNet": 0.3, "VGG16": 0.4},
                0.3,
                0.76,
            )
        self.assertEqual(result["target_class"], "malignant")
        self.assertEqual(result["feature_order"], names)
        self.assertTrue(result["additivity_valid"])
        self.assertAlmostEqual(result["reconstructed_probability"], 0.75)
        self.assertAlmostEqual(result["additivity_error"], 0.0)
        self.assertFalse(invalid_result["additivity_valid"])
        self.assertGreater(invalid_result["additivity_error"], 0.0)

    def test_raw_fallback_is_compared_to_decision_function(self):
        model = SimpleNamespace(
            classes_=np.asarray([0, 1]),
            n_features_in_=3,
            predict_proba=lambda x: np.asarray([[0.2, 0.8]]),
            decision_function=lambda x: np.asarray([1.25]),
        )
        names = ["EfficientNet", "ResNet", "VGG16"]
        raw_explanation = SimpleNamespace(
            values=np.zeros((1, 3)),
            base_values=np.asarray([1.25]),
        )
        calls = []

        def tree_explainer(*args, **kwargs):
            calls.append((kwargs["model_output"], kwargs["feature_perturbation"]))
            if kwargs["model_output"] == "probability":
                raise RuntimeError("probability mode unsupported")
            return lambda values, **options: raw_explanation

        with (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_meta_feature_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_meta_learner", return_value=model),
            patch.object(explainability, "_background_features", return_value=np.ones((5, 3))),
            patch.object(explainability.shap, "TreeExplainer", side_effect=tree_explainer),
            patch.object(explainability.logger, "warning") as warning,
        ):
            result = explainability.explain_meta_learner(
                {"EfficientNet": 0.2, "ResNet": 0.3, "VGG16": 0.4},
                0.3,
                0.8,
            )
        self.assertEqual(calls, [("probability", "interventional"), ("raw", "interventional")])
        warning.assert_called_once()
        self.assertEqual(result["output_space"], "raw_decision_function")
        self.assertEqual(result["additivity_comparison"], "decision_function")
        self.assertTrue(result["additivity_valid"])
        self.assertIsNone(result["reconstructed_probability"])

    def test_four_feature_baseline_uses_tree_path_dependent_raw_output(self):
        model = SimpleNamespace(
            classes_=np.asarray([0, 1]),
            n_features_in_=4,
            predict_proba=lambda x: np.asarray([[0.2, 0.8]]),
            decision_function=lambda x: np.asarray([1.25]),
        )
        names = ["EfficientNet", "ResNet", "VGG16", "Late Fusion"]
        raw_explanation = SimpleNamespace(
            values=np.zeros((1, 4)),
            base_values=np.asarray([1.25]),
        )
        with (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=names[:3]),
            patch.object(explainability.PSOModelRegistry, "get_meta_feature_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_meta_learner", return_value=model),
            patch.object(explainability, "_background_features", return_value=np.ones((5, 4))),
            patch.object(explainability.shap, "TreeExplainer", return_value=lambda values, **options: raw_explanation) as tree,
        ):
            result = explainability.explain_meta_learner(
                {"EfficientNet": 0.2, "ResNet": 0.3, "VGG16": 0.4},
                0.3,
                0.8,
            )
        self.assertEqual(tree.call_args.kwargs["model_output"], "raw")
        self.assertEqual(tree.call_args.kwargs["feature_perturbation"], "tree_path_dependent")
        self.assertEqual(result["output_space"], "raw_decision_function")
        self.assertTrue(result["additivity_valid"])

    def test_background_rejects_missing_file_wrong_dimensions_and_nonfinite_values(self):
        names = ["EfficientNet", "ResNet", "VGG16"]
        registry_patches = (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_indices", return_value=[0, 2, 3]),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_selection", return_value={
                "model_order": ["EfficientNet", "DenseNet", "ResNet", "VGG16", "Xception"]
            }),
            patch.object(explainability.PSOModelRegistry, "get_fusion_weights", return_value={name: 1 / 3 for name in names}),
            patch.object(explainability.PSOModelRegistry, "get_meta_feature_names", return_value=names),
        )
        with registry_patches[0], registry_patches[1], registry_patches[2], registry_patches[3], registry_patches[4], registry_patches[5]:
            with patch.object(explainability.os.path, "exists", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    explainability._background_features()
            with patch.object(explainability.os.path, "exists", return_value=True), patch.object(
                explainability.np, "load", return_value=np.ones(10)
            ):
                with self.assertRaises(ValueError):
                    explainability._background_features()
            for nonfinite in (np.nan, np.inf):
                invalid = np.ones((5, 8))
                invalid[0, 0] = nonfinite
                with patch.object(explainability.os.path, "exists", return_value=True), patch.object(
                    explainability.np, "load", return_value=invalid
                ):
                    with self.assertRaises(ValueError):
                        explainability._background_features()

    def test_background_sampling_is_deterministic_and_feature_ordered(self):
        names = ["EfficientNet", "ResNet", "VGG16"]
        values = np.asarray([np.arange(10) + i * 10 for i in range(5)], dtype=float)
        with (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_indices", return_value=[0, 2, 3]),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=names),
            patch.object(explainability.PSOModelRegistry, "get_selection", return_value={
                "model_order": ["EfficientNet", "DenseNet", "ResNet", "VGG16", "Xception"]
            }),
            patch.object(explainability.PSOModelRegistry, "get_fusion_weights", return_value={name: 1 / 3 for name in names}),
            patch.object(explainability.os.path, "exists", return_value=True),
            patch.object(explainability.np, "load", return_value=values),
        ):
            first = explainability._background_features(names, max_samples=4)
            second = explainability._background_features(names, max_samples=4)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first[:, 0], [0, 3, 6, 9])
        np.testing.assert_allclose(first[:, 1], [20, 23, 26, 29])


class AgreementTest(TestCase):
    def test_agreement_reports_votes_not_accuracy(self):
        result = explainability.calculate_model_agreement(
            {"EfficientNet": 0.8, "ResNet": 0.7, "VGG16": 0.2},
            "malignant",
        )
        self.assertEqual(result["support_count"], 2)
        self.assertEqual(result["total_models"], 3)
        self.assertAlmostEqual(result["agreement_ratio"], 2 / 3)
        self.assertFalse(result["unanimous"])
        self.assertEqual(result["individual_predictions"]["VGG16"], "benign")


class NormalPredictionTest(TestCase):
    def test_single_prediction_skips_gradcam_and_uses_four_feature_baseline(self):
        from prediction.services import prediction

        names = ["EfficientNet", "ResNet", "VGG16"]
        registry = SimpleNamespace(
            get_models=lambda: {name: object() for name in names},
            get_meta_learner=lambda: SimpleNamespace(
                classes_=np.asarray([0, 1]),
                predict_proba=lambda features: np.asarray([[0.6, 0.4]]),
            ),
            get_fusion_config=lambda: {"fusion_strategy": "decision_level_late_fusion"},
            get_selected_model_names=lambda: names,
            get_fusion_weights=lambda: {name: 1 / 3 for name in names},
            get_meta_feature_names=lambda: [*names, "Late Fusion"],
            get_pso_weights=lambda: {},
        )
        prepared = {
            name: (np.zeros((1, 2, 2, 3)), np.zeros((2, 2, 3)))
            for name in names
        }
        with (
            patch.object(prediction.PSOModelRegistry, "initialize", return_value=registry),
            patch.object(prediction, "preprocess_image_for_all_models", return_value=prepared),
            patch.object(prediction, "_predict_model", return_value=np.asarray([[0.4]])),
            patch.object(prediction, "generate_gradcam_for_image") as gradcam,
        ):
            result = prediction.predict_single("image.png")
        gradcam.assert_not_called()
        self.assertEqual(result["meta_feature_schema"], [*names, "Late Fusion"])
        self.assertEqual(result["prediction"], "benign")
        self.assertEqual(result["heatmap_base64"], "")
