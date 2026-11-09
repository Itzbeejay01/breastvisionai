from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from prediction.services import explainability


class ConsensusGradCamTest(TestCase):
    def setUp(self):
        self.names = ["EfficientNet", "ResNet", "VGG16"]
        self.prediction = SimpleNamespace(
            id=7,
            prediction_result="benign",
            image_type="raw",
            uploaded_image=SimpleNamespace(image=SimpleNamespace(path="/tmp/image.png")),
        )
        self.preprocessed = {
            name: (np.zeros((1, 2, 2, 3)), np.zeros((2, 2, 3), dtype=np.uint8))
            for name in self.names
        }

    def run_visual(self, resized_shape=(2, 2)):
        heatmap_calls = []
        target_calls = []

        def make_artifacts(batch, original, model, target_class):
            target_calls.append((model, target_class))
            scale = (self.names.index(model) + 1) / 10
            return {
                "heatmap": np.full((2, 2), scale, dtype=np.float32),
                "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                "last_conv_layer": "conv",
            }

        def resize(array, *args, **kwargs):
            result = np.resize(array, (*resized_shape, 1))
            return SimpleNamespace(numpy=lambda: result)

        def overlay(image, heatmap):
            heatmap_calls.append(np.asarray(heatmap))
            return np.zeros_like(image)

        with (
            patch.object(explainability.PSOModelRegistry, "initialize", return_value=explainability.PSOModelRegistry),
            patch.object(explainability.PSOModelRegistry, "get_selected_model_names", return_value=self.names),
            patch.object(explainability.PSOModelRegistry, "get_fusion_weights", return_value={n: 1 / 3 for n in self.names}),
            patch.object(explainability, "preprocess_image_for_all_models", return_value=self.preprocessed),
            patch.object(explainability, "generate_gradcam_artifacts", side_effect=make_artifacts),
            patch.object(explainability.tf.image, "resize", side_effect=resize),
            patch.object(explainability, "_save_overlay", return_value="/api/image.png"),
            patch.object(explainability, "overlay_heatmap_on_image", side_effect=overlay),
        ):
            result = explainability.build_visual_explanation(self.prediction)
        return result, target_calls, heatmap_calls

    def test_three_models_receive_final_target_and_consensus_is_normalized(self):
        result, targets, heatmaps = self.run_visual()
        self.assertEqual(targets, [(name, "benign") for name in self.names])
        self.assertEqual(set(result["models"]), set(self.names))
        self.assertEqual(heatmaps[0].shape, (2, 2))
        self.assertAlmostEqual(float(heatmaps[0].max()), 1.0)
        self.assertAlmostEqual(float(heatmaps[0].min()), 1.0)

    def test_consensus_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            self.run_visual(resized_shape=(2, 3))
