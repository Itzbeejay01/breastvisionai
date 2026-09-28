"""
Tests for the prediction app.
"""

from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth import get_user_model


class PredictionAPITest(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="test-admin", password="test-password-123"
        )

    def test_health_endpoint(self):
        from django.test import Client
        client = Client()
        client.force_login(self.user)
        response = client.get("/api/health/")
        self.assertEqual(response.status_code, 200)

    def test_models_endpoint(self):
        from django.test import Client
        client = Client()
        response = client.get("/api/models/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 3)  # PSO-selected 3 models
        names = [m["name"] for m in data]
        self.assertIn("EfficientNet", names)
        self.assertIn("ResNet", names)
        self.assertIn("VGG16", names)

    def test_ensemble_endpoint(self):
        from django.test import Client
        client = Client()
        response = client.get("/api/ensemble/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("selected_models", data)
        self.assertIn("fusion_weights", data)

    def test_history_empty(self):
        from django.test import Client
        client = Client()
        response = client.get("/api/history/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 0)

    def test_predict_raw_image(self):
        """
        Upload a test image with image_type=raw and verify the PSO pipeline
        returns a prediction with model breakdown and heatmap.
        """
        from django.test import Client
        import os

        test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets_split", "test", "benign", "benign10.png",
        )
        if not os.path.exists(test_image_path):
            self.skipTest(f"Test image not found: {test_image_path}")

        with open(test_image_path, "rb") as f:
            image_file = SimpleUploadedFile(
                name="benign10.png",
                content=f.read(),
                content_type="image/png",
            )

        client = Client()
        client.force_login(self.user)
        response = client.post(
            "/api/predict/",
            {"image": image_file, "image_type": "raw"},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("model_breakdown", data)
        self.assertIn("pso_weights", data)
        self.assertIn("heatmap_base64", data)
        self.assertIn("ensemble_method", data)

    def test_predict_processed_image(self):
        """
        Upload a test image with image_type=processed and verify prediction.
        """
        from django.test import Client
        import os

        test_image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets_split", "test", "malignant", "malignant10005.png",
        )
        if not os.path.exists(test_image_path):
            self.skipTest(f"Test image not found: {test_image_path}")

        with open(test_image_path, "rb") as f:
            image_file = SimpleUploadedFile(
                name="malignant10005.png",
                content=f.read(),
                content_type="image/png",
            )

        client = Client()
        client.force_login(self.user)
        response = client.post(
            "/api/predict/",
            {"image": image_file, "image_type": "processed"},
            format="multipart",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)



class ExplainabilityUnitTest(TestCase):
    def test_model_agreement_summary(self):
        from prediction.services.explainability import calculate_model_agreement

        agreement = calculate_model_agreement(
            {
                "EfficientNet": 0.91,
                "ResNet": 0.82,
                "VGG16": 0.73,
            },
            "malignant",
        )

        self.assertEqual(agreement["support_count"], 3)
        self.assertEqual(agreement["total_models"], 3)
        self.assertTrue(agreement["unanimous"])
        self.assertAlmostEqual(agreement["probability_spread"], 0.18)

    def test_explanation_endpoint_returns_cached_payload(self):
        from django.test import Client
        from prediction.models import Prediction

        prediction = Prediction.objects.create(
            prediction_result="benign",
            confidence=0.8,
            fused_probability=0.2,
            prediction_probability=0.2,
            model_breakdown={
                "EfficientNet": {"probability": 0.2, "weight": 1 / 3},
                "ResNet": {"probability": 0.3, "weight": 1 / 3},
                "VGG16": {"probability": 0.1, "weight": 1 / 3},
            },
            pso_weights={},
            fusion_weights={
                "EfficientNet": 1 / 3,
                "ResNet": 1 / 3,
                "VGG16": 1 / 3,
            },
            ensemble_method="decision_level_late_fusion",
            explanation_data={
                "version": "xai-v1",
                "target_class": "benign",
            },
        )

        client = Client()
        client.force_login(self.user)
        response = client.post(f"/api/history/{prediction.id}/explain/", {})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["cached"])
        self.assertEqual(data["explanation_data"]["version"], "xai-v1")
