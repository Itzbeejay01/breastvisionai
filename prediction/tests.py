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
        self.assertEqual(len(data), 5)  # Five models are listed; three are selected.
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
        client.force_login(self.user)
        response = client.get("/api/history/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 0)

    def test_sensitive_routes_require_authentication(self):
        from django.test import Client
        from prediction.models import Prediction
        prediction = Prediction.objects.create(
            prediction_result="benign", confidence=0.8, fused_probability=0.2,
            prediction_probability=0.2, model_breakdown={}, pso_weights={},
            explanation_data={"secret": "cached"},
        )
        client = Client()
        for path, method in [
            ("/api/history/", "get"),
            (f"/api/history/{prediction.id}/", "get"),
            (f"/api/history/{prediction.id}/explain/", "post"),
            (f"/api/report/{prediction.id}/", "get"),
        ]:
            response = getattr(client, method)(path)
            self.assertIn(response.status_code, (401, 403))

    def test_history_and_actions_are_owner_scoped_with_admin_override(self):
        from django.test import Client
        from prediction.models import Prediction
        other = get_user_model().objects.create_user(username="other-user", password="test-password-123")
        record = Prediction.objects.create(
            owner=other, prediction_result="benign", confidence=0.8,
            fused_probability=0.2, prediction_probability=0.2,
            model_breakdown={}, pso_weights={}, explanation_data={"secret": True},
        )
        client = Client()
        client.force_login(self.user)
        self.assertEqual(client.get("/api/history/").json()["count"], 0)
        self.assertEqual(client.get(f"/api/history/{record.id}/").status_code, 404)
        self.assertEqual(client.post(f"/api/history/{record.id}/explain/", {}).status_code, 404)
        self.assertEqual(
            client.get(f"/api/history/{record.id}/explanation-image/efficientnet_gradcam.png/").status_code,
            404,
        )
        self.assertEqual(client.get(f"/api/report/{record.id}/").status_code, 404)
        admin = get_user_model().objects.create_superuser(
            username="review-admin", email="admin@example.test", password="test-password-123"
        )
        client.force_login(admin)
        self.assertEqual(client.get(f"/api/history/{record.id}/").status_code, 200)

    def test_legacy_ownerless_records_are_hidden_from_users(self):
        from django.test import Client
        from prediction.models import Prediction
        legacy = Prediction.objects.create(
            owner=None, prediction_result="benign", confidence=0.8,
            fused_probability=0.2, prediction_probability=0.2,
            model_breakdown={}, pso_weights={},
        )
        client = Client()
        client.force_login(self.user)
        self.assertEqual(client.get(f"/api/history/{legacy.id}/").status_code, 404)
        admin = get_user_model().objects.create_superuser(
            username="legacy-admin", email="legacy@example.test", password="test-password-123"
        )
        client.force_login(admin)
        self.assertEqual(client.get(f"/api/history/{legacy.id}/").status_code, 200)

    def test_normal_serializer_never_exposes_cached_explanation_or_heatmap(self):
        from prediction.models import Prediction
        from prediction.serializers import PredictionSerializer
        record = Prediction.objects.create(
            owner=self.user, prediction_result="benign", confidence=0.8,
            fused_probability=0.2, prediction_probability=0.2,
            model_breakdown={}, pso_weights={}, heatmap_base64="secret",
            explanation_data={"private": "cached"},
        )
        data = PredictionSerializer(record).data
        self.assertEqual(data["explanation_data"], {})
        self.assertEqual(data["heatmap_base64"], "")

    def test_user_cannot_fetch_another_users_uploaded_image(self):
        from django.test import Client
        from prediction.models import UploadedImage
        from django.core.files.uploadedfile import SimpleUploadedFile
        other = get_user_model().objects.create_user(username="image-owner", password="test-password-123")
        image = UploadedImage.objects.create(
            owner=other, filename="private.png", image_type="raw",
            image=SimpleUploadedFile("private.png", b"not-real-image", content_type="image/png"),
        )
        client = Client()
        client.force_login(self.user)
        self.assertEqual(client.get(f"/api/uploads/{image.id}/image/").status_code, 404)
        client.force_login(get_user_model().objects.create_superuser(
            username="image-admin", email="admin2@example.test", password="test-password-123"
        ))
        self.assertEqual(client.get(f"/api/uploads/{image.id}/image/").status_code, 200)

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
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="xai-test-user", password="test-password-123"
        )

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
        self.assertEqual(agreement["individual_predictions"]["EfficientNet"], "malignant")
        self.assertEqual(agreement["supports_final_prediction"], 3)

    def test_malignant_class_column_uses_estimator_labels(self):
        from prediction.services.class_labels import malignant_column
        class Estimator:
            classes_ = ["malignant", "benign"]
        self.assertEqual(malignant_column(Estimator()), 0)

    def test_shap_values_accept_supported_binary_shapes_and_reject_mismatch(self):
        from types import SimpleNamespace
        from prediction.services.explainability import _extract_tree_shap
        explanation = SimpleNamespace(
            values=[[[0.1, 0.7], [0.2, 0.8]]],
            base_values=[[0.4, 0.6]],
        )
        values, base = _extract_tree_shap(explanation, 2, 1)
        self.assertEqual(values.tolist(), [0.7, 0.8])
        self.assertAlmostEqual(base, 0.6)
        with self.assertRaises(ValueError):
            _extract_tree_shap(explanation, 3, 1)

    def test_explanation_endpoint_returns_cached_payload(self):
        from django.test import Client
        from prediction.models import Prediction

        prediction = Prediction.objects.create(
            owner=self.user,
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
                "version": "xai-v2",
                "target_class": "benign",
            },
        )

        client = Client()
        client.force_login(self.user)
        response = client.post(f"/api/history/{prediction.id}/explain/", {})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["cached"])
        self.assertEqual(data["explanation_data"]["version"], "xai-v2")

    def test_explanation_is_generated_only_by_explicit_endpoint(self):
        from django.test import Client
        from unittest.mock import patch
        from prediction.models import Prediction
        record = Prediction.objects.create(
            owner=self.user, prediction_result="malignant", confidence=0.7,
            fused_probability=0.65, prediction_probability=0.7,
            model_breakdown={}, pso_weights={},
        )
        client = Client()
        client.force_login(self.user)
        normal = client.get(f"/api/history/{record.id}/")
        self.assertEqual(normal.json()["explanation_data"], {})
        with patch("prediction.views.build_explanation_for_prediction", return_value={"version": "new"}) as build:
            response = client.post(f"/api/history/{record.id}/explain/", {})
        self.assertEqual(response.status_code, 200)
        build.assert_called_once()
        self.assertFalse(response.json()["cached"])
        record.refresh_from_db()
        self.assertEqual(record.explanation_data, {"version": "new"})

    def test_stale_cached_explanation_is_rebuilt_only_after_explicit_action(self):
        from django.test import Client
        from unittest.mock import patch
        from prediction.models import Prediction
        record = Prediction.objects.create(
            owner=self.user, prediction_result="malignant", confidence=0.7,
            fused_probability=0.65, prediction_probability=0.7,
            model_breakdown={}, pso_weights={},
            explanation_data={"version": "xai-v1"},
        )
        client = Client()
        client.force_login(self.user)
        self.assertEqual(client.get(f"/api/history/{record.id}/").json()["explanation_data"], {})
        with patch("prediction.views.build_explanation_for_prediction", return_value={"version": "xai-v2"}) as build:
            response = client.post(f"/api/history/{record.id}/explain/", {})
        build.assert_called_once()
        self.assertFalse(response.json()["cached"])
        self.assertEqual(response.json()["explanation_data"]["version"], "xai-v2")
