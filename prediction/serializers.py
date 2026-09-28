"""
DRF serializers for prediction app.
"""

from rest_framework import serializers
from prediction.models import UploadedImage, Prediction


class UploadedImageSerializer(serializers.ModelSerializer):
    preview_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedImage
        fields = ["id", "filename", "image_type", "uploaded_at", "preview_url"]

    def get_preview_url(self, obj):
        request = self.context.get("request")
        if obj.image:
            url = obj.image.url
            if request:
                return request.build_absolute_uri(url)
            return url
        return None


class PredictionSerializer(serializers.ModelSerializer):
    uploaded_image = UploadedImageSerializer(read_only=True)
    explanation_data = serializers.SerializerMethodField()

    def get_explanation_data(self, obj):
        data = obj.explanation_data or {}
        if not data:
            return {}

        # Keep relative media paths in the database, but expose absolute URLs
        # when a request context is available (important for split frontend/API
        # deployments).
        import copy
        result = copy.deepcopy(data)
        request = self.context.get("request")
        if not request:
            return result

        visual = result.get("visual_evidence", {})
        for item in visual.get("models", {}).values():
            url = item.get("image_url")
            if url and url.startswith("/"):
                item["image_url"] = request.build_absolute_uri(url)

        consensus = visual.get("consensus", {})
        url = consensus.get("image_url")
        if url and url.startswith("/"):
            consensus["image_url"] = request.build_absolute_uri(url)

        return result

    class Meta:
        model = Prediction
        fields = [
            "id",
            "uploaded_image",
            "prediction_result",
            "confidence",
            "fused_probability",
            "prediction_probability",
            "model_breakdown",
            "pso_weights",
            "fusion_weights",
            "ensemble_method",
            "heatmap_base64",
            "explanation_data",
            "image_type",
            "timestamp",
        ]
