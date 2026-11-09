"""
DRF serializers for prediction app.
"""

from rest_framework import serializers
from prediction.models import UploadedImage, Prediction


def serialize_explanation(data, request=None):
    """Convert cached relative media paths for the explicit explain response."""
    import copy
    result = copy.deepcopy(data or {})
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


class UploadedImageSerializer(serializers.ModelSerializer):
    preview_url = serializers.SerializerMethodField()

    class Meta:
        model = UploadedImage
        fields = ["id", "filename", "image_type", "uploaded_at", "preview_url"]

    def get_preview_url(self, obj):
        request = self.context.get("request")
        if obj.image:
            url = f"/api/uploads/{obj.pk}/image/"
            if request:
                return request.build_absolute_uri(url)
            return url
        return None


class PredictionSerializer(serializers.ModelSerializer):
    uploaded_image = UploadedImageSerializer(read_only=True)
    # Detailed evidence is returned only by POST /history/{id}/explain/.
    explanation_data = serializers.SerializerMethodField()
    heatmap_base64 = serializers.SerializerMethodField()

    def get_explanation_data(self, obj):
        return {}

    def get_heatmap_base64(self, obj):
        return ""

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
            "meta_feature_schema",
            "heatmap_base64",
            "explanation_data",
            "image_type",
            "timestamp",
        ]
