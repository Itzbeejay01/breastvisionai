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
            "image_type",
            "timestamp",
        ]
