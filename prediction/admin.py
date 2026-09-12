from django.contrib import admin
from prediction.models import UploadedImage, Prediction


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ["filename", "image_type", "uploaded_at"]
    list_filter = ["image_type", "uploaded_at"]


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = [
        "prediction_result",
        "confidence",
        "ensemble_method",
        "image_type",
        "timestamp",
    ]
    list_filter = ["prediction_result", "image_type", "ensemble_method", "timestamp"]
    readonly_fields = ["model_breakdown", "pso_weights", "heatmap_base64"]
