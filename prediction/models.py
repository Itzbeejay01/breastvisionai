"""
Django models for storing uploaded images and prediction results.
"""

from django.db import models


class UploadedImage(models.Model):
    IMAGE_TYPE_CHOICES = [
        ("raw", "Raw (needs preprocessing)"),
        ("processed", "Pre-processed"),
    ]

    image = models.ImageField(upload_to="uploads/")
    filename = models.CharField(max_length=255)
    image_type = models.CharField(
        max_length=20,
        choices=IMAGE_TYPE_CHOICES,
        default="raw",
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.filename


class Prediction(models.Model):
    CLASS_BENIGN = "benign"
    CLASS_MALIGNANT = "malignant"
    PREDICTION_CHOICES = [
        (CLASS_BENIGN, "Benign"),
        (CLASS_MALIGNANT, "Malignant"),
    ]

    uploaded_image = models.ForeignKey(
        UploadedImage,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )
    prediction_result = models.CharField(max_length=20, choices=PREDICTION_CHOICES)
    confidence = models.FloatField()
    fused_probability = models.FloatField()
    prediction_probability = models.FloatField()
    model_breakdown = models.JSONField()
    pso_weights = models.JSONField()
    fusion_weights = models.JSONField(blank=True, default=dict)
    ensemble_method = models.CharField(max_length=100, blank=True)
    heatmap_base64 = models.TextField(blank=True)
    image_type = models.CharField(max_length=20, default="raw")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.prediction_result} ({self.confidence:.2%}) @ {self.timestamp}"
