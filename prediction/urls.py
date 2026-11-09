"""
URL routes for the prediction app.

All endpoints are under /api/ (included from breastvisionai/urls.py).
"""

from django.urls import path
from prediction.views import (
    PredictView,
    BatchPredictView,
    ModelListView,
    EnsembleConfigView,
    HistoryListView,
    HistoryDetailView,
    ExplainPredictionView,
    UploadView,
    ReportView,
    UploadedImageView,
    ExplanationImageView,
)

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("predict/batch/", BatchPredictView.as_view(), name="predict-batch"),
    path("models/", ModelListView.as_view(), name="models"),
    path("ensemble/", EnsembleConfigView.as_view(), name="ensemble"),
    path("history/", HistoryListView.as_view(), name="history"),
    path("history/<int:id>/", HistoryDetailView.as_view(), name="history-detail"),
    path("history/<int:id>/explain/", ExplainPredictionView.as_view(), name="history-explain"),
    path("upload/", UploadView.as_view(), name="upload"),
    path("uploads/<int:id>/image/", UploadedImageView.as_view(), name="uploaded-image"),
    path("history/<int:id>/explanation-image/<str:filename>/", ExplanationImageView.as_view(), name="explanation-image"),
    path("report/<int:id>/", ReportView.as_view(), name="report"),
]
