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
    UploadView,
    ReportView,
)

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("predict/batch/", BatchPredictView.as_view(), name="predict-batch"),
    path("models/", ModelListView.as_view(), name="models"),
    path("ensemble/", EnsembleConfigView.as_view(), name="ensemble"),
    path("history/", HistoryListView.as_view(), name="history"),
    path("history/<int:id>/", HistoryDetailView.as_view(), name="history-detail"),
    path("upload/", UploadView.as_view(), name="upload"),
    path("report/<int:id>/", ReportView.as_view(), name="report"),
]
