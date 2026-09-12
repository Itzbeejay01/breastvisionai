"""
API views for the prediction app.

Endpoints:
  POST /api/predict/           - Single image prediction (PSO ensemble + GB)
  POST /api/predict/batch/     - Batch image prediction
  GET  /api/models/            - List PSO-selected models with metrics
  GET  /api/ensemble/          - Ensemble configuration + PSO weights
  GET  /api/history/           - Prediction history (paginated)
  GET  /api/history/{id}/      - Single prediction detail
  POST /api/upload/            - Upload image to temp storage
  GET  /api/report/{id}/       - Download PDF report for a prediction
"""

import os
import json

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status, pagination
from rest_framework.permissions import IsAuthenticated
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from django.http import FileResponse, Http404

from prediction.models import UploadedImage, Prediction
from prediction.serializers import (
    PredictionSerializer,
    UploadedImageSerializer,
)
from prediction.services.prediction import predict_single, predict_batch
from prediction.services.model_registry import PSOModelRegistry, MODEL_NAMES_ALL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StandardResultsSetPagination(pagination.PageNumberPagination):
    page_size = 20
    page_size_query_param = "limit"
    max_page_size = 100


class PredictView(APIView):
    """
    POST /api/predict/

    Run PSO ensemble prediction on a single image.

    Request: multipart/form-data
      - image:          the image file (PNG, JPEG, DICOM)
      - image_type:     "raw" (needs preprocessing) or "processed" (already normalized)
      - model:          (optional) specific model to use (default: all PSO-selected)

    Response:
      {
        prediction, confidence, fused_probability,
        prediction_probability, model_breakdown, pso_weights,
        ensemble_method, heatmap_base64, image_type,
        selected_models
      }
    """

    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        image = request.FILES.get("image")
        image_type = request.data.get("image_type", "raw")

        if not image:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if image_type not in ("raw", "processed"):
            return Response(
                {"error": "image_type must be 'raw' or 'processed'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save uploaded image
        uploaded = UploadedImage.objects.create(
            image=image,
            filename=image.name,
            image_type=image_type,
        )

        # Run PSO prediction
        result = predict_single(uploaded.image.path, image_type=image_type)

        # Save prediction to database
        prediction = Prediction.objects.create(
            uploaded_image=uploaded,
            prediction_result=result["prediction"],
            confidence=result["confidence"],
            fused_probability=result["fused_probability"],
            prediction_probability=result["prediction_probability"],
            model_breakdown=result["model_breakdown"],
            pso_weights=result["pso_weights"],
            fusion_weights=result["fusion_weights"],
            ensemble_method=result["ensemble_method"],
            heatmap_base64=result["heatmap_base64"],
            image_type=result["image_type"],
        )

        response_data = PredictionSerializer(prediction, context={"request": request}).data
        response_data["prediction"] = result["prediction"]
        response_data["selected_models"] = result["selected_models"]

        return Response(response_data, status=status.HTTP_200_OK)


class BatchPredictView(APIView):
    """
    POST /api/predict/batch/

    Run PSO ensemble prediction on multiple images.

    Request: multipart/form-data
      - images:      multiple image files
      - image_type:  "raw" or "processed"
    """

    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        images = request.data.getlist("images")
        image_type = request.data.get("image_type", "raw")

        if not images:
            return Response(
                {"error": "No images provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save all uploaded images
        saved_images = []
        for img in images:
            uploaded = UploadedImage.objects.create(
                image=img,
                filename=img.name,
                image_type=image_type,
            )
            saved_images.append(uploaded)

        # Run batch prediction
        image_paths = [u.image.path for u in saved_images]
        result = predict_batch(image_paths, image_type=image_type)

        # Save each prediction
        predictions = []
        for u, r in zip(saved_images, result["results"]):
            p = Prediction.objects.create(
                uploaded_image=u,
                prediction_result=r["prediction"],
                confidence=r["confidence"],
                fused_probability=r["fused_probability"],
                prediction_probability=r["prediction_probability"],
                model_breakdown=r["model_breakdown"],
                pso_weights=r["pso_weights"],
                fusion_weights=r["fusion_weights"],
                ensemble_method=r["ensemble_method"],
                heatmap_base64=r["heatmap_base64"],
                image_type=r["image_type"],
            )
            predictions.append(p)

        serializer = PredictionSerializer(predictions, many=True, context={"request": request})

        return Response(
            {
                "results": serializer.data,
                "total": result["total"],
                "summary": result["summary"],
            },
            status=status.HTTP_200_OK,
        )


class ModelListView(APIView):
    """
    GET /api/models/

    List all 5 models with individual metrics.
    PSO-selected models are marked with their weights.
    """

    def get(self, request):
        registry = PSOModelRegistry.initialize()
        selection = registry.get_selection()
        pso_weights = registry.get_pso_weights()
        fusion_weights = registry.get_fusion_weights()

        # Load individual model metrics from evaluate result files
        result_dirs = {
            "EfficientNet": "EfficientNet_Result",
            "DenseNet": "DenseNet_Result",
            "ResNet": "ResNet_Result",
            "VGG16": "VGG16_Result",
            "Xception": "Xception_Result",
        }

        models = []
        cache_dir = os.path.join(BASE_DIR, "pso_cache")
        labels_path = os.path.join(cache_dir, "test_labels.npy")
        cached_labels = None
        if os.path.exists(labels_path):
            import numpy as np
            cached_labels = np.load(labels_path)

        for i, name in enumerate(MODEL_NAMES_ALL):
            metrics = {}
            result_dir = os.path.join(BASE_DIR, result_dirs.get(name, ""))
            metrics_file = os.path.join(result_dir, "metrics.txt")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    content = f.read()
                for line in content.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_").replace("-", "_")
                        key = {"f1_score": "f1", "auc_roc": "auc"}.get(key, key)
                        val = val.strip()
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            pass

            # The cached test probabilities are the source of truth when an
            # individual model metrics.txt file is not present.
            cache_path = os.path.join(cache_dir, f"{name}_test.npy")
            if cached_labels is not None and os.path.exists(cache_path):
                import numpy as np
                probabilities = np.asarray(np.load(cache_path)).reshape(-1)
                if len(probabilities) == len(cached_labels):
                    predictions = (probabilities >= 0.5).astype(int)
                    metrics.update({
                        "accuracy": float(accuracy_score(cached_labels, predictions)),
                        "precision": float(precision_score(cached_labels, predictions, zero_division=0)),
                        "recall": float(recall_score(cached_labels, predictions, zero_division=0)),
                        "f1": float(f1_score(cached_labels, predictions, zero_division=0)),
                        "auc": float(roc_auc_score(cached_labels, probabilities)),
                    })

            is_selected = name in selection["selected_models"]
            model_data = {
                "name": name,
                "architecture": name,
                "index": i,
                "selected_by_pso": is_selected,
                "pso_weight": pso_weights.get(name, 0.0),
                "fusion_weight": fusion_weights.get(name, 0.0),
                "metrics": metrics,
            }
            models.append(model_data)

        return Response(models)


class EnsembleConfigView(APIView):
    """
    GET /api/ensemble/

    Return the PSO selection + stacking ensemble configuration.
    """

    def get(self, request):
        registry = PSOModelRegistry.initialize()
        selection = registry.get_selection()
        fusion_config = registry.get_fusion_config()
        fusion_weights = registry.get_fusion_weights()
        pso_weights = registry.get_pso_weights()

        # Load stacking metrics
        stacking_metrics_path = os.path.join(
            BASE_DIR, "Stacking_Result", "stacking_metrics.txt"
        )
        stacking_metrics = {}
        if os.path.exists(stacking_metrics_path):
            with open(stacking_metrics_path, "r") as f:
                content = f.read()
            for line in content.split("\n"):
                if ":" in line and any(
                    kw in line.lower()
                    for kw in ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
                ):
                    key, val = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_").replace("-", "_")
                    key = {"f1_score": "f1", "auc_roc": "auc"}.get(key, key)
                    val = val.strip()
                    try:
                        stacking_metrics[key] = float(val)
                    except ValueError:
                        pass

        response = {
            "method": "PSO-Weighted Late Fusion + GradientBoosting",
            "fusion_strategy": fusion_config.get("fusion_strategy", "decision_level_late_fusion"),
            "fusion_method": fusion_config.get("fusion_method", "equal_weight_probability_average"),
            "selected_models": selection["selected_models"],
            "selected_indices": selection["selected_indices"],
            "dropped_models": selection["dropped_models"],
            "fusion_weights": fusion_weights,
            "pso_weights": pso_weights,
            "pso_parameters": selection.get("pso_parameters", {}),
            "validation_metrics": selection.get("validation_metrics", {}),
            "stacking_metrics": stacking_metrics,
            "meta_learner": {
                "type": "GradientBoostingClassifier",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
            },
        }

        return Response(response)


class HistoryListView(APIView):
    """
    GET /api/history/?page=1&limit=20

    Paginated list of prediction history.
    """

    pagination_class = StandardResultsSetPagination

    def get(self, request):
        predictions = Prediction.objects.all()
        paginator = StandardResultsSetPagination()
        page = paginator.paginate_queryset(predictions, request)
        serializer = PredictionSerializer(page, many=True, context={"request": request})
        return paginator.get_paginated_response(serializer.data)


class HistoryDetailView(APIView):
    """
    GET /api/history/{id}/

    Single prediction detail with full model breakdown.
    """

    def get(self, request, id):
        try:
            prediction = Prediction.objects.get(id=id)
        except Prediction.DoesNotExist:
            raise Http404

        serializer = PredictionSerializer(prediction, context={"request": request})
        return Response(serializer.data)


class UploadView(APIView):
    """
    POST /api/upload/

    Upload an image to temp storage and get a file_id for later prediction.

    Request: multipart/form-data
      - image:      the image file
      - image_type: "raw" or "processed"

    Response: {file_id, filename, preview_url, image_type}
    """

    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        image = request.FILES.get("image")
        image_type = request.data.get("image_type", "raw")

        if not image:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if image_type not in ("raw", "processed"):
            return Response(
                {"error": "image_type must be 'raw' or 'processed'"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        uploaded = UploadedImage.objects.create(
            image=image,
            filename=image.name,
            image_type=image_type,
        )

        serializer = UploadedImageSerializer(uploaded, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ReportView(APIView):
    """
    GET /api/report/{id}/

    Generate and download a PDF report for a prediction.

    Uses the heatmap and prediction data to create a clinical report.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, id):
        from prediction.services.report_generator import generate_pdf_report

        try:
            prediction = Prediction.objects.get(id=id)
        except Prediction.DoesNotExist:
            raise Http404

        pdf_path = generate_pdf_report(prediction, request)

        if pdf_path and os.path.exists(pdf_path):
            return FileResponse(
                open(pdf_path, "rb"),
                as_attachment=True,
                filename=f"breastvisionai_report_{prediction.id}.pdf",
            )

        return Response(
            {"error": "Could not generate report"},
            status=status.HTTP_500_SERVER_ERROR,
        )
