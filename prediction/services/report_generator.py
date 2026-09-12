"""
Generate PDF reports for prediction results using matplotlib.

Creates a clinical-style report with:
  - Diagnosis result and confidence
  - Model breakdown table
  - PSO weights
  - Grad-CAM heatmap (if available)
"""

import os
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_pdf_report(prediction, request=None):
    """
    Generate a PDF report for a Prediction object.

    Args:
        prediction: Prediction model instance
        request: Django request object (for building absolute URLs)

    Returns:
        Path to the generated PDF file, or None on failure.
    """
    try:
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "media", "reports"
        )
        os.makedirs(reports_dir, exist_ok=True)

        filename = f"report_{prediction.id}.pdf"
        filepath = os.path.join(reports_dir, filename)

        with PdfPages(filepath) as pdf:
            # --- Page 1: Diagnosis Report Header ---
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle("BreastVisionAI — Diagnostic Report",
                         fontsize=18, fontweight="bold", y=0.95)

            # Diagnosis result
            result_text = (
                f"Diagnosis: {prediction.prediction_result.upper()}\n"
                f"Confidence: {prediction.confidence:.2%}\n"
                f"Ensemble Method: {prediction.ensemble_method}\n"
                f"Prediction Probability: {prediction.prediction_probability:.4f}\n"
                f"Date: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            color = "red" if prediction.prediction_result == "malignant" else "green"

            fig.text(0.1, 0.85, result_text, fontsize=12, family="monospace",
                     verticalalignment="top")
            fig.text(0.1, 0.78, "●", fontsize=36, color=color, fontweight="bold")

            # Model breakdown
            breakdown_text = "Model Breakdown:\n"
            for model_name, info in prediction.model_breakdown.items():
                breakdown_text += (
                    f"  {model_name:<20s}  "
                    f"Probability: {info['probability']:.4f}  "
                    f"Weight: {info.get('weight', 0):.4f}\n"
                )

            fig.text(0.1, 0.65, breakdown_text, fontsize=10, family="monospace",
                     verticalalignment="top")

            # PSO weights
            pso_text = "PSO Weights:\n"
            for model_name, weight in prediction.pso_weights.items():
                pso_text += f"  {model_name:<20s}  {weight:.4f}\n"

            fig.text(0.1, 0.45, pso_text, fontsize=10, family="monospace",
                     verticalalignment="top")

            # Heatmap
            if prediction.heatmap_base64 and prediction.heatmap_base64.startswith("data:image"):
                import base64
                img_data = prediction.heatmap_base64.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = plt.imread(BytesIO(img_bytes))

                ax = fig.add_axes([0.1, 0.05, 0.8, 0.3])
                ax.imshow(img)
                ax.set_title("Grad-CAM Heatmap", fontsize=12)
                ax.axis("off")

            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 2: Model Breakdown Charts ---
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 11))

            # Bar chart of model probabilities
            model_names = list(prediction.model_breakdown.keys())
            probs = [prediction.model_breakdown[m]["probability"] for m in model_names]
            weights = [prediction.model_breakdown[m].get("weight", 0) for m in model_names]

            ax1 = axes[0]
            bars = ax1.bar(model_names, probs, color=["#2196F3", "#4CAF50", "#FF9800"])
            ax1.set_ylabel("Malignancy Probability")
            ax1.set_title("Per-Model Prediction Probabilities")
            ax1.set_ylim(0, 1)
            for bar, prob in zip(bars, probs):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{prob:.4f}", ha="center", fontsize=10)

            # PSO weights pie chart
            ax2 = axes[1]
            selected_models = list(prediction.pso_weights.keys())
            pso_vals = list(prediction.pso_weights.values())
            ax2.pie(pso_vals, labels=selected_models, autopct="%1.1f%%", startangle=90)
            ax2.set_title("PSO-Optimized Model Weights")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        return filepath

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return None
