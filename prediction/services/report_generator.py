"""
Generate research-oriented PDF reports for saved BreastVisionAI predictions.

The report includes the predictive result and, when full XAI has been generated,
the class-targeted Grad-CAM evidence, ensemble consensus, model agreement, and
TreeSHAP contributions for the Gradient Boosting meta-learner.
"""

import base64
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings
from matplotlib.backends.backend_pdf import PdfPages


def _media_file_from_url(url):
    """Resolve a locally stored MEDIA_URL path to MEDIA_ROOT."""
    if not url:
        return None

    media_prefix = settings.MEDIA_URL.rstrip("/") + "/"
    relative = url
    if relative.startswith(media_prefix):
        relative = relative[len(media_prefix):]
    elif relative.startswith("/"):
        relative = relative.lstrip("/")

    path = os.path.join(settings.MEDIA_ROOT, relative)
    return path if os.path.exists(path) else None


def _add_cached_xai_pages(pdf, prediction):
    explanation = prediction.explanation_data or {}
    if not explanation:
        return

    visual = explanation.get("visual_evidence", {})
    model_visuals = visual.get("models", {})
    consensus = visual.get("consensus", {})

    images = [
        (name, item.get("image_url"))
        for name, item in model_visuals.items()
    ]
    if consensus.get("image_url"):
        images.append(("Ensemble Consensus", consensus["image_url"]))

    if images:
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            "Explainable AI — Visual Evidence",
            fontsize=17,
            fontweight="bold",
            y=0.96,
        )

        positions = [
            [0.08, 0.54, 0.38, 0.32],
            [0.54, 0.54, 0.38, 0.32],
            [0.08, 0.13, 0.38, 0.32],
            [0.54, 0.13, 0.38, 0.32],
        ]

        for (title, image_url), position in zip(images[:4], positions):
            ax = fig.add_axes(position)
            image_path = _media_file_from_url(image_url)
            if image_path:
                ax.imshow(plt.imread(image_path))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Image unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

        fig.text(
            0.08,
            0.06,
            visual.get(
                "note",
                "Grad-CAM highlights model-influential regions and is not segmentation.",
            ),
            fontsize=9,
            wrap=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

    shap_data = explanation.get("meta_learner_explanation", {})
    agreement = explanation.get("model_agreement", {})
    features = shap_data.get("features", [])

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(
        "Explainable AI — Final Decision",
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )

    ax = fig.add_axes([0.12, 0.52, 0.78, 0.32])
    if features:
        display_features = list(reversed(features))
        labels = [item["feature"] for item in display_features]
        values = [item["shap_value"] for item in display_features]
        colors = ["#c62828" if value >= 0 else "#2e7d32" for value in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"SHAP contribution ({shap_data.get('output_space', 'unknown')})")
        ax.set_title("Gradient Boosting TreeSHAP Contributions")
    else:
        ax.text(0.5, 0.5, "SHAP explanation unavailable", ha="center", va="center")
        ax.axis("off")

    agreement_lines = [
        "Base-model agreement:",
        f"  Support for final class: {agreement.get('support_count', 0)}/{agreement.get('total_models', 0)}",
        f"  Unanimous: {agreement.get('unanimous', False)}",
        f"  Probability spread: {agreement.get('probability_spread', 0.0):.4f}",
        "",
        "Per-model votes:",
    ]
    for vote in agreement.get("votes", []):
        agreement_lines.append(
            f"  {vote['model']:<14s} {vote['prediction']:<10s} "
            f"P(malignant)={vote['malignancy_probability']:.4f}"
        )

    agreement_lines.extend(
        [
            "",
            f"TreeSHAP output space: {shap_data.get('output_space', 'unknown')}",
            f"TreeSHAP base value: {shap_data.get('base_value', 0.0):.6f}",
            (
                "Final malignancy probability: "
                f"{shap_data.get('final_malignancy_probability', prediction.prediction_probability):.4f}"
            ),
        ]
    )

    fig.text(
        0.1,
        0.42,
        "\n".join(agreement_lines),
        fontsize=10,
        family="monospace",
        verticalalignment="top",
    )

    limitations = explanation.get("limitations", [])
    if limitations:
        fig.text(
            0.1,
            0.13,
            "Interpretation notes:\n"
            + "\n".join(f"• {item}" for item in limitations),
            fontsize=9,
            verticalalignment="top",
            wrap=True,
        )

    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(prediction, request=None):
    """Generate a PDF report for a saved Prediction object."""
    try:
        reports_dir = os.path.join(settings.MEDIA_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        filename = f"report_{prediction.id}.pdf"
        filepath = os.path.join(reports_dir, filename)

        with PdfPages(filepath) as pdf:
            # --- Page 1: diagnosis and model-level evidence ---
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(
                "BreastVisionAI — Research Diagnostic Report",
                fontsize=18,
                fontweight="bold",
                y=0.95,
            )

            result_text = (
                f"Prediction: {prediction.prediction_result.upper()}\n"
                f"Confidence: {prediction.confidence:.2%}\n"
                f"Final malignancy probability: {prediction.prediction_probability:.4f}\n"
                f"Fused malignancy probability: {prediction.fused_probability:.4f}\n"
                f"Ensemble method: {prediction.ensemble_method}\n"
                f"Date: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            fig.text(
                0.1,
                0.86,
                result_text,
                fontsize=11,
                family="monospace",
                verticalalignment="top",
            )

            breakdown_text = "PSO-selected model breakdown:\n"
            for model_name, info in prediction.model_breakdown.items():
                breakdown_text += (
                    f"  {model_name:<14s} "
                    f"P(malignant)={info['probability']:.4f}  "
                    f"fusion_weight={info.get('weight', 0):.4f}\n"
                )

            fig.text(
                0.1,
                0.65,
                breakdown_text,
                fontsize=10,
                family="monospace",
                verticalalignment="top",
            )

            if (
                prediction.heatmap_base64
                and prediction.heatmap_base64.startswith("data:image")
            ):
                img_data = prediction.heatmap_base64.split(",", 1)[1]
                img = plt.imread(BytesIO(base64.b64decode(img_data)))
                ax = fig.add_axes([0.15, 0.12, 0.7, 0.38])
                ax.imshow(img)
                ax.set_title(
                    "Immediate Grad-CAM Preview (primary selected CNN)",
                    fontsize=11,
                )
                ax.axis("off")

            fig.text(
                0.1,
                0.05,
                "Research-use system: outputs and attribution maps are not substitutes "
                "for clinical diagnosis or lesion segmentation.",
                fontsize=8.5,
                wrap=True,
            )

            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 2: model probabilities and fusion weights ---
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle(
                "Ensemble Composition",
                fontsize=17,
                fontweight="bold",
                y=0.95,
            )

            model_names = list(prediction.model_breakdown.keys())
            probs = [
                prediction.model_breakdown[name]["probability"]
                for name in model_names
            ]
            weights = [
                prediction.fusion_weights.get(
                    name,
                    prediction.model_breakdown[name].get("weight", 0),
                )
                for name in model_names
            ]

            ax1 = fig.add_axes([0.12, 0.56, 0.76, 0.28])
            bars = ax1.bar(model_names, probs)
            ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Malignancy probability")
            ax1.set_title("Per-model probabilities")
            for bar, probability in zip(bars, probs):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(probability + 0.025, 0.97),
                    f"{probability:.3f}",
                    ha="center",
                    fontsize=9,
                )

            ax2 = fig.add_axes([0.12, 0.18, 0.76, 0.24])
            weight_bars = ax2.bar(model_names, weights)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Fusion weight")
            ax2.set_title("Equal-weight decision-level fusion")
            for bar, weight in zip(weight_bars, weights):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(weight + 0.025, 0.97),
                    f"{weight:.3f}",
                    ha="center",
                    fontsize=9,
                )

            fig.text(
                0.1,
                0.08,
                "PSO selected EfficientNet, ResNet, and VGG16. The deployed late-fusion "
                "stage uses equal weights; these are not PSO-optimized fusion weights.",
                fontsize=9,
                wrap=True,
            )

            pdf.savefig(fig)
            plt.close(fig)

            _add_cached_xai_pages(pdf, prediction)

        return filepath

    except Exception as exc:
        print(f"Error generating PDF report: {exc}")
        return None
