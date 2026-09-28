export default function ExplanationPanel({
  result,
  onGenerate,
  loading = false,
  error = null,
}) {
  const explanation = result?.explanation_data;
  const hasExplanation =
    explanation && Object.keys(explanation).length > 0;

  if (!hasExplanation) {
    return (
      <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-5">
          <div className="max-w-2xl">
            <p className="font-label-caps text-label-caps text-primary mb-2">
              EXPLAINABLE AI
            </p>
            <h2 className="font-headline-md text-headline-md text-on-surface">
              Why did the ensemble make this decision?
            </h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-3 leading-relaxed">
              Generate class-targeted Grad-CAM evidence for all three
              PSO-selected CNNs, an ensemble consensus map, model-agreement
              metrics, and a TreeSHAP explanation of the Gradient Boosting
              meta-learner.
            </p>
            {error && (
              <p className="text-error font-body-sm text-body-sm mt-3">
                {error}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onGenerate}
            disabled={loading || !result?.id}
            className="bg-primary text-on-primary px-6 py-3 rounded-full font-body-md text-body-md shadow-sm hover:bg-primary-container hover:text-on-primary-container transition-colors disabled:opacity-50 flex items-center justify-center gap-2 min-w-[220px]"
          >
            <span className="material-symbols-outlined">
              {loading ? "progress_activity" : "psychology"}
            </span>
            {loading ? "Generating explanation…" : "Generate Full Explanation"}
          </button>
        </div>
      </div>
    );
  }

  const visual = explanation.visual_evidence || {};
  const agreement = explanation.model_agreement || {};
  const shap = explanation.meta_learner_explanation || {};
  const modelVisuals = Object.entries(visual.models || {});
  const features = shap.features || [];
  const maxAbsShap = Math.max(
    ...features.map((item) => Math.abs(item.shap_value || 0)),
    0.000001
  );

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <p className="font-label-caps text-label-caps text-primary mb-2">
              EXPLAINABLE AI / XAI-V1
            </p>
            <h2 className="font-headline-md text-headline-md text-on-surface">
              Why this result?
            </h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-2">
              Target class:{" "}
              <strong className="text-on-surface">
                {(explanation.target_class || "unknown").toUpperCase()}
              </strong>
            </p>
          </div>
          <div className="bg-surface-container-low rounded-xl px-5 py-4 min-w-[220px]">
            <span className="font-label-caps text-label-caps text-on-surface-variant">
              Model agreement
            </span>
            <div className="font-headline-md text-headline-md text-on-surface mt-1">
              {agreement.support_count ?? 0}/{agreement.total_models ?? 0}
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
              {agreement.unanimous ? "Unanimous support" : "Mixed model support"}
            </p>
          </div>
        </div>
      </div>

      <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6">
        <div className="mb-5">
          <h3 className="font-headline-sm text-headline-sm text-on-surface">
            Visual evidence — class-targeted Grad-CAM
          </h3>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-2">
            Each map explains the same final target class. The consensus map is
            the fusion-weighted average of the three normalized attribution maps.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {modelVisuals.map(([name, item]) => (
            <figure
              key={name}
              className="rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low"
            >
              <img
                src={item.image_url}
                alt={`${name} Grad-CAM for ${item.target_class}`}
                className="w-full aspect-[4/3] object-cover"
              />
              <figcaption className="p-4">
                <strong className="font-body-md text-body-md text-on-surface">
                  {name}
                </strong>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
                  Last convolutional layer: {item.last_conv_layer}
                </p>
              </figcaption>
            </figure>
          ))}

          {visual.consensus?.image_url && (
            <figure className="rounded-xl overflow-hidden border-2 border-primary/30 bg-primary-container/5">
              <img
                src={visual.consensus.image_url}
                alt="Ensemble consensus Grad-CAM"
                className="w-full aspect-[4/3] object-cover"
              />
              <figcaption className="p-4">
                <strong className="font-body-md text-body-md text-primary">
                  Ensemble Consensus
                </strong>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
                  Fusion-weighted shared visual evidence across the selected CNNs.
                </p>
              </figcaption>
            </figure>
          )}
        </div>

        <div className="mt-5 bg-surface-container-low rounded-lg p-4 flex gap-3">
          <span className="material-symbols-outlined text-primary">info</span>
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            {visual.note ||
              "Grad-CAM shows influential image regions; it is not a lesion segmentation."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6">
          <h3 className="font-headline-sm text-headline-sm text-on-surface">
            Gradient Boosting decision factors — TreeSHAP
          </h3>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-2 mb-6">
            Positive values push the final decision toward malignancy; negative
            values push it toward benign. SHAP output space:{" "}
            <strong>{shap.output_space || "unknown"}</strong>.
          </p>

          <div className="space-y-4">
            {features.map((item) => {
              const magnitude =
                (Math.abs(item.shap_value || 0) / maxAbsShap) * 100;
              const towardMalignant = item.shap_value >= 0;
              return (
                <div key={item.feature}>
                  <div className="flex items-center justify-between gap-3 text-sm mb-1.5">
                    <span className="font-medium text-on-surface">
                      {item.feature}
                    </span>
                    <span className="font-data-num text-data-num text-on-surface-variant">
                      SHAP {item.shap_value >= 0 ? "+" : ""}
                      {Number(item.shap_value).toFixed(4)}
                    </span>
                  </div>
                  <div className="h-2.5 bg-surface-variant rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        towardMalignant ? "bg-error" : "bg-secondary"
                      }`}
                      style={{ width: `${Math.max(magnitude, 2)}%` }}
                    />
                  </div>
                  <div className="flex justify-between mt-1 text-xs text-on-surface-variant">
                    <span>
                      input probability {(Number(item.value) * 100).toFixed(1)}%
                    </span>
                    <span>
                      {towardMalignant
                        ? "toward malignant"
                        : "toward benign"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-6 pt-4 border-t border-outline-variant/30 flex justify-between gap-4 text-sm">
            <span className="text-on-surface-variant">Final malignancy probability</span>
            <strong className="text-on-surface">
              {((shap.final_malignancy_probability ?? 0) * 100).toFixed(1)}%
            </strong>
          </div>
        </div>

        <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6">
          <h3 className="font-headline-sm text-headline-sm text-on-surface">
            Base-model agreement
          </h3>
          <p className="font-body-sm text-body-sm text-on-surface-variant mt-2 mb-6">
            Probability spread:{" "}
            {((agreement.probability_spread ?? 0) * 100).toFixed(1)} percentage
            points.
          </p>

          <div className="space-y-3">
            {(agreement.votes || []).map((vote) => (
              <div
                key={vote.model}
                className="rounded-lg border border-outline-variant/30 p-4 flex items-center justify-between gap-4"
              >
                <div>
                  <strong className="text-on-surface">{vote.model}</strong>
                  <p className="text-sm text-on-surface-variant mt-1">
                    {(vote.malignancy_probability * 100).toFixed(1)}% malignancy
                    probability
                  </p>
                </div>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    vote.supports_final_prediction
                      ? "bg-primary/10 text-primary"
                      : "bg-surface-variant text-on-surface-variant"
                  }`}
                >
                  {vote.prediction.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {(explanation.limitations || []).length > 0 && (
        <div className="bg-surface-container-low rounded-xl border border-outline-variant/30 p-5">
          <h3 className="font-body-md text-body-md font-semibold text-on-surface">
            Interpretation notes
          </h3>
          <ul className="mt-3 space-y-2 list-disc pl-5 text-sm text-on-surface-variant">
            {explanation.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
