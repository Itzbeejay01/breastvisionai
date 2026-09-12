import { useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import ImageUploader from "../components/ImageUploader";
import ModelSelector from "../components/ModelSelector";
import ReportGenerator from "../components/ReportGenerator";
import { verdictLabel } from "../utils/verdict";
import useStore from "../store/index";

export default function DiagnosisPage() {
  const { id } = useParams();
  const location = useLocation();
  const { currentResult, predict, fetchHistoryDetail } = useStore();
  const [selected, setSelected] = useState(
    location.state?.selectedImage || null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [scanning, setScanning] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    (async () => {
      try {
        const detail = await fetchHistoryDetail(id);
        if (!cancelled) {
          setSelected(null);
          useStore.setState({ currentResult: detail });
        }
      } catch (e) {
        if (!cancelled) setError("Failed to load history result");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id, fetchHistoryDetail]);

  const handleSelect = (item) => {
    setSelected(item);
    useStore.setState({ currentResult: null });
  };

  const runPrediction = async () => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setScanning(true);
    try {
      await predict(selected.file, selected.imageType);
    } catch (e) {
      setError(e.message || "Prediction failed");
    } finally {
      setLoading(false);
      setScanning(false);
    }
  };

  const resetAnalysis = () => {
    setSelected(null);
    useStore.setState({ currentResult: null });
    setShowHeatmap(true);
  };

  const result = currentResult;
  const verdict = result?.prediction_result || result?.prediction;
  const isMalignant = verdict === "malignant";
  const confidence = result?.confidence ?? 0;
  const imageSrc = selected?.preview || result?.uploaded_image?.preview_url;
  const isSavedAnalysis = Boolean(id);

  return (
    <main className="diagnosis-page flex-grow pt-28 pb-12 px-5 md:px-10 max-w-container-max mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-7">
      <header className="diagnosis-header lg:col-span-12">
        <div>
          <p className="diagnosis-kicker">{isSavedAnalysis ? "HISTORY / CLINICAL RECORD" : "WORKSPACE / NEW SESSION"}</p>
          <h1 className="font-display text-4xl md:text-5xl font-semibold mt-2">{isSavedAnalysis ? "Saved analysis" : "New diagnosis"}</h1>
          <p className="text-on-surface-variant mt-3 text-lg">{isSavedAnalysis ? `Reviewing prediction record #${id} and its explainable evidence.` : "Upload an image to begin an explainable ensemble analysis."}</p>
        </div>
        <div className={`diagnosis-mode-badge ${isSavedAnalysis ? "saved" : "new"}`}><span className="material-symbols-outlined">{isSavedAnalysis ? "history" : "add_circle"}</span><div><strong>{isSavedAnalysis ? "SAVED RESULT" : "NEW ANALYSIS"}</strong><small>{isSavedAnalysis ? "Read-only history view" : "Ready for image upload"}</small></div></div>
      </header>
      {/* Left Panel: Image Selection & Viewer (7 columns) */}
      <section className="lg:col-span-7 flex flex-col gap-6">
        {/* Step 1: Choose an image (always available until a result is shown) */}
        {!result && !selected && (
          <div className="bg-surface-container-lowest rounded-xl shadow-card border border-surface-container-high p-6 flex flex-col gap-4 spring-hover">
            <div className="diagnosis-panel-label"><span className="material-symbols-outlined">cloud_upload</span><div><h2 className="font-headline-md text-headline-md text-on-surface">Choose a scan</h2><p>Select a file or choose one from your history.</p></div></div>
            <ImageUploader onSelect={handleSelect} />
          </div>
        )}

        {/* Selected image preview before analysis */}
        {!result && selected && (
          <div className="bg-surface-container-lowest rounded-xl shadow-card border border-surface-container-high overflow-hidden flex flex-col spring-hover">
            <div className="relative w-full aspect-[4/3] bg-surface-container flex items-center justify-center overflow-hidden">
              {selected.preview && (
                <img
                  src={selected.preview}
                  alt="Selected preview"
                  className="object-cover w-full h-full opacity-90"
                />
              )}
              {scanning && <div className="scan-line" />}
              <div className="absolute top-4 left-4 glass-panel px-3 py-1 rounded-full border border-outline-variant/30 flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    selected.imageType === "processed"
                      ? "bg-secondary"
                      : "bg-primary"
                  }`}
                />
                <span className="font-label-caps text-label-caps text-on-surface">
                  {(selected.imageType || "raw").toUpperCase()} U/S
                </span>
              </div>
            </div>
            <div className="p-4 border-t border-surface-variant flex justify-between items-center bg-surface-container-lowest">
              <div className="flex items-center gap-2 text-on-surface-variant text-body-sm min-w-0">
                <span className="material-symbols-outlined text-[18px]">
                  info
                </span>
                <span className="truncate">
                  {selected.file?.name || "Selected image"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={resetAnalysis}
                  className="text-primary hover:bg-primary/5 transition-colors px-3 py-2 rounded-lg font-body-sm text-body-sm touch-target"
                >
                  Change
                </button>
                <button
                  type="button"
                  onClick={runPrediction}
                  disabled={loading}
                  className="bg-primary text-on-primary px-6 py-2 rounded-full font-body-md text-body-md hover:bg-primary-container hover:text-on-primary-container transition-colors shadow-sm flex items-center gap-2 touch-target disabled:opacity-50"
                >
                  {/* <span className="material-symbols-outlined">
                    {loading ? "progress_activity animate-spin" : "play_arrow"}
                  </span> */}
                  {loading ? "Analyzing…" : "Start Analysis"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Result viewer after analysis */}
        {result && (
          <div className="bg-surface-container-lowest rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] border border-surface-variant overflow-hidden flex flex-col spring-hover">
            {/* Main Image Preview */}
            <div className="relative w-full aspect-[4/3] bg-surface-container flex items-center justify-center group overflow-hidden">
              {imageSrc ? (
                <img
                  src={imageSrc}
                  alt="Ultrasound Scan"
                  className="object-cover w-full h-full opacity-90 transition-opacity duration-500"
                />
              ) : (
                <div className="text-on-surface-variant font-body-sm">
                  No image available
                </div>
              )}
              {result?.heatmap_base64 && showHeatmap && (
                <img
                  src={result.heatmap_base64}
                  alt="Grad-CAM heatmap overlay"
                  className="absolute top-0 left-0 w-full h-full object-cover mix-blend-overlay opacity-70 transition-opacity duration-500"
                />
              )}
              {scanning && <div className="scan-line" />}

              {/* Badges & Overlays */}
              <div className="absolute top-4 left-4 glass-panel px-3 py-1 rounded-full border border-outline-variant/30 flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    result?.image_type === "processed"
                      ? "bg-secondary"
                      : "bg-primary"
                  }`}
                />
                <span className="font-label-caps text-label-caps text-on-surface">
                  {(result?.image_type || "raw").toUpperCase()} U/S
                </span>
              </div>

              {/* Heatmap Toggle Overlay */}
              {result?.heatmap_base64 && (
                <div className="absolute bottom-4 right-4 glass-panel p-2 rounded-lg flex items-center gap-3 shadow-lg">
                  <span className="font-label-caps text-label-caps text-on-surface">
                    GRAD-CAM Overlay
                  </span>
                  <button
                    type="button"
                    onClick={() => setShowHeatmap((v) => !v)}
                    className={`w-10 h-5 rounded-full relative transition-colors duration-200 ${
                      showHeatmap ? "bg-primary" : "bg-surface-variant"
                    }`}
                    aria-label="Toggle Grad-CAM overlay"
                    aria-pressed={showHeatmap}
                  >
                    <div
                      className={`w-4 h-4 bg-surface-container-lowest rounded-full absolute top-[2px] transition-transform duration-200 ${
                        showHeatmap ? "left-[22px]" : "left-[2px]"
                      }`}
                    />
                  </button>
                </div>
              )}
            </div>

            {/* Action Bar */}
            <div className="p-4 border-t border-surface-variant flex justify-between items-center bg-surface-container-lowest">
              <div className="flex items-center gap-2 text-on-surface-variant text-body-sm">
                <span className="material-symbols-outlined text-[18px]">
                  info
                </span>
                <span>
                  {result?.uploaded_image?.filename || "Scan"}
                  {" · "}
                  {new Date(result?.timestamp).toLocaleString()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={resetAnalysis}
                  className="text-primary hover:bg-primary/5 transition-colors px-3 py-2 rounded-lg font-body-sm text-body-sm touch-target"
                >
                  New Image
                </button>
                <button
                  type="button"
                  onClick={() => setScanning((v) => !v)}
                  className="bg-primary text-on-primary px-6 py-2 rounded-full font-body-md text-body-md hover:bg-primary-container hover:text-on-primary-container transition-colors shadow-sm flex items-center gap-2 touch-target"
                >
                  <span className="material-symbols-outlined">play_arrow</span>
                  {scanning ? "Stop Scan" : "Replay Scan"}
                </button>
              </div>
            </div>
          </div>
        )}

        {error && (
          <p className="text-error font-body-sm text-body-sm">⚠️ {error}</p>
        )}
      </section>

      {/* Right Panel: Verdict & Models (5 columns) */}
      <section className="lg:col-span-5 flex flex-col gap-6">
        {result ? (
          <>
            {/* The Hero Verdict Card */}
            <div className="bg-surface-container-lowest rounded-xl shadow-[0_8px_30px_rgba(0,0,0,0.08)] border border-surface-variant p-6 relative overflow-hidden spring-hover">
              <div
                className={`absolute top-0 left-0 w-full h-2 ${
                  isMalignant ? "bg-error" : "bg-secondary"
                }`}
              />
              <div className="flex justify-between items-start mb-6 mt-2">
                <h2 className="font-headline-sm text-headline-sm text-on-surface">
                  AI Diagnostic Verdict
                </h2>
                <div
                  className={`px-4 py-1 rounded-full flex items-center gap-2 border ${
                    isMalignant
                      ? "bg-error/10 text-error border-error/20"
                      : "bg-secondary/10 text-secondary border-secondary/20"
                  }`}
                >
                  <span className="material-symbols-outlined text-[18px]">
                    {isMalignant ? "warning" : "check_circle"}
                  </span>
                  <span className="font-label-caps text-label-caps tracking-wider font-bold">
                    {verdictLabel(verdict)}
                  </span>
                </div>
              </div>
              <div className="flex items-baseline gap-2 mb-4">
                <span className="font-headline-lg text-headline-lg text-on-surface">
                  {(confidence * 100).toFixed(1)}
                  <span className="text-headline-sm">%</span>
                </span>
                <span className="font-body-sm text-body-sm text-on-surface-variant uppercase tracking-wide">
                  Confidence
                </span>
              </div>
              {/* Confidence Bar */}
              <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full relative ${
                    isMalignant ? "bg-error" : "bg-secondary"
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                >
                  <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-r from-transparent to-white/30" />
                </div>
              </div>
              <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant mb-6">
                <span>Low</span>
                <span>{isMalignant ? "High Risk" : "Low Risk"}</span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-6 leading-relaxed">
                The ensemble model indicates a{" "}
                <strong>{verdict === "malignant" ? "high" : "low"}</strong>{" "}
                probability of malignancy based on fused predictions from{" "}
                {result.selected_models?.length || 3} PSO-selected models, with
                a fused probability of{" "}
                {((result.fused_probability ?? 0) * 100).toFixed(1)}%.
              </p>
              {/* Actions */}
              <div className="grid grid-cols-2 gap-3">
                {result.id && (
                  <ReportGenerator
                    predictionId={result.id}
                    className="col-span-2 bg-surface-container-low border border-outline-variant hover:bg-surface-variant transition-colors text-on-surface py-2 rounded-lg font-body-md text-body-md flex justify-center items-center gap-2"
                  />
                )}
                <button
                  type="button"
                  className="bg-surface-container-low border border-outline-variant hover:bg-surface-variant transition-colors text-on-surface py-2 rounded-lg font-body-md text-body-md flex justify-center items-center gap-2 touch-target"
                >
                  <span className="material-symbols-outlined text-[20px]">
                    share
                  </span>
                  Share
                </button>
                <button
                  type="button"
                  onClick={resetAnalysis}
                  className="text-primary hover:bg-primary/5 transition-colors py-2 rounded-lg font-body-md text-body-md flex justify-center items-center gap-2 touch-target"
                >
                  <span className="material-symbols-outlined text-[20px]">
                    add
                  </span>
                  New Analysis
                </button>
              </div>
            </div>

            {/* Model Breakdown Card */}
            <div className="bg-surface-container-lowest rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.05)] border border-surface-variant p-6 spring-hover">
              <h3 className="font-headline-sm text-headline-sm text-on-surface mb-6">
                Ensemble Model Breakdown
              </h3>
              <ModelSelector
                modelBreakdown={result.model_breakdown}
                psoWeights={result.pso_weights}
                fusionWeights={result.fusion_weights}
              />
            </div>
          </>
        ) : (
          <div className="bg-surface-container-lowest rounded-xl shadow-card border border-surface-container-high p-6 flex flex-col items-center justify-center text-center gap-3 flex-grow">
            <span className="material-symbols-outlined text-5xl text-outline-variant">
              medical_information
            </span>
            <p className="font-body-md text-body-md text-on-surface-variant">
              {isSavedAnalysis ? "Loading the saved diagnostic verdict and evidence for this record." : "Select an image and start the analysis to see the AI diagnostic verdict here."}
            </p>
          </div>
        )}
      </section>
    </main>
  );
}
