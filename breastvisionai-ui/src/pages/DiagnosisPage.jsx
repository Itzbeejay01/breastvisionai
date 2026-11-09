import { useEffect, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import ImageUploader from "../components/ImageUploader";
import ReportGenerator from "../components/ReportGenerator";
import ExplanationPanel from "../components/ExplanationPanel";
import { verdictLabel } from "../utils/verdict";
import useStore from "../store/index";

export default function DiagnosisPage() {
  const { id } = useParams();
  const location = useLocation();
  const { currentResult, predict, fetchHistoryDetail, explainPrediction } = useStore();
  const result = currentResult;
  const [selected, setSelected] = useState(
    location.state?.selectedImage || null
  );
  const [loading, setLoading] = useState(false);
  const [errorEntry, setErrorEntry] = useState(null);
  const error = errorEntry && errorEntry.key === location.key ? errorEntry.message : null;
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [expandedKey, setExpandedKey] = useState(null);
  const explanationKey = [location.key, result?.id ?? "none"].join(":");
  const explanationExpanded = expandedKey === explanationKey;
  const [scanning, setScanning] = useState(false);
  const [xaiLoadingKey, setXaiLoadingKey] = useState(null);
  const xaiLoading = xaiLoadingKey === explanationKey;
  const [xaiErrorEntry, setXaiErrorEntry] = useState(null);
  const xaiError = xaiErrorEntry?.key === explanationKey ? xaiErrorEntry.message : null;
  const explanationRequest = useRef(0);

  useEffect(() => {
    const requestToken = ++explanationRequest.current;
    useStore.setState({ currentResult: null });
    if (!id) return;
    let cancelled = false;
    (async () => {
      try {
        const detail = await fetchHistoryDetail(id);
        if (!cancelled && requestToken === explanationRequest.current) {
          setSelected(null);
          useStore.setState({ currentResult: detail });
        }
      } catch {
        if (!cancelled && requestToken === explanationRequest.current) {
          setErrorEntry({ key: location.key, message: "Failed to load history result" });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id, fetchHistoryDetail, location.key]);

  const handleSelect = (item) => {
    explanationRequest.current += 1;
    setSelected(item);
    setExpandedKey(null);
    setXaiLoadingKey(null);
    setXaiErrorEntry(null);
    setErrorEntry(null);
    setShowHeatmap(false);
    useStore.setState({ currentResult: null });
  };

  const runPrediction = async () => {
    if (!selected) return;
    explanationRequest.current += 1;
    setLoading(true);
    setErrorEntry(null);
    setExpandedKey(null);
    setXaiLoadingKey(null);
    setXaiErrorEntry(null);
    setShowHeatmap(false);
    setScanning(true);
    try {
      await predict(selected.file, selected.imageType);
    } catch (e) {
      setErrorEntry({ key: location.key, message: e.message || "Prediction failed" });
    } finally {
      setLoading(false);
      setScanning(false);
    }
  };

  const resetAnalysis = () => {
    explanationRequest.current += 1;
    setSelected(null);
    useStore.setState({ currentResult: null });
    setShowHeatmap(false);
    setXaiErrorEntry(null);
    setErrorEntry(null);
    setExpandedKey(null);
  };

  const generateExplanation = async () => {
    if (!currentResult?.id) return;
    if (currentResult.explanation_data && Object.keys(currentResult.explanation_data).length) {
      setExpandedKey(explanationKey);
      return;
    }
    const requestToken = ++explanationRequest.current;
    const resultId = currentResult.id;
    const activeKey = [location.key, resultId].join(":");
    setXaiLoadingKey(activeKey);
    setXaiErrorEntry(null);
    try {
      await explainPrediction(resultId);
      if (requestToken === explanationRequest.current && String(useStore.getState().currentResult?.id) === String(resultId)) {
        setExpandedKey(activeKey);
      }
    } catch (e) {
      if (requestToken === explanationRequest.current) setXaiErrorEntry({
        key: activeKey,
        message: e?.response?.data?.detail || e?.message || "Could not generate the explanation. Please try again.",
      });
    } finally {
      if (requestToken === explanationRequest.current) setXaiLoadingKey(null);
    }
  };

  const verdict = result?.prediction_result || result?.prediction;
  const isMalignant = verdict === "malignant";
  const confidence = result?.confidence ?? 0;
  const malignancyProbability = result?.prediction_probability ?? (isMalignant ? confidence : 1 - confidence);
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
                  crossOrigin="use-credentials"
                  className="object-cover w-full h-full opacity-90 transition-opacity duration-500"
                />
              ) : (
                <div className="text-on-surface-variant font-body-sm">
                  No image available
                </div>
              )}
              {result?.heatmap_base64 && showHeatmap && explanationExpanded && (
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
              {result?.heatmap_base64 && explanationExpanded && (
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
                  AI Model Prediction
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
                  {(malignancyProbability * 100).toFixed(1)}
                  <span className="text-headline-sm">%</span>
                </span>
                <span className="font-body-sm text-body-sm text-on-surface-variant uppercase tracking-wide">
                  AI-estimated probability of malignancy
                </span>
              </div>
              {/* Confidence Bar */}
              <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden mb-2">
                <div
                  className={`h-full rounded-full relative ${
                    isMalignant ? "bg-error" : "bg-secondary"
                  }`}
                  style={{ width: `${malignancyProbability * 100}%` }}
                >
                  <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-r from-transparent to-white/30" />
                </div>
              </div>
              <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant mb-6">
                <span>Low</span>
                <span>Model prediction: {verdictLabel(verdict)}</span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-6 leading-relaxed">
                The model estimates a {(malignancyProbability * 100).toFixed(1)}% probability of malignancy. This is an AI model prediction, not a confirmed diagnosis.
              </p>
              <p className="font-body-sm text-body-sm text-on-surface-variant -mt-2 mb-5">Model confidence for the displayed class: {(confidence * 100).toFixed(1)}%.</p>
              <section className="bg-surface-container-low rounded-xl p-5 mb-5">
                <h3 className="font-headline-sm text-headline-sm text-on-surface">How Does the AI Reach This Result?</h3>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-2 leading-relaxed">Three image-recognition models independently estimate malignancy likelihood from patterns learned during training. Their estimates are combined using equal-weight probability fusion. Gradient Boosting then reviews the inputs used by its trained configuration ({(result.meta_feature_schema || []).join(", ") || "model probabilities"}) and produces the final model prediction.</p>
                <div className="mt-4 flex flex-col sm:flex-row gap-2 items-center justify-between text-center text-xs font-semibold text-on-surface-variant">
                  {["Ultrasound", "Three CNN models", "Probability fusion", "Gradient Boosting", "Final result"].map((step, index) => <div key={step} className="flex items-center gap-2"><div className="rounded-lg bg-surface-container-lowest px-3 py-3">{step}</div>{index < 4 && <span className="hidden sm:block text-primary">→</span>}</div>)}
                </div>
                <p className="mt-4 text-xs leading-relaxed text-on-surface-variant">This AI-generated result is intended for research and decision-support purposes. It is not a substitute for clinical assessment, radiological interpretation or pathological confirmation.</p>
              </section>
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

      {result && (
        <section className="lg:col-span-12">
          {explanationExpanded ? (
            <div className="space-y-4">
              <div className="flex justify-end">
                <button type="button" onClick={() => setExpandedKey(null)} className="text-primary hover:bg-primary/5 px-4 py-2 rounded-lg font-body-md">Hide Explanation</button>
              </div>
              <ExplanationPanel result={result} />
            </div>
          ) : (
            <div className="bg-surface-container-lowest rounded-xl border border-surface-variant shadow-sm p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-5">
              <div>
                <p className="font-label-caps text-label-caps text-primary mb-2">OPTIONAL DETAIL</p>
                <h2 className="font-headline-md text-headline-md text-on-surface">Want to see the image evidence and decision factors?</h2>
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-2">Detailed image maps, model agreement and feature attributions are created only when you ask.</p>
                {xaiError && <p role="alert" className="text-error font-body-sm mt-3">{xaiError} You can retry below.</p>}
              </div>
              <button type="button" onClick={generateExplanation} disabled={xaiLoading || !result.id} className="bg-primary text-on-primary px-6 py-3 rounded-full font-body-md shadow-sm disabled:opacity-50 flex items-center justify-center gap-2 min-w-[240px]">
                <span aria-hidden="true" className={xaiLoading ? "material-symbols-outlined animate-spin" : "material-symbols-outlined"}>{xaiLoading ? "progress_activity" : "psychology"}</span>
                {xaiLoading ? "Generating explanation..." : "Explain This Result"}
              </button>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
