import { useEffect, useMemo, useState } from "react";
import Footer from "../components/Footer";
import { apiHelpers } from "../api/client";

const MODEL_ORDER = ["EfficientNet", "DenseNet", "ResNet", "VGG16", "Xception"];
const METRICS = [
  ["accuracy", "Accuracy", "target"],
  ["precision", "Precision", "pink"],
  ["recall", "Recall", "green"],
  ["f1", "F1 score", "purple"],
  ["auc", "AUC-ROC", "blue"],
];

function asRatio(value) {
  if (value == null || Number.isNaN(Number(value))) return null;
  const number = Number(value);
  return number > 1 ? number / 100 : number;
}

function percent(value) {
  const ratio = asRatio(value);
  return ratio == null ? "—" : `${(ratio * 100).toFixed(1)}%`;
}

export default function ModelComparisonPage() {
  const [models, setModels] = useState([]);
  const [ensemble, setEnsemble] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [modelList, ensembleConfig] = await Promise.all([apiHelpers.getModels(), apiHelpers.getEnsemble()]);
        if (!cancelled) { setModels(modelList || []); setEnsemble(ensembleConfig); }
      } catch { if (!cancelled) setError("Failed to load model configuration"); }
      finally { if (!cancelled) setLoading(false); }
    })();
    return () => { cancelled = true; };
  }, []);

  const selected = ensemble?.selected_models || [];
  const weights = useMemo(() => {
    const pso = ensemble?.pso_weights || {};
    const fusion = ensemble?.fusion_weights || {};
    return Object.keys(pso).length ? pso : fusion;
  }, [ensemble]);
  const weightTotal = Object.values(weights).reduce((sum, value) => sum + (Number(value) || 0), 0) || 1;
  const metrics = ensemble?.stacking_metrics || {};

  if (loading) return <main className="models-page flex-grow pt-28 px-5 md:px-10 max-w-container-max mx-auto w-full"><div className="models-loading"><span className="material-symbols-outlined animate-spin">progress_activity</span> Loading ensemble intelligence…</div></main>;

  return (
    <main className="models-page flex-grow pt-28 pb-12 px-5 md:px-10 max-w-container-max mx-auto w-full">
      <header className="models-hero"><div><p className="models-kicker">MODEL INTELLIGENCE / 01</p><h1 className="font-display text-4xl md:text-6xl font-semibold tracking-tight mt-3">The science<br /><span>behind the signal.</span></h1><p className="mt-5 max-w-xl text-on-surface-variant text-lg leading-8">Explore the ensemble architecture, validation performance, and contribution of every model in the BreastVisionAI pipeline.</p></div><div className="models-hero-badge"><span className="material-symbols-outlined">account_tree</span><div><strong>{selected.length || 3} active models</strong><small>PSO-optimized ensemble</small></div></div></header>
      {error && <p className="text-error mb-5">{error}</p>}

      <section className="models-metric-grid mb-7">{METRICS.map(([key, label, tone]) => <div className={`models-metric-card ${tone}`} key={key}><div className="models-metric-icon"><span className="material-symbols-outlined">{key === "auc" ? "insights" : key === "accuracy" ? "verified" : "analytics"}</span></div><p>{label}</p><strong>{percent(metrics[key])}</strong><small>Test-set performance</small></div>)}</section>

      <section className="models-weight-card mb-7"><div className="models-section-heading"><div><p className="models-kicker">ENSEMBLE COMPOSITION</p><h2 className="font-display text-2xl md:text-3xl font-semibold mt-2">Contribution by model</h2></div><span className="models-method-pill"><span className="material-symbols-outlined">auto_awesome</span> {ensemble?.fusion_method || "Late fusion"}</span></div><p className="text-on-surface-variant mt-3 max-w-2xl leading-7">The line width and percentage below are calculated from the API weight values. Selected models contribute to the final fused probability.</p><div className="models-bars mt-8">{MODEL_ORDER.map((name, index) => { const raw = Number(weights[name] || 0); const share = Math.max(0, Math.min(100, (raw / weightTotal) * 100)); const active = selected.includes(name); return <div className="models-bar-row" key={name}><div className="models-bar-label"><span className={`models-model-dot ${active ? "active" : ""}`} />{name}{active && <em>ACTIVE</em>}</div><div className="models-bar-track"><div className={`models-bar-fill ${active ? "active" : ""}`} style={{ width: `${share}%`, animationDelay: `${index * 100}ms` }} /></div><strong>{share.toFixed(1)}%</strong></div>; })}</div>{!Object.keys(weights).length && <p className="text-on-surface-variant mt-5 text-sm">No ensemble weights were returned by the API.</p>}</section>

      <section className="models-table-card"><div className="models-section-heading"><div><p className="models-kicker">BASELINE VIEW</p><h2 className="font-display text-2xl md:text-3xl font-semibold mt-2">Model performance</h2></div><span className="models-count-badge">{models.length} architectures</span></div><div className="overflow-x-auto mt-6"><table className="models-table"><thead><tr><th>MODEL</th><th>STATUS</th><th>ACCURACY</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>AUC</th></tr></thead><tbody>{models.map((model) => { const active = model.selected_by_pso; return <tr key={model.name}><td><div className="models-name"><span className={`models-model-dot ${active ? "active" : ""}`} /><strong>{model.name}</strong></div></td><td>{active ? <span className="models-status-active"><span className="material-symbols-outlined">check_circle</span> Selected</span> : <span className="models-status-idle">Available</span>}</td>{["accuracy", "precision", "recall", "f1", "auc"].map((key) => <td key={key}>{percent(model.metrics?.[key])}</td>)}</tr>; })}</tbody></table></div><p className="models-table-note"><span className="material-symbols-outlined">info</span> Base-model metrics display when their evaluation files are available. Ensemble metrics above are loaded from the current test-set report.</p></section>
      <Footer />
    </main>
  );
}
