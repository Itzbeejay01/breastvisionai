import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import ImageUploader from "../components/ImageUploader";
import SystemStatus from "../components/SystemStatus";
import Footer from "../components/Footer";
import { verdictLabel } from "../utils/verdict";
import useStore from "../store/index";

const PIPELINE = [
  ["upload", "01", "Upload", "Add a DICOM, PNG, or JPEG scan."],
  ["account_tree", "02", "Ensemble", "Three models evaluate it in parallel."],
  ["fact_check", "03", "Verdict", "Receive a calibrated confidence score."],
  ["layers", "04", "Explain", "Inspect the Grad-CAM heatmap."],
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const { history, fetchHistory, status, error, user } = useStore();
  const [selected, setSelected] = useState(null);

  useEffect(() => { fetchHistory(1); }, [fetchHistory]);

  const runDiagnosis = () => {
    if (selected) navigate("/diagnosis", { state: { selectedImage: selected } });
  };
  const recent = (history || []).slice(0, 5);

  return (
    <main className="dashboard-page flex-grow pt-28 pb-12 px-5 md:px-10 max-w-container-max mx-auto w-full">
      <header className="dashboard-heading flex flex-col md:flex-row md:items-end justify-between gap-6 mb-8">
        <div><p className="dashboard-kicker">CLINICAL RESEARCH WORKSPACE</p><h1 className="font-display text-4xl md:text-5xl font-semibold mt-2 tracking-tight">Good to see you, <span>{user?.username || "researcher"}.</span></h1><p className="text-on-surface-variant mt-3 text-lg">Your diagnostic workspace is ready when you are.</p></div>
        <div className="flex gap-3"><Link to="/models" className="dashboard-quiet-button"><span className="material-symbols-outlined">tune</span><span className="hidden sm:inline">Model settings</span></Link><Link to="/batch" className="dashboard-dark-button"><span className="material-symbols-outlined">layers</span> Batch workspace</Link></div>
      </header>

      <div className="dashboard-status mb-7"><SystemStatus /></div>

      <section className="dashboard-hero-grid mb-7">
        <div className="dashboard-upload-card">
          <div className="dashboard-card-header"><div><p className="dashboard-kicker">START HERE</p><h2 className="font-display text-3xl font-semibold mt-2">Quick diagnosis</h2><p className="text-on-surface-variant mt-2 max-w-xl">Upload one scan for an immediate ensemble verdict with explainable AI.</p></div><div className="dashboard-live-badge"><span /> SINGLE IMAGE</div></div>
          <div className="dashboard-upload-content">
            {selected ? <div className="dashboard-selected-state"><div className="dashboard-preview-frame"><img src={selected.preview} alt={selected.file?.name || "Selected scan"} /><span className="dashboard-preview-label"><span className="material-symbols-outlined">check_circle</span> Scan ready</span></div><div className="text-center"><p className="font-semibold truncate max-w-sm">{selected.file?.name || "Selected image"}</p><span className={`inline-flex mt-2 px-3 py-1 rounded-full text-[11px] font-bold tracking-wider ${selected.imageType === "raw" ? "bg-primary-container text-on-primary-container" : "bg-secondary-container text-on-secondary-container"}`}>{selected.imageType === "raw" ? "RAW IMAGE" : "PRE-PROCESSED"}</span></div><div className="flex flex-col sm:flex-row gap-3"><button type="button" onClick={runDiagnosis} className="dashboard-primary-button">Start analysis <span className="material-symbols-outlined">arrow_forward</span></button><button type="button" onClick={() => setSelected(null)} className="dashboard-quiet-button">Choose another</button></div></div> : <ImageUploader onSelect={setSelected} />}
          </div>
        </div>

        <aside className="dashboard-insight-card"><div className="dashboard-insight-glow dashboard-insight-glow-one" /><div className="dashboard-insight-glow dashboard-insight-glow-two" /><div className="relative z-10"><div className="dashboard-icon-bubble"><span className="material-symbols-outlined">auto_awesome</span></div><p className="dashboard-kicker dashboard-kicker-light mt-5">THE BREASTVISION ENGINE</p><h2 className="font-display text-3xl font-semibold mt-3">Clarity at<br />clinical speed.</h2><p className="text-white/70 mt-4 leading-7">A PSO-optimized ensemble designed to make complex imaging easier to review.</p></div><div className="relative z-10 dashboard-model-stack mt-9"><div className="flex justify-between items-center mb-4"><span className="text-sm font-semibold text-white">Active model ensemble</span><span className="text-xs text-[#b9fff2]">● ONLINE</span></div><div className="dashboard-model-row"><span>EfficientNet</span><b>01</b></div><div className="dashboard-model-row"><span>ResNet</span><b>02</b></div><div className="dashboard-model-row"><span>VGG16</span><b>03</b></div></div></aside>
      </section>

      <section className="grid xl:grid-cols-[.72fr_1.28fr] gap-7 mb-10">
        <div className="dashboard-pipeline-card"><div className="flex justify-between items-start mb-8"><div><p className="dashboard-kicker">YOUR WORKFLOW</p><h2 className="font-display text-2xl font-semibold mt-2">From image to insight</h2></div><span className="material-symbols-outlined text-primary">route</span></div><div className="dashboard-pipeline-list">{PIPELINE.map(([icon, number, title, desc]) => <div className="dashboard-pipeline-step" key={number}><div className="dashboard-step-icon"><span className="material-symbols-outlined">{icon}</span></div><div><span className="dashboard-step-number">{number}</span><h3 className="font-semibold mt-1">{title}</h3><p className="text-sm text-on-surface-variant mt-1 leading-6">{desc}</p></div></div>)}</div></div>

        <div className="dashboard-history-card"><div className="dashboard-history-header"><div><p className="dashboard-kicker">RECENT ACTIVITY</p><h2 className="font-display text-2xl font-semibold mt-2">Latest predictions</h2></div><Link to="/batch" className="dashboard-link">Open batch workspace <span className="material-symbols-outlined">arrow_outward</span></Link></div>{recent.length === 0 ? <div className="dashboard-empty-state"><div className="dashboard-empty-icon"><span className="material-symbols-outlined">image_search</span></div><h3 className="font-display text-xl font-semibold mt-4">Your results will appear here</h3><p className="text-on-surface-variant text-sm max-w-sm mt-2 leading-6">Start with a single scan above or move to batch processing for a larger queue.</p><Link to="/diagnosis" className="dashboard-primary-button mt-5">Start first analysis <span className="material-symbols-outlined">arrow_forward</span></Link></div> : <div className="overflow-x-auto"><table className="dashboard-table"><thead><tr><th>SCAN</th><th>FILE / DATE</th><th>VERDICT</th><th>CONFIDENCE</th><th /></tr></thead><tbody>{recent.map((prediction) => { const malignant = prediction.prediction_result === "malignant"; return <tr key={prediction.id}><td><div className="dashboard-thumb">{prediction.uploaded_image?.preview_url ? <img src={prediction.uploaded_image.preview_url} alt="" /> : <span className="material-symbols-outlined">image</span>}</div></td><td><p className="font-semibold truncate max-w-[180px]">{prediction.uploaded_image?.filename || `Prediction #${prediction.id}`}</p><small>{new Date(prediction.timestamp).toLocaleString()}</small></td><td><span className={`dashboard-verdict ${malignant ? "dashboard-verdict-danger" : "dashboard-verdict-safe"}`}><span className="material-symbols-outlined">{malignant ? "warning" : "check_circle"}</span>{verdictLabel(prediction.prediction_result)}</span></td><td><div className="dashboard-confidence"><div><i className={malignant ? "danger" : "safe"} style={{ width: `${(prediction.confidence ?? 0) * 100}%` }} /></div><span>{((prediction.confidence ?? 0) * 100).toFixed(1)}%</span></div></td><td><Link to={`/diagnosis/${prediction.id}`} className="dashboard-view-button" aria-label={`View prediction ${prediction.id}`}><span className="material-symbols-outlined">arrow_outward</span></Link></td></tr>})}</tbody></table></div>}{error && <p className="p-4 text-error text-sm border-t border-outline-variant">{error}</p>}{status === "loading" && <p className="p-4 text-on-surface-variant text-sm border-t border-outline-variant">Refreshing predictions…</p>}</div>
      </section>
      <Footer simple />
    </main>
  );
}
