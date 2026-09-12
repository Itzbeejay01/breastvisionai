import { Link } from "react-router-dom";
import Footer from "../components/Footer";

const FEATURES = [
  { icon: "account_tree", tone: "landing-card-teal", tag: "01 / ENSEMBLE", title: "Three models. One clearer signal.", desc: "EfficientNet, ResNet, and VGG16 work together through PSO-optimized fusion for dependable diagnostic confidence." },
  { icon: "visibility", tone: "landing-card-lilac", tag: "02 / EXPLAINABILITY", title: "See what the model sees.", desc: "Grad-CAM heatmaps reveal the tissue regions influencing each verdict, helping clinicians review every result with context." },
  { icon: "dynamic_feed", tone: "landing-card-coral", tag: "03 / SCALE", title: "Built for real workflows.", desc: "Move from one scan to a full triage queue with batch processing for DICOM, PNG, and JPEG studies." },
];

const WORKFLOW_STEPS = [
  { icon: "cloud_upload", number: "01", title: "Upload", desc: "Add a raw or pre-processed scan." },
  { icon: "biotech", number: "02", title: "Analyze", desc: "Three models evaluate it in parallel." },
  { icon: "assignment_turned_in", number: "03", title: "Review", desc: "Receive a calibrated clinical verdict." },
  { icon: "visibility", number: "04", title: "Explain", desc: "Inspect the diagnostic heatmap." },
];

export default function WelcomePage() {
  return (
    <div className="landing-page min-h-screen flex flex-col font-body-md text-body-md relative overflow-hidden">
      <div className="landing-orb landing-orb-one" /><div className="landing-orb landing-orb-two" />
      <header className="landing-header fixed top-0 left-0 w-full z-50 flex justify-between items-center px-5 md:px-10 h-20">
        <Link to="/" className="flex items-center gap-3 text-headline-md font-headline-md font-bold text-primary"><span className="landing-logo"><img src="/logo.png" alt="" width={31} height={31} /></span>BreastVisionAI</Link>
        <Link to="/login" className="landing-header-action">Sign in <span className="material-symbols-outlined text-[18px]">arrow_forward</span></Link>
      </header>

      <main className="flex-grow pt-20 relative z-10">
        <section className="landing-hero max-w-container-max mx-auto px-5 md:px-10 pt-16 md:pt-24 pb-20">
          <div className="grid lg:grid-cols-[.95fr_1.05fr] gap-12 lg:gap-20 items-center">
            <div className="max-w-2xl">
              <div className="landing-eyebrow mb-7"><span className="landing-eyebrow-dot" /> CLINICAL AI / IMAGING INTELLIGENCE</div>
              <h1 className="font-display text-5xl sm:text-6xl lg:text-[5.25rem] leading-[.98] font-semibold tracking-[-.06em] text-on-surface">See the signal<br /><span className="landing-gradient-text">behind every scan.</span></h1>
              <p className="mt-7 max-w-xl text-lg md:text-xl leading-8 text-on-surface-variant">BreastVisionAI turns complex breast imaging into a clearer, explainable starting point for clinical research and review.</p>
              <div className="flex flex-col sm:flex-row gap-4 mt-9"><Link to="/login" className="landing-primary-button">Open the workspace <span className="material-symbols-outlined">arrow_outward</span></Link><a href="#capabilities" className="landing-secondary-button">Explore capabilities <span className="material-symbols-outlined">south</span></a></div>
              <div className="flex flex-wrap items-center gap-5 mt-9 text-sm text-on-surface-variant"><span className="flex items-center gap-2"><span className="material-symbols-outlined text-secondary text-[19px]">verified</span> Explainable results</span><span className="flex items-center gap-2"><span className="material-symbols-outlined text-primary text-[19px]">lock</span> Secure workspace</span></div>
            </div>
            <div className="landing-visual relative">
              <div className="landing-visual-halo" />
              <div className="landing-scan-card"><div className="landing-scan-top"><span><i /> LIVE MODEL VIEW</span><span>SCAN 0248</span></div><div className="landing-scan-image"><img src="/hero-scan.png" alt="AI analysis visualization of a breast scan" /><div className="landing-crosshair landing-crosshair-a" /><div className="landing-crosshair landing-crosshair-b" /><div className="landing-scan-line" /></div><div className="landing-scan-bottom"><div><p>ENSEMBLE CONFIDENCE</p><strong>94.8<span>%</span></strong></div><div className="landing-risk-pill"><span /> LOW RISK</div></div></div>
              <div className="landing-float-card landing-float-card-top"><span className="material-symbols-outlined">hub</span><div><strong>3 models</strong><small>working in parallel</small></div></div><div className="landing-float-card landing-float-card-bottom"><span className="material-symbols-outlined">visibility</span><div><strong>Grad-CAM ready</strong><small>explain every verdict</small></div></div>
            </div>
          </div>
        </section>

        <section id="capabilities" className="max-w-container-max mx-auto px-5 md:px-10 pb-24"><div className="flex flex-col md:flex-row justify-between md:items-end gap-5 mb-9"><div><p className="landing-section-kicker">WHY BREASTVISIONAI</p><h2 className="font-display text-3xl md:text-5xl font-semibold mt-3">Intelligence you can<br /><span className="landing-gradient-text-alt">understand and act on.</span></h2></div><p className="max-w-sm text-on-surface-variant leading-7">A focused toolkit for imaging teams who need speed, transparency, and a better view of the evidence.</p></div><div className="grid md:grid-cols-3 gap-5">{FEATURES.map((feature) => <article key={feature.title} className={`landing-feature-card ${feature.tone}`}><div className="flex justify-between items-start"><span className="landing-feature-icon material-symbols-outlined">{feature.icon}</span><span className="landing-feature-tag">{feature.tag}</span></div><div className="mt-14"><h3 className="font-display text-2xl font-semibold leading-tight">{feature.title}</h3><p className="mt-4 leading-7 opacity-80">{feature.desc}</p></div><span className="material-symbols-outlined landing-card-arrow">arrow_outward</span></article>)}</div></section>

        <section className="landing-workflow-wrap mx-5 md:mx-10 mb-24"><div className="max-w-container-max mx-auto px-5 md:px-10 py-16 md:py-20"><div className="grid lg:grid-cols-[.7fr_1.3fr] gap-12 items-start"><div><p className="landing-section-kicker">A SIMPLE PATH TO CLARITY</p><h2 className="font-display text-3xl md:text-5xl font-semibold mt-3">From image<br />to insight.</h2><p className="mt-5 max-w-sm text-on-surface-variant leading-7">A calm, repeatable workflow that keeps the clinical team in control at every step.</p></div><div className="grid sm:grid-cols-2 gap-x-8 gap-y-10">{WORKFLOW_STEPS.map((step) => <div key={step.number} className="landing-step"><div className="flex items-center justify-between"><span className="landing-step-number">{step.number}</span><span className="material-symbols-outlined text-primary">{step.icon}</span></div><h3 className="font-display text-xl font-semibold mt-5">{step.title}</h3><p className="mt-2 text-sm text-on-surface-variant leading-6">{step.desc}</p></div>)}</div></div></div></section>

        <section className="max-w-4xl mx-auto px-5 md:px-10 text-center pb-28"><div className="landing-cta"><span className="material-symbols-outlined landing-cta-icon">radiology</span><p className="landing-section-kicker">READY WHEN YOU ARE</p><h2 className="font-display text-3xl md:text-5xl font-semibold mt-3">Make every scan<br /><span className="landing-gradient-text">more meaningful.</span></h2><p className="mt-5 text-on-surface-variant leading-7 max-w-lg mx-auto">Sign in to access your secure analysis workspace and start exploring BreastVisionAI.</p><Link to="/login" className="landing-primary-button mt-8 inline-flex">Enter the platform <span className="material-symbols-outlined">arrow_forward</span></Link></div></section>
      </main><Footer />
    </div>
  );
}
