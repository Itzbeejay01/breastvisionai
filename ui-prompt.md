# BreastVisionAI — UI/UX Design Prompt for Google Stitch

> Paste this entire document into Google Stitch (or any UI design tool) as the design brief.
> It covers the product context, the 4 pages, the design system, interaction states, and
> accessibility requirements. Design in high fidelity: modern, polished, and richly interactive.

---

## 1. Product Context

**Product:** BreastVisionAI — a clinical research tool that detects breast cancer (benign vs. malignant) from ultrasound images using an ensemble of 3 PSO-optimized deep learning models (EfficientNet, ResNet, VGG16) with late fusion and a GradientBoosting meta-learner, plus Grad-CAM heatmap explainability.

**Users:** Radiologists, clinicians, and medical researchers. They are decision-makers under time pressure. They need to trust the tool quickly: results must be legible at a glance, explainable, and never ambiguous.

**Core workflow (the "Decide" pattern):** upload/select an image → run analysis → see a verdict with confidence, per-model breakdown, and a visual heatmap. The verdict is the hero; everything else supports it.

**Domain artifact:** The diagnosis — a malignancy probability backed by per-model evidence and a Grad-CAM heatmap overlay on the original image.

**What makes this product trustworthy (evidence):** confidence scores, the per-model probability breakdown, the PSO-selected weights, the Grad-CAM heatmap overlaid on the actual image, and prediction history.

---

## 2. Design Direction

### Overall character
- **Modern, calm, clinical, and confident.** Not a generic SaaS dashboard, and absolutely not a scary "dark red alert" medical app. Think premium medical device software (like Philips or GE imaging consoles): composed, trustworthy, precise.
- **Interactivity everywhere.** Micro-interactions, animated state transitions, live progress feedback, smooth page transitions, animated data reveals. The app should feel alive while remaining serious.
- **The 60-30-10 rule.** 60% clean neutral surface, 30% supporting structure, 10% accent used sparingly so the verdict color always means something.

### Color system
- **Primary / brand:** a clinical teal or deep cyan (not blue-violet — refuse the generic tech hue). Teal reads "medical, clean, advanced" without the coldness of pure blue.
- **Semantic verdict colors — never use them for decoration:**
  - **Malignant:** a carefully chosen red (e.g., `#D64545` range) — used ONLY for the malignant verdict, low-confidence states, and destructive actions.
  - **Benign:** a medical green (e.g., `#2E9E6B` range) — used ONLY for the benign verdict and success states.
  - **Moderate/amber:** for mid-confidence (80–95%) states.
- **Neutrals:** light gray-blue tints (`#F7F9FA`, `#E9EEF1` surfaces, `#6B7A82` secondary text, `#101418` near-black text). Tint neutrals subtly toward the brand hue so gray never feels empty.
- **Ensure red/green verdict colors remain distinguishable under deuteranopia/protanopia** (check contrast and pair with icons, not color alone).
- **Background:** light, airy. Use soft glassmorphism sparingly for floating elements, never for primary surfaces.

### Typography
- A **humanist sans-serif** for body and UI (e.g., Inter, Figtree, or system font stack) — friendly but professional.
- Optional **serif or mono accent** for the data/numerical readouts (confidence %, probability) to give them a "scientific instrument" feel.
- Clear 3-level hierarchy everywhere: hook (page title), bridge (subtitle), detail (body).
- Body measure 60–76 characters. Numbers use tabular figures so they align in tables and stat grids.
- Never below 14px for body text in the UI; 16px+ for primary actions.

### Shape, depth, and borders
- **Large soft radii** (12–16px) on cards and modals; **pill shapes** reserved for verdict badges and confidence chips.
- **Subtle, layered shadows** (2–3 layers, low opacity) for elevation instead of heavy borders. Border strokes stay at 1px, low-contrast grays.
- **Depth planes:** background canvas → content cards → floating attention elements (toasts, modals, heatmap popovers) that animate in from the edge nearest their trigger.

### Motion system (professional, not playful)
- **Physics:** cards and modals use a gentle spring (tension ~170, damping ~0.8); tooltips and micro-popovers are fast (100–150ms) with light damping; never bouncy.
- **Entrance choreography:** staggered reveals with `delay = index × 20ms + small jitter`. Animate opacity + transform (translateY 8–12px, scale 0.98→1) only. No layout-thrashing animations.
- **Loading states name the work:** "Analyzing image…", "Running 3-model ensemble…", "Generating Grad-CAM heatmap…" with a progress shimmer or step indicator.
- **Data reveals:** confidence bars fill, probability numbers count up, model breakdown bars grow with a subtle stagger when a result lands.
- **Transitions between pages/verdicts:** smooth 200–300ms fades/slides; a verdict card "resolves" into place with a soft pop.
- **Respect `prefers-reduced-motion`:** provide reduced-motion fallback (instant or 100ms fades only).

### Interactive patterns to include
- **Hover:** lift + shadow deepen on cards; verdict badges glow subtly; images zoom slightly on hover with a zoom affordance.
- **Active:** scale 0.98 press feedback on all buttons.
- **Focus:** visible 2–3px focus rings (3:1 contrast), never removed.
- **Live feedback:** button loading spinners inside the button, disabled states with reason, undoable actions where possible.
- **Touch targets:** minimum 44×44px.
- **Empty / loading / error / success states designed for every surface** — never a blank box.

---

## 3. App Shell & Navigation

- **Sticky top navbar** with the product logo (🩺 or a custom breast/medical glyph), the name "BreastVisionAI", and nav links: Dashboard, Diagnosis, Batch, Models. Active link clearly highlighted. On mobile, collapse to a bottom tab bar (thumb zone) or a hamburger menu.
- **Global system status indicator** in the navbar or a small dot: API online/offline, models loaded, ensemble ready. Subtle green pulse when healthy, amber when warming up, red when offline.
- **Consistent page shell:** max-width 1280px, generous padding (24–32px), 24px vertical rhythm between sections.

---

## 4. Page Designs

### 4.1 Dashboard (`/`)

**Purpose:** fast entry into the tool + at-a-glance system health.

- **Hero row:** "Quick Diagnosis" — the two-way image selector (see §5) front and center. This is the primary action; make it the visual anchor.
- **System status strip:** 4 stat cards — API status, Models Loaded (3), PSO-Selected (3), Ensemble (GradientBoosting). Animated count-up numbers, subtle icon per card, color-coded health.
- **Recent predictions:** a compact table/list with thumbnail, filename, timestamp, verdict badge, and confidence. Hover rows highlight; click navigates to the diagnosis detail. Include a "View all" link.
- Optional: a small "How it works" strip (Upload → Ensemble → Verdict → Heatmap) as an illustrated 4-step flow for first-time users.
- **Interactivity:** staggered card entrance, hover lifts, live-updating health pulse, and a subtle empty state when no predictions exist yet.

### 4.2 Diagnosis (`/diagnosis`, `/diagnosis/:id`)

**Purpose:** the core decision surface. Left = input, right = verdict.

- **Left panel — image selection:** the two-way selector (§5). Once an image is selected, show a large preview with the chosen mode badge (Raw / Pre-processed) and a prominent "Start Analysis" button.
- **Right panel — the verdict (the hero):**
  - A **verdict card** that resolves in with a soft pop: large icon, "MALIGNANT" or "BENIGN" badge, malignancy probability as a giant number (count-up animation), and the ensemble method caption.
  - **Confidence bar** with 3-zone coloring (≥95% green, 80–95% amber, <80% red) and an animated fill.
  - **Model breakdown:** per-model rows (EfficientNet, ResNet, VGG16) with probability bars that grow staggered, weight chips, and a mini PSO-weight distribution.
  - **Grad-CAM heatmap:** the original image with the heatmap overlay as a toggleable layer (slider or toggle button "Show heatmap" / "Show original"), plus a side-by-side comparison mode. Animate the overlay fade-in.
  - **Actions:** "Download PDF report", "Share", "New analysis".
- **History mode (`/diagnosis/:id`):** same layout, but read-only with a "viewed from history" banner and the original timestamp.
- **Interactivity:** verdict resolve animation, staggered breakdown bars, heatmap toggle with crossfade, button loading states during analysis, error state with retry.

### 4.3 Batch Analysis (`/batch`)

**Purpose:** multi-image triage (up to 50 images) with a results table.

- **Upload zone:** large drag-and-drop area (react-dropzone) accepting PNG/JPEG/DICOM, with a per-file image-type toggle (Raw / Pre-processed), file count, and a clear list of queued files with thumbnails and remove buttons.
- **Run button:** "Run Batch Analysis" with a prominent loading state showing progress ("Analyzing 12/50…") — use a progress bar or ring that advances per image.
- **Results:** summary stat cards (Total, Malignant, Benign, Avg Confidence) with count-up animations, then a dense results table: index, thumbnail, filename, verdict badge, confidence bar (mini), fused probability. Sortable columns. Row click → open that image's diagnosis.
- **Export:** "Export CSV" button with a success toast on download.
- **Interactivity:** file-add animations (thumbnails pop in), per-row reveal stagger after analysis, live progress updates, empty state with a clear call to action.

### 4.4 Model Comparison (`/models`)

**Purpose:** technical transparency — show which models PSO selected and why.

- **Model metrics table:** all 5 models (EfficientNet, DenseNet, ResNet, VGG16, Xception) with columns: Model, PSO Selected (badge/check), PSO Weight, Fusion Weight, Accuracy, Precision, Recall, F1, AUC, Specificity. Selected rows subtly tinted with the brand color; sortable columns.
- **PSO weight distribution:** horizontal animated bar chart (or Recharts) showing each model's weight; selected models highlighted.
- **Stacking ensemble metrics:** stat cards for accuracy/precision/recall/F1/AUC of the meta-learner, plus a "Meta-Learner: GradientBoostingClassifier" info card.
- **Interactivity:** bars grow on scroll into view, hover tooltips on metrics, table row hover, sort animations.

---

## 5. The Signature Component: Two-Way Image Selector

This is the product's core UX pattern — design it exceptionally well.

**Tab 1 — "Upload New Image":**
- Large drag-and-drop zone (PNG, JPEG, DICOM) with an animated dashed-border pulse when dragging.
- Below it, an **image-type radio toggle**: "Raw (needs preprocessing)" vs "Pre-processed (normalized)". Use segmented control styling with a subtle slide animation between the two options, each with a short explanatory tooltip.
- After selection: thumbnail preview card slides in with filename, type badge, size, and a change/remove control.

**Tab 2 — "Select Pre-processed":**
- A grid gallery of previously processed images from history (thumbnail, filename, verdict color strip, date).
- Click to select → the card lifts and shows a checkmark ring; "Use this image" confirmation appears.
- Loading skeleton grid while fetching.

**Upload-to-analysis flow:** after selection, an inline "Start Analysis" CTA with a magnetic/hover lift. During analysis the preview card shows a scanning shimmer line sweeping across the image (like a medical scanner) — a distinctive, on-brand loading treatment.

---

## 6. States Every Surface Must Have

Design all 9 states, not just the happy path:

1. **Idle** — at rest
2. **Hover** — anticipation (lift, glow, cursor)
3. **Active** — pressed (scale 0.98)
4. **Focused** — visible keyboard focus ring
5. **Loading** — naming the actual work ("Analyzing…", "Generating heatmap…") with progress
6. **Empty** — teaches the space ("No predictions yet. Run your first analysis to see results here.")
7. **Error** — recovery path: what broke, why, and a Retry action; never blame the user
8. **Disabled** — greyed with a reason shown on hover
9. **Overflow** — long filenames truncate with ellipsis + tooltip; tables scroll horizontally

---

## 7. Accessibility (non-negotiable floor)

- Verdict never communicated by color alone: always pair with icon + text label.
- All interactive elements are real buttons/links with visible focus rings; full keyboard navigation; logical tab order; modals trap focus and restore it on close.
- Form inputs have always-visible labels (placeholders are examples, not labels).
- Touch targets ≥ 44×44px.
- Survives 200% zoom and reflows cleanly at 320px.
- `prefers-reduced-motion` respected; `prefers-contrast` honored (high-contrast mode keeps verdict colors legible).

---

## 8. Responsive Behavior

- **Desktop (1024–1440+):** two-column diagnosis layout; full data tables; navbar links.
- **Tablet (768):** diagnosis stacks to single column (verdict below input); tables compress; bottom-nav or condensed header.
- **Mobile (320–375):** bottom tab bar in the thumb zone; single column everything; the verdict card is the first thing after the image preview; heatmap toggle becomes a full-width segmented control; large tap targets.
- Adapt by container, not just breakpoint, where practical (e.g., the model-breakdown card reflows whether it sits in a 400px or 700px column).

---

## 9. Design Do's and Don'ts

**Do:**
- Keep the verdict and confidence as the visual centerpiece of every result.
- Use brand teal as the voice; red/green only for semantic verdicts.
- Make every state animated: entrances, reveals, fills, transitions.
- Design for real data: long filenames, 50-image batches, 5 decimal probabilities, 3 model names.
- Let the Grad-CAM heatmap feel like the "money shot" of the diagnosis view.

**Don't:**
- Don't use generic blue-violet gradients, purple CTAs, or AI-slop illustrations.
- Don't make it look like a generic SaaS analytics dashboard.
- Don't use red as a decorative color or make malignant feel like a system error.
- Don't bury the confidence score; it's the second most important thing after the verdict.
- Don't use cards inside cards; flatten with type and dividers.
- Don't skip empty/loading/error states for polish.

---

## 10. Deliverables Expected from Stitch

- Full high-fidelity design for all 4 pages + the app shell.
- A reusable component library: buttons (all states), confidence bar, verdict badge, stat card, model-breakdown row, heatmap viewer, two-way image selector, upload dropzone, data table, segmented control, toast, skeleton, modal.
- Design tokens: color scales (with contrast-checked semantic pairs), type scale, spacing scale (4px base), radii, shadows, motion durations/easings.
- Annotated interaction notes on key animations and state transitions so they can be implemented in React (Tailwind + Framer Motion).
- Responsive layouts for desktop, tablet, and mobile for at least Dashboard and Diagnosis.

---

*End of prompt. Save output as a Stitch project and share the component + token library for implementation.*
