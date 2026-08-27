## UI/UX Design Analysis for BreastVisionAI

### Current State Assessment

The project currently has **zero UI/UX** — it's purely a backend/ML pipeline:

- **Django backend** with only a health check endpoint (`/api/health/`)
- **No frontend** whatsoever (no templates, no static files, no React/Vue)
- **No user-facing interface** — everything runs via CLI (`python model_training/ensembles.py`)
- **No image upload**, no prediction display, no results visualization for end users
- The `src/` directories are empty placeholders

This is a **research-grade ML project** that needs to become a **clinical decision support tool**.

---

### Phase 1: Core User Flows & Personas

#### Primary Personas

| Persona       | Role         | Needs                                                                          | Technical Level                |
| ------------- | ------------ | ------------------------------------------------------------------------------ | ------------------------------ |
| **Dr. Ade**   | Radiologist  | Upload mammogram/MRI, get instant malignancy prediction, see confidence scores | Low — wants one-click          |
| **Dr. Bello** | Researcher   | Compare model outputs, see ensemble weights, download results for publication  | Medium — wants configurability |
| **Admin**     | System Admin | Monitor API health, manage users, view logs                                    | High — wants dashboards        |

#### Core User Flows

```
Flow 1: Single Image Classification (Dr. Ade — 90% of usage)
  Upload Image → Preview → Select Model(s) → Predict → View Results → Download Report

Flow 2: Batch Classification (Dr. Bello — 8% of usage)
  Upload Multiple Images → Select Ensemble Method → Batch Predict → Compare Results → Export CSV

Flow 3: Model Comparison (Dr. Bello — 2% of usage)
  Select Models → Upload Test Set → Run Comparison → View Metrics Table → View ROC Curves
```

---

### Phase 2: Screen Architecture

#### Screen 1: Dashboard / Home

```
┌─────────────────────────────────────────────────────────────┐
│  🏥 BreastVisionAI                    [Dr. Ade] [Logout]   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │  Quick Diagnosis     │  │  System Status               │  │
│  │                     │  │  ┌────────────────────────┐  │  │
│  │  [Drop image here]  │  │  │ ✅ API: Healthy        │  │  │
│  │  or click to upload │  │  │ ✅ Models: 5/5 loaded  │  │  │
│  │                     │  │  │ ✅ Ensemble: Active     │  │  │
│  │  [Start Analysis]   │  │  └────────────────────────┘  │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Recent Predictions                                   │   │
│  │  ┌──────┬────────┬──────────┬────────┬──────────┐   │   │
│  │  │ Date │ Result │ Confidence│ Model │ Report   │   │   │
│  │  ├──────┼────────┼──────────┼────────┼──────────┤   │   │
│  │  │ ...  │  ...   │   ...    │  ...   │ [View]   │   │   │
│  │  └──────┴────────┴──────────┴────────┴──────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### Screen 2: Prediction Result

```
┌─────────────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌────────────────────────────────────┐   │
│  │  Image Preview│  │  Diagnosis Result                  │   │
│  │              │  │                                    │   │
│  │  [Mammogram] │  │  🔴 Malignant Detected             │   │
│  │              │  │  Confidence: 98.22%                │   │
│  │              │  │  Ensemble: PSO-Weighted (3 models) │   │
│  │              │  │                                    │   │
│  │              │  │  [Download Report] [Share]         │   │
│  └──────────────┘  └────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Breakdown                                      │   │
│  │                                                       │   │
│  │  EfficientNetB0:  ████████████████░░ 96.2%           │   │
│  │  VGG16:           ██████████████░░░░ 94.8%           │   │
│  │  ResNet50:        █████████████░░░░░ 93.1%           │   │
│  │                                                       │   │
│  │  PSO Weights: [0.383, 0.289, 0.231]                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Heatmap Overlay (Grad-CAM)                          │   │
│  │  [Image with activation heatmap overlay]             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### Screen 3: Batch Analysis

```
┌─────────────────────────────────────────────────────────────┐
│  Batch Analysis                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Upload Multiple Images                               │   │
│  │  [Drop files here or click to browse]                 │   │
│  │  Supported: PNG, DICOM, JPEG (max 50 files)          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Ensemble Configuration                               │   │
│  │                                                       │   │
│  │  ○ PSO-Weighted (Recommended — 98.22% Acc)           │   │
│  │  ○ Stacking-GB (98.15% Acc)                          │   │
│  │  ○ Soft Voting (97.70% Acc)                          │   │
│  │  ○ All 5 Models (for comparison)                     │   │
│  │                                                       │   │
│  │  [Run Batch Analysis]                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Results Table                                        │   │
│  │  ┌──────┬────────┬──────────┬────────┬──────────┐   │   │
│  │  │ #    │ Result │ Conf.    │ Model  │ Download │   │   │
│  │  ├──────┼────────┼──────────┼────────┼──────────┤   │   │
│  │  │ 1/50 │ 🔴 Mal │ 98.2%   │ PSO-3  │ [PDF]    │   │   │
│  │  │ 2/50 │ 🟢 Ben │ 99.1%   │ PSO-3  │ [PDF]    │   │   │
│  │  │ ...  │  ...   │  ...    │  ...   │  ...     │   │   │
│  │  └──────┴────────┴──────────┴────────┴──────────┘   │   │
│  │  [Export All as CSV] [Export All as PDF]             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### Screen 4: Model Comparison Dashboard (Research Mode)

```
┌─────────────────────────────────────────────────────────────┐
│  Model Comparison Dashboard                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Performance Comparison                               │   │
│  │                                                       │   │
│  │  ┌────────┬────────┬────────┬────────┬────────┐     │   │
│  │  │ Method │ Acc    │ AUC    │ F1     │ Recall │     │   │
│  │  ├────────┼────────┼────────┼────────┼────────┤     │   │
│  │  │ PSO-W  │ 98.22% │ 0.9988 │ 0.9827 │ 0.9820 │     │   │
│  │  │ Stack  │ 98.15% │ 0.9985 │ 0.9820 │ 0.9796 │     │   │
│  │  │ Soft V │ 97.70% │ 0.9981 │ 0.9777 │ 0.9796 │     │   │
│  │  │ Hard V │ 96.99% │ 0.9698 │ 0.9709 │ 0.9742 │     │   │
│  │  └────────┴────────┴────────┴────────┴────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────┐  ┌────────────────────────┐    │
│  │  ROC Curves            │  │  Confusion Matrices    │    │
│  │  [Interactive plot]    │  │  [Interactive grid]    │    │
│  └────────────────────────┘  └────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PSO Weight Distribution                              │   │
│  │  [Bar chart: EfficientNet 38.3%, VGG16 28.9%, ...]   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

### Phase 3: Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BREASTVISIONAI UI ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FRONTEND (React + TailwindCSS)                             │    │
│  │                                                              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │ Dashboard│ │ Diagnosis│ │ Batch    │ │ Model Compare│  │    │
│  │  │ Page     │ │ Page     │ │ Page     │ │ Page         │  │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │    │
│  │       │            │            │              │           │    │
│  │  ┌────▼────────────▼────────────▼──────────────▼───────┐   │    │
│  │  │  Shared Components                                   │   │    │
│  │  │  - ImageUploader (drag & drop, DICOM support)       │   │    │
│  │  │  - ResultCard (prediction display)                  │   │    │
│  │  │  - ConfidenceBar (animated progress)                │   │    │
│  │  │  - ModelSelector (checkbox/dropdown)                │   │    │
│  │  │  - MetricsTable (sortable, filterable)              │   │    │
│  │  │  - ROCPlot (interactive chart.js/recharts)          │   │    │
│  │  │  - HeatmapOverlay (Grad-CAM visualization)          │   │    │
│  │  │  - ReportGenerator (PDF download)                   │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  │                                                              │    │
│  │  ┌─────────────────────────────────────────────────────┐   │    │
│  │  │  State Management (React Context / Zustand)         │   │    │
│  │  │  - PredictionHistory (recent results)               │   │    │
│  │  │  - ModelConfig (selected models, weights)           │   │    │
│  │  │  - UserPreferences (theme, defaults)                │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  API LAYER (Django REST Framework)                          │    │
│  │                                                              │    │
│  │  POST /api/predict/          → Single image prediction      │    │
│  │  POST /api/predict/batch/    → Batch prediction             │    │
│  │  GET  /api/models/           → List available models        │    │
│  │  GET  /api/models/{id}/      → Model details + metrics      │    │
│  │  GET  /api/ensemble/         → Ensemble config + weights    │    │
│  │  GET  /api/history/          → Prediction history            │    │
│  │  GET  /api/health/           → System status (exists ✓)     │    │
│  │  POST /api/upload/           → Upload image to temp storage │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  BACKEND (Django + TensorFlow)                              │    │
│  │                                                              │    │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐    │    │
│  │  │  Prediction Engine│  │  Model Registry              │    │    │
│  │  │  - Load 3 models  │  │  - EfficientNetB0           │    │    │
│  │  │  - Preprocess     │  │  - VGG16                    │    │    │
│  │  │  - PSO-Weighted   │  │  - ResNet50                 │    │    │
│  │  │  - Return result  │  │  - PSO weights: [0.383,...] │    │    │
│  │  └──────────────────┘  └──────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: API Endpoints Needed

| Endpoint              | Method | Purpose                  | Request                      | Response                                             |
| --------------------- | ------ | ------------------------ | ---------------------------- | ---------------------------------------------------- |
| `/api/predict/`       | POST   | Single image prediction  | `multipart: image file`      | `{prediction, confidence, model_breakdown, heatmap}` |
| `/api/predict/batch/` | POST   | Batch prediction         | `multipart: multiple images` | `{results: [{id, prediction, confidence}]}`          |
| `/api/models/`        | GET    | List available models    | —                            | `[{name, architecture, accuracy, auc, weight}]`      |
| `/api/ensemble/`      | GET    | Ensemble configuration   | —                            | `{method, models, weights, metrics}`                 |
| `/api/history/`       | GET    | Prediction history       | `?page=1&limit=20`           | `{results: [...], total, page}`                      |
| `/api/history/{id}/`  | GET    | Single prediction detail | —                            | `{image, prediction, breakdown, timestamp}`          |
| `/api/upload/`        | POST   | Upload image (temp)      | `multipart: image file`      | `{file_id, filename, preview_url}`                   |
| `/api/report/{id}/`   | GET    | Download PDF report      | —                            | `application/pdf`                                    |

---

### Phase 5: Key UX Considerations

#### 1. Medical-Grade UX Requirements

- **High contrast mode** for clinical environments (bright monitors)
- **Large touch targets** for use with gloves or on tablets
- **Minimal clicks to diagnosis** — 3 clicks max: Upload → Confirm → View Result
- **Clear confidence communication** — Use color-coded system:
  - 🟢 Green: >95% confidence (high)
  - 🟡 Yellow: 80-95% (moderate)
  - 🔴 Red: <80% (low — flag for human review)
- **DICOM support** — Must handle medical imaging formats natively

#### 2. Performance Requirements

- **Prediction time**: <5 seconds per image (3 models × ~1.5s each)
- **Batch processing**: Show progress bar with ETA for each image
- **Image preview**: Client-side thumbnail generation before upload
- **Caching**: Cache model weights in memory, not reloaded per request
- **Lazy loading**: Model comparison page loads charts asynchronously

#### 3. Accessibility

- **WCAG 2.1 AA compliance** — color contrast, keyboard navigation, screen reader support
- **ARIA labels** on all interactive elements
- **Focus management** for modal dialogs (report download, error states)
- **Reduced motion** option for animations

#### 4. Error Handling UX

- **Upload errors**: Show specific error (wrong format, too large, corrupted)
- **Model errors**: Graceful degradation — if one model fails, fall back to remaining 2
- **Network errors**: Retry with exponential backoff, show cached results if available
- **Empty states**: Illustrations + clear CTAs for empty history, no results yet

#### 5. Mobile Responsiveness

- **Tablet-first design** (radiologists use iPads)
- **Responsive grid**: 2-column on desktop, 1-column on mobile
- **Touch-optimized**: Swipeable result cards, pinch-to-zoom on images
- **Offline capability**: Service worker for cached model info and recent results

---

### Phase 6: Implementation Roadmap

| Phase                   | Duration | Deliverables                                                         | Dependencies                  |
| ----------------------- | -------- | -------------------------------------------------------------------- | ----------------------------- |
| **P0: Foundation**      | Week 1   | React project setup, TailwindCSS, routing, API client, health check  | Django backend (exists)       |
| **P1: Core Diagnosis**  | Week 2   | Image upload, single prediction, result display, confidence bars     | P0 + `/api/predict/` endpoint |
| **P2: Batch & History** | Week 3   | Batch upload, progress tracking, history page, pagination            | P1 + `/api/predict/batch/`    |
| **P3: Research Tools**  | Week 4   | Model comparison dashboard, ROC plots, confusion matrices            | P2 + `/api/models/` endpoint  |
| **P4: Polish**          | Week 5   | PDF reports, heatmap overlay, DICOM viewer, accessibility audit      | P3                            |
| **P5: Production**      | Week 6   | Performance optimization, error handling, mobile testing, deployment | P4                            |

---

### Phase 7: Technology Stack Recommendation

| Layer                | Technology                    | Why                                                   |
| -------------------- | ----------------------------- | ----------------------------------------------------- |
| **Framework**        | React 18 + Vite               | Fast dev, excellent DX, large ecosystem               |
| **Styling**          | TailwindCSS 3                 | Utility-first, rapid prototyping, consistent design   |
| **Charts**           | Recharts                      | React-native, responsive, good for medical charts     |
| **Image handling**   | react-dropzone + DICOM parser | Drag-drop upload, DICOM metadata extraction           |
| **State management** | Zustand                       | Lightweight, no boilerplate, persists to localStorage |
| **Routing**          | React Router v6               | Standard, lazy loading, nested routes                 |
| **HTTP client**      | Axios                         | Interceptors for error handling, progress tracking    |
| **PDF generation**   | jsPDF + html2canvas           | Client-side report generation                         |
| **Animation**        | Framer Motion                 | Smooth transitions, gesture support                   |
| **Testing**          | Vitest + Playwright           | Fast unit tests + E2E testing                         |

---

### Summary

The project needs a **complete frontend build** — currently there is zero UI. The recommended approach is a **React + TailwindCSS SPA** that communicates with the existing Django backend via REST APIs. The design should be **medical-grade**: high contrast, minimal clicks, clear confidence communication, DICOM support, and tablet-optimized. The 4 core screens (Dashboard, Diagnosis, Batch, Model Compare) map directly to the 3 user personas (Radiologist, Researcher, Admin). The PSO-discovered optimal 3-model ensemble should be the **default recommendation** with a simple toggle to explore other methods.
