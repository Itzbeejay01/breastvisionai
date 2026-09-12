import { useRef, useState } from "react";
import useStore, { IMAGE_TYPES } from "../store/index";
import Footer from "../components/Footer";
import { verdictLabel } from "../utils/verdict";

export default function BatchPage() {
  const [files, setFiles] = useState([]);
  const [imageType, setImageType] = useState(IMAGE_TYPES.RAW);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const { predictBatch } = useStore();

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList || []).filter(
      (file) => !files.some((f) => f.file.name === file.name)
    );
    if (incoming.length === 0) return;
    setFiles((prev) => [
      ...prev,
      ...incoming.map((file) => ({
        file,
        preview: URL.createObjectURL(file),
      })),
    ]);
  };

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.file.name !== name));
  };

  const runBatch = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await predictBatch(files.map((f) => f.file), imageType);
      setResults(res);
    } catch (e) {
      setError(e.message || "Batch prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = () => {
    if (!results) return;
    const rows = results.results.map((r, i) => [
      r.uploaded_image?.filename || `image_${i + 1}`,
      r.prediction_result || r.prediction,
      (r.confidence ?? 0).toFixed(4),
      (r.fused_probability ?? 0).toFixed(4),
    ]);
    const header = "filename,prediction,confidence,fused_probability\n";
    const csv = header + rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "batch_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const summary = results?.summary;
  const resultRows = results?.results || [];
  const doneCount = resultRows.length;

  return (
    <main className="batch-page flex-grow pt-28 pb-12 px-5 md:px-10 max-w-container-max mx-auto w-full flex flex-col gap-7">
      {/* Header & Upload */}
      <section className="spring-enter">
        <div
          className={`glass-light rounded-xl p-8 border-2 border-dashed transition-colors cursor-pointer flex flex-col items-center justify-center min-h-[200px] text-center ${
            dragOver
              ? "border-primary bg-primary-container/10"
              : "border-outline-variant hover:border-primary"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            addFiles(e.dataTransfer.files);
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <span className="material-symbols-outlined text-4xl text-primary mb-4">
            upload_file
          </span>
          <h2 className="font-headline-sm text-headline-sm text-on-surface mb-2">
            Drag &amp; Drop Batch Files
          </h2>
          <p className="font-body-sm text-body-sm text-on-surface-variant">
            Support for PNG, JPEG, and DICOM formats. Up to 500 files.
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.dcm"
            multiple
            className="hidden"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="mt-6 bg-primary-container text-on-primary-container px-6 py-2 rounded-full font-label-caps text-label-caps hover:bg-primary hover:text-on-primary transition-colors shadow-sm touch-target"
          >
            Browse Files
          </button>
        </div>
      </section>

      {/* Main Workspace: Queue & Results Side-by-Side on large screens */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter flex-grow">
        {/* Left Panel: Queue & Progress (4 cols) */}
        <aside className="lg:col-span-4 flex flex-col gap-gutter spring-enter" style={{ animationDelay: "0.1s" }}>
          {/* Action / Progress Card */}
          <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm border border-outline-variant/30 flex flex-col gap-4">
            <button
              type="button"
              onClick={runBatch}
              disabled={loading || files.length === 0}
              className="w-full bg-primary text-on-primary py-3 rounded-lg font-headline-sm text-headline-sm hover:bg-primary-container hover:text-on-primary-container transition-colors shadow-md flex justify-center items-center gap-2 touch-target disabled:opacity-50"
            >
              <span className="material-symbols-outlined">play_arrow</span>
              {loading ? "Analyzing…" : "Run Batch Analysis"}
            </button>
            {loading && (
              <div className="mt-1">
                <div className="flex justify-between mb-2">
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    Analyzing…
                  </span>
                  <span className="font-data-num text-data-num text-primary">
                    {doneCount}/{files.length}
                  </span>
                </div>
                <div className="w-full bg-surface-variant rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-500 ease-out"
                    style={{
                      width: `${files.length ? (doneCount / files.length) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            )}
            {error && (
              <p className="font-body-sm text-body-sm text-error">⚠️ {error}</p>
            )}
          </div>

          {/* Queue List */}
          <div className="bg-surface-container-lowest rounded-xl shadow-sm border border-outline-variant/30 flex-grow flex flex-col overflow-hidden">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center bg-surface-bright">
              <h3 className="font-headline-sm text-headline-sm text-on-surface">
                Queue ({files.length})
              </h3>
              <div className="flex bg-surface-variant rounded-lg p-1">
                {[
                  { value: IMAGE_TYPES.RAW, label: "Raw" },
                  { value: IMAGE_TYPES.PROCESSED, label: "Pre-processed" },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setImageType(opt.value)}
                    className={`px-3 py-1 rounded font-label-caps text-label-caps transition-colors touch-target ${
                      imageType === opt.value
                        ? "bg-surface-container-lowest shadow-sm text-primary"
                        : "text-on-surface-variant hover:text-on-surface"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {files.length === 0 ? (
              <div className="flex-grow flex flex-col items-center justify-center gap-2 py-12 text-center px-6">
                <span className="material-symbols-outlined text-4xl text-outline-variant">
                  queue
                </span>
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Drop files above to build your analysis queue.
                </p>
              </div>
            ) : (
              <div className="overflow-y-auto max-h-[600px] flex flex-col">
                {files.map((f) => (
                  <div
                    key={f.file.name}
                    className="p-3 border-b border-outline-variant/20 flex items-center gap-3 hover:bg-surface-bright transition-colors group"
                  >
                    <img
                      src={f.preview}
                      alt={f.file.name}
                      className="w-12 h-12 rounded object-cover border border-outline-variant/30"
                    />
                    <div className="flex-grow min-w-0">
                      <p className="font-body-sm text-body-sm text-on-surface truncate">
                        {f.file.name}
                      </p>
                      <p className="font-label-caps text-label-caps text-on-surface-variant mt-1">
                        Pending
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(f.file.name)}
                      className="p-1 text-on-surface-variant hover:text-error opacity-0 group-hover:opacity-100 transition-opacity touch-target"
                      aria-label={`Remove ${f.file.name}`}
                    >
                      <span className="material-symbols-outlined text-sm">
                        close
                      </span>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* Right Panel: Results & Stats (8 cols) */}
        <section className="lg:col-span-8 flex flex-col gap-gutter spring-enter" style={{ animationDelay: "0.2s" }}>
          {/* Stats Summary Bento */}
          {summary ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-surface-container-lowest rounded-xl p-5 shadow-sm border border-outline-variant/30 flex flex-col justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  Total Analyzed
                </span>
                <div className="font-display-lg text-display-lg text-on-surface mt-2">
                  {summary.total}
                </div>
              </div>
              <div className="bg-error-container/20 rounded-xl p-5 shadow-sm border border-error/20 flex flex-col justify-between">
                <span className="font-label-caps text-label-caps text-tertiary flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm">
                    warning
                  </span>
                  Malignant
                </span>
                <div className="font-display-lg text-display-lg text-tertiary mt-2">
                  {summary.malignant_count}
                </div>
              </div>
              <div className="bg-secondary-container/20 rounded-xl p-5 shadow-sm border border-secondary/20 flex flex-col justify-between">
                <span className="font-label-caps text-label-caps text-secondary flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm">
                    check_circle
                  </span>
                  Non-Malignant
                </span>
                <div className="font-display-lg text-display-lg text-secondary mt-2">
                  {summary.benign_count}
                </div>
              </div>
              <div className="bg-primary-container/10 rounded-xl p-5 shadow-sm border border-primary/20 flex flex-col justify-between">
                <span className="font-label-caps text-label-caps text-primary">
                  Avg Confidence
                </span>
                <div className="font-display-lg text-display-lg text-primary mt-2">
                  {(summary.avg_confidence * 100).toFixed(0)}
                  <span className="text-headline-sm">%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {["Total Analyzed", "Malignant", "Non-Malignant", "Avg Confidence"].map(
                (label) => (
                  <div
                    key={label}
                    className="bg-surface-container-lowest rounded-xl p-5 shadow-sm border border-outline-variant/30 flex flex-col justify-between"
                  >
                    <span className="font-label-caps text-label-caps text-on-surface-variant">
                      {label}
                    </span>
                    <div className="font-display-lg text-display-lg text-on-surface/20 mt-2">
                      —
                    </div>
                  </div>
                )
              )}
            </div>
          )}

          {/* Results Table */}
          <div className="bg-surface-container-lowest rounded-xl shadow-sm border border-outline-variant/30 flex-grow flex flex-col overflow-hidden">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center bg-surface-bright">
              <h3 className="font-headline-sm text-headline-sm text-on-surface">
                Analysis Results
              </h3>
              <button
                type="button"
                onClick={exportCsv}
                disabled={!results}
                className="flex items-center gap-2 border border-outline text-on-surface px-4 py-1.5 rounded-full font-label-caps text-label-caps hover:bg-surface-variant transition-colors touch-target disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-[18px]">
                  download
                </span>
                Export CSV
              </button>
            </div>

            {resultRows.length === 0 ? (
              <div className="flex-grow flex flex-col items-center justify-center gap-2 py-16 text-center px-6">
                <span className="material-symbols-outlined text-4xl text-outline-variant">
                  table_chart
                </span>
                <p className="font-body-sm text-body-sm text-on-surface-variant">
                  Run the batch analysis to see results here.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-outline-variant/30 bg-surface-container-low/50">
                      <th className="p-3 font-label-caps text-label-caps text-on-surface-variant font-medium w-12">
                        Scan
                      </th>
                      <th className="p-3 font-label-caps text-label-caps text-on-surface-variant font-medium">
                        Filename
                      </th>
                      <th className="p-3 font-label-caps text-label-caps text-on-surface-variant font-medium">
                        Verdict
                      </th>
                      <th className="p-3 font-label-caps text-label-caps text-on-surface-variant font-medium w-48">
                        Confidence
                      </th>
                      <th className="p-3 font-label-caps text-label-caps text-on-surface-variant font-medium text-right">
                        Fused Prob
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultRows.map((r, i) => {
                      const verdict =
                        r.prediction_result || r.prediction || "n/a";
                      const isMalignant = verdict === "malignant";
                      return (
                        <tr
                          key={r.id ?? i}
                          className="border-b border-outline-variant/20 hover:bg-surface-bright transition-colors"
                        >
                          <td className="p-3">
                            {r.uploaded_image?.preview_url ? (
                              <img
                                src={r.uploaded_image.preview_url}
                                alt={r.uploaded_image.filename}
                                className="w-10 h-10 rounded object-cover"
                              />
                            ) : (
                              <div className="w-10 h-10 rounded bg-surface-container flex items-center justify-center">
                                <span className="material-symbols-outlined text-on-surface-variant">
                                  image
                                </span>
                              </div>
                            )}
                          </td>
                          <td className="p-3 font-body-sm text-body-sm text-on-surface font-medium">
                            {r.uploaded_image?.filename || `image_${i + 1}`}
                          </td>
                          <td className="p-3">
                            <span
                              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full font-label-caps text-label-caps border ${
                                isMalignant
                                  ? "bg-error-container/20 text-tertiary border-error/20"
                                  : "bg-secondary-container/20 text-secondary border-secondary/20"
                              }`}
                            >
                              <span className="material-symbols-outlined text-[14px]">
                                {isMalignant ? "warning" : "check_circle"}
                              </span>
                              {verdictLabel(verdict)}
                            </span>
                          </td>
                          <td className="p-3">
                            <div className="flex items-center gap-2">
                              <div className="flex-grow bg-surface-variant h-1.5 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    isMalignant ? "bg-tertiary" : "bg-secondary"
                                  }`}
                                  style={{
                                    width: `${(r.confidence ?? 0) * 100}%`,
                                  }}
                                />
                              </div>
                              <span className="font-data-num text-data-num text-on-surface text-xs">
                                {((r.confidence ?? 0) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="p-3 font-data-num text-data-num text-on-surface text-right">
                            {(r.fused_probability ?? 0).toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Footer */}
      <Footer />
    </main>
  );
}
