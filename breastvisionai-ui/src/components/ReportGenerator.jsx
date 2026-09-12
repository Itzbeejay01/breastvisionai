import { apiHelpers } from "../api/client";

export default function ReportGenerator({
  predictionId,
  filename = "report",
  className,
}) {
  const downloadPdf = () => {
    const url = apiHelpers.reportUrl(predictionId);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener noreferrer";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <button
      type="button"
      onClick={downloadPdf}
      className={`flex items-center gap-2 px-4 py-2 touch-target ${
        className ||
        "bg-sky-600 text-white text-sm font-medium rounded-md hover:bg-sky-700 transition-colors"
      }`}
    >
      <span className="material-symbols-outlined text-[20px]">download</span>
      Download PDF Report
    </button>
  );
}
