function confidenceColor(confidence) {
  if (confidence >= 0.95) return "bg-benign";
  if (confidence >= 0.8) return "bg-moderate";
  return "bg-malignant";
}

function confidenceLabel(confidence) {
  if (confidence >= 0.95) return "High confidence";
  if (confidence >= 0.8) return "Moderate confidence";
  return "Low confidence";
}

export default function ConfidenceBar({ confidence }) {
  const pct = Math.round(confidence * 100);
  const barColor = confidenceColor(confidence);
  return (
    <div className="w-full">
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-sm font-medium text-gray-700">Confidence</span>
        <span className="text-sm font-semibold text-gray-900">{pct}%</span>
      </div>
      <div className="h-4 w-full bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-1 text-xs text-gray-500">{confidenceLabel(confidence)}</div>
    </div>
  );
}

export { confidenceColor, confidenceLabel, ConfidenceBar };
