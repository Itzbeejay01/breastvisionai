import ConfidenceBar from "./ConfidenceBar";
import { verdictLabel } from "../utils/verdict";

export default function ResultCard({ result }) {
  if (!result) return null;

  const prediction = result.prediction_result || result.prediction || "n/a";
  const isMalignant = prediction === "malignant";
  const confidence = result.confidence ?? 0;
  const fused = result.fused_probability ?? 0;
  const probability = result.prediction_probability ?? 0;

  const badge = isMalignant
    ? "bg-malignant text-white"
    : "bg-benign text-white";

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-3xl" role="img" aria-label="diagnosis">
            {isMalignant ? "⚠️" : "✅"}
          </span>
          <div>
            <h3 className="text-sm text-gray-500">Diagnosis</h3>
            <p className={`inline-block px-3 py-1 rounded-full font-semibold ${badge}`}>
              {verdictLabel(prediction)}
            </p>
          </div>
        </div>
        <div className="text-right">
          <h3 className="text-sm text-gray-500">Malignancy probability</h3>
          <p className="text-2xl font-bold">{Math.round(probability * 100)}%</p>
        </div>
      </div>

      <ConfidenceBar confidence={confidence} />

      <div className="grid grid-cols-2 gap-2 text-sm">
        <Stat label="Confidence" value={`${Math.round(confidence * 100)}%`} />
        <Stat label="Fused probability" value={fused.toFixed(4)} />
      </div>

      {result.ensemble_method && (
        <p className="text-xs text-gray-500">
          Ensemble: {result.ensemble_method}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-gray-50 rounded p-2">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="ml-2 text-gray-900 font-medium">{value}</span>
    </div>
  );
}
